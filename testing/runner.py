"""
runner.py - Run inference results against test cases using Docker.

One container per language is started at the beginning, kept alive for all
test cases of that language, then stopped at the end. Code is copied into
the running container via the Docker API — no restarts between test cases.
Languages are evaluated in parallel (one thread per language); test cases
within a language run sequentially to avoid file conflicts in the container.

Usage:
    python3 testing/runner.py --results results.jsonl
"""

import atexit
import io
import json
import re
import signal
import sys
import tarfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from tqdm import tqdm

from util import dataset
from testing.languages import LANGUAGES
from testing.sanitize import sanitize

TIMEOUT = 30  # seconds per test case
_docker = docker.from_env()

# Global container registry — used by atexit/signal handler to ensure cleanup
_live_containers: list = []
_live_containers_lock = threading.Lock()


def _register(container) -> None:
    with _live_containers_lock:
        _live_containers.append(container)


def _deregister(container) -> None:
    with _live_containers_lock:
        try:
            _live_containers.remove(container)
        except ValueError:
            pass


def _cleanup_all() -> None:
    with _live_containers_lock:
        for c in list(_live_containers):
            try:
                c.stop()
            except Exception:
                pass
        _live_containers.clear()


atexit.register(_cleanup_all)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def extract_code(completion: str) -> str:
    match = re.search(r"```(?:\w+)?\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    # rstrip only: preserve leading whitespace (e.g. 4-space indent on Python bodies)
    return completion.rstrip()


def _write_file_to_container(container, filename: str, source: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        encoded = source.encode()
        info = tarfile.TarInfo(name=filename)
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
    buf.seek(0)
    container.put_archive("/tmp", buf)


def _pull_with_progress(lang: str, image: str) -> None:
    """Pull image with a byte-level progress bar. No-op if already cached."""
    try:
        _docker.images.get(image)
        tqdm.write(f"  {lang:<14} ✓ cached ({image})")
        return
    except docker.errors.ImageNotFound:
        pass

    layer_total: dict = {}
    layer_done: dict = {}

    with tqdm(desc=f"  {lang:<14}", unit="B", unit_scale=True,
              unit_divisor=1024, dynamic_ncols=True, leave=False) as pbar:
        for event in _docker.api.pull(image, stream=True, decode=True):
            detail = event.get("progressDetail", {})
            lid = event.get("id")
            if not lid:
                continue
            total = detail.get("total")
            current = detail.get("current", 0)
            if total and lid not in layer_total:
                layer_total[lid] = total
                pbar.total = sum(layer_total.values())
                pbar.refresh()
            if current:
                prev = layer_done.get(lid, 0)
                pbar.update(current - prev)
                layer_done[lid] = current
    tqdm.write(f"  {lang:<14} ✓ pulled  ({image})")


def _start_container(lang: str, config: dict):
    """Pull image if needed, start a container. Returns (container | None, error_str | None)."""
    try:
        _pull_with_progress(lang, config["image"])
    except Exception as e:
        return None, f"image pull failed: {e}"

    # Remove any stale container with the same name from a previous interrupted run
    container_name = f"evalrunner-{lang}"
    try:
        stale = _docker.containers.get(container_name)
        stale.remove(force=True)
    except docker.errors.NotFound:
        pass

    mem = config.get("mem_limit", "256m")
    try:
        container = _docker.containers.run(
            image=config["image"],
            name=container_name,
            command="sleep infinity",
            detach=True,
            auto_remove=True,
            mem_limit=mem,
        )
        _register(container)
    except docker.errors.ImageNotFound:
        return None, f"image not found: {config['image']}"
    except Exception as e:
        return None, str(e)

    if config["setup_cmd"]:
        result = container.exec_run(
            ["sh", "-c", config["setup_cmd"]], workdir="/tmp",
        )
        if result.exit_code != 0:
            output = result.output.decode(errors="replace").strip()
            tqdm.write(f"  {lang:<14} ✗ setup FAILED (exit {result.exit_code}):\n{output[:400]}")
            return None, f"setup failed (exit {result.exit_code})"

    return container, None


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def _run_language(lang: str, records: list, container,
                  write_lock: threading.Lock, f_results,
                  log_dir: Path, pbar: tqdm) -> dict:
    """Run all test cases for one language. Returns a summary dict."""
    config = LANGUAGES[lang]
    summary = {"lang": lang, "total": len(records), "passed": 0, "failed": 0, "skipped": 0}

    with open(log_dir / f"{lang}.log", "w") as log:
        def logline(msg: str):
            log.write(msg + "\n")
            log.flush()

        for rec, problem in records:
            task_id = rec["task_id"]
            model = rec.get("chosen_model", "?")
            extracted = extract_code(rec["completion"])

            if container is None:
                summary["skipped"] += 1
                logline(f"[SKIP] {task_id} — no container")
                with write_lock:
                    f_results.write(json.dumps({"task_id": task_id, "passed": False, "extracted_code": extracted}) + "\n")
                pbar.update(1)
                continue

            try:
                body, use_prompt = sanitize(extracted, problem.entry_point, lang)
                prompt = problem.prompt if use_prompt else ""
                source = config["assemble"](prompt, body, problem.test, problem.entry_point)
                _write_file_to_container(container, config["filename"], source)

                result = container.exec_run(
                    ["sh", "-c", f"timeout {TIMEOUT} {config['run_cmd']}"],
                    workdir="/tmp",
                )
                passed = result.exit_code == 0
                output = result.output.decode().strip()

                status = "PASS" if passed else "FAIL"
                logline(f"\n{'='*60}")
                logline(f"[{status}] {task_id} ({model})")
                logline(f"--- extracted body ---\n{body}")
                logline(f"--- output ---\n{output}" if output else "--- output --- (none)")

                summary["passed" if passed else "failed"] += 1
                with write_lock:
                    f_results.write(json.dumps({"task_id": task_id, "passed": passed, "extracted_code": extracted}) + "\n")

            except docker.errors.NotFound:
                tqdm.write(f"[{lang}] container died — skipping remaining {len(records) - records.index((rec, problem)) - 1} case(s)")
                logline(f"CONTAINER DIED on {task_id} — skipping remaining cases")
                summary["skipped"] += 1
                container = None
                with write_lock:
                    f_results.write(json.dumps({"task_id": task_id, "passed": False, "extracted_code": extracted}) + "\n")

            except Exception as e:
                tqdm.write(f"[{lang}] ERROR on {task_id}: {e}")
                logline(f"ERROR on {task_id}: {e}")
                summary["failed"] += 1
                with write_lock:
                    f_results.write(json.dumps({"task_id": task_id, "passed": False, "extracted_code": extracted}) + "\n")

            pbar.update(1)

    _deregister(container)
    try:
        container.stop()
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tests(results_path: Path) -> None:
    problems = {p.task_id: p for p in dataset.load(split="all")}

    by_lang: dict[str, list] = defaultdict(list)
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            problem = problems.get(rec["task_id"])
            if problem:
                by_lang[problem.programming_language].append((rec, problem))

    total = sum(len(v) for v in by_lang.values())
    langs_sorted = sorted(by_lang)

    w = 60
    print("=" * w)
    print(f"  {total} problems  ·  {len(by_lang)} languages: {', '.join(langs_sorted)}")
    print("=" * w)

    # ── Phase 1a: pull images sequentially so bars don't overlap ────────────
    print("\n── Pulling images ──────────────────────────────────────")
    needed_images = {LANGUAGES[l]["image"] for l in langs_sorted if l in LANGUAGES}
    # deduplicate: only pull each image once, but report for all langs sharing it
    pulled: set = set()
    for lang in langs_sorted:
        if lang not in LANGUAGES:
            continue
        image = LANGUAGES[lang]["image"]
        if image not in pulled:
            try:
                _pull_with_progress(lang, image)
            except Exception as e:
                tqdm.write(f"  {lang:<14} ✗ pull failed: {e}")
            pulled.add(image)

    # ── Phase 1b: start containers in parallel ───────────────────────────────
    print("\n── Starting containers ─────────────────────────────────")
    containers: dict[str, object] = {}

    def start_one(lang):
        if lang not in LANGUAGES:
            return lang, None, "no runner configured"
        return lang, *_start_container(lang, LANGUAGES[lang])

    with ThreadPoolExecutor(max_workers=len(langs_sorted)) as pool:
        futs = {pool.submit(start_one, lang): lang for lang in langs_sorted}
        for fut in as_completed(futs):
            lang, container, err = fut.result()
            containers[lang] = container
            if container is None:
                tqdm.write(f"  {lang:<14} ✗ {err}")
            elif err:
                tqdm.write(f"  {lang:<14} ⚠ {err}")
            else:
                tqdm.write(f"  {lang:<14} ✓ ready")
    print()

    # ── Phase 2: evaluate with per-language progress bars ───────────────────
    print("── Evaluating ──────────────────────────────────────────")
    testing_results_path = results_path.parent / "testing_results.jsonl"
    log_dir = results_path.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    write_lock = threading.Lock()

    pbars = {
        lang: tqdm(total=len(by_lang[lang]), desc=f"{lang:<14}", position=i, leave=True)
        for i, lang in enumerate(langs_sorted)
    }

    summaries = []
    with open(testing_results_path, "w") as f_results:
        with ThreadPoolExecutor(max_workers=len(langs_sorted)) as pool:
            futures = {
                pool.submit(
                    _run_language, lang, by_lang[lang], containers.get(lang),
                    write_lock, f_results, log_dir, pbars[lang]
                ): lang
                for lang in langs_sorted
            }
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as e:
                    tqdm.write(f"[{lang}] ERROR: unexpected thread failure: {e}")

    for pbar in pbars.values():
        pbar.close()

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'LANGUAGE':<14} {'PASS':>6} {'FAIL':>6} {'SKIP':>6} {'TOTAL':>6}")
    print("-" * 50)
    total_pass = total_fail = total_skip = 0
    for s in sorted(summaries, key=lambda x: x["lang"]):
        print(f"{s['lang']:<14} {s['passed']:>6} {s['failed']:>6} {s['skipped']:>6} {s['total']:>6}")
        total_pass += s["passed"]
        total_fail += s["failed"]
        total_skip += s["skipped"]
    print("-" * 50)
    print(f"{'TOTAL':<14} {total_pass:>6} {total_fail:>6} {total_skip:>6} {total:>6}")
    print("=" * 50)


