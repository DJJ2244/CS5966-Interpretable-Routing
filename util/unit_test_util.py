"""
unit_test_util.py - Multi-language Docker-based code evaluation.

Merged from testing/languages.py, testing/sanitize.py, and testing/runner.py.

Main entry point:
    run_tests(results_path: Path, model: str) -> None
"""

import ast
import atexit
import io
import re
import signal
import sys
import tarfile
import textwrap
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from tqdm import tqdm


# =============================================================================
# Language configs (from testing/languages.py)
# =============================================================================

def _python(prompt, inner, test, entry_point):
    return prompt + inner + "\n" + test + f"\ncheck({entry_point})"


def _brace(prompt, inner, test, entry_point):
    if prompt:
        return prompt + inner + "\n}\n" + test
    return inner + "\n" + test


def _java(prompt, inner, test, entry_point):
    return prompt + inner + "\n    }\n}\n" + test


def _scala(prompt, inner, test, entry_point):
    return prompt + inner + "\n    }\n" + test


def _ruby(prompt, inner, test, entry_point):
    if prompt:
        return prompt + "\n" + inner + "\nend\n" + test
    return inner + "\n" + test


def _perl(prompt, inner, test, entry_point):
    if prompt:
        return prompt + inner + "\n}\n" + test
    return inner + "\n" + test


def _php(prompt, inner, test, entry_point):
    return prompt + inner + "\n}\n" + test


def _csharp(prompt, inner, test, entry_point):
    return prompt + inner + "\n        }\n" + test


LANGUAGES: dict[str, dict] = {
    "python": {
        "image":     "python:3.11-alpine",
        "filename":  "solution.py",
        "setup_cmd": None,
        "run_cmd":   "python solution.py",
        "assemble":  _python,
    },
    "java": {
        "image":     "eclipse-temurin:21-jdk-alpine",
        "filename":  "Main.java",
        "setup_cmd": None,
        "run_cmd":   "javac Main.java && java Main",
        "assemble":  _java,
    },
    "javascript": {
        "image":     "node:20-alpine",
        "filename":  "solution.js",
        "setup_cmd": "npm install --silent lodash",
        "run_cmd":   "node solution.js",
        "assemble":  _brace,
    },
    "typescript": {
        "image":     "node:20-alpine",
        "filename":  "solution.ts",
        "setup_cmd": (
            "npm install --silent ts-node typescript @types/node @types/assert "
            "&& echo '{\"type\":\"commonjs\"}' > /tmp/package.json"
        ),
        "run_cmd":   "npx ts-node --transpile-only --skip-project solution.ts",
        "assemble":  _brace,
    },
    "go": {
        "image":     "golang:1.21-alpine",
        "filename":  "solution.go",
        "setup_cmd": None,
        "run_cmd":   "go run solution.go",
        "assemble":  _brace,
    },
    "kotlin": {
        "image":     "eclipse-temurin:17-jdk-jammy",
        "filename":  "solution.kt",
        "setup_cmd": "apt-get update -qq && apt-get install -yqq kotlin",
        "run_cmd":   "kotlinc solution.kt -include-runtime -d solution.jar && java -jar solution.jar",
        "assemble":  _brace,
        "mem_limit": "1g",
    },
    "scala": {
        "image":     "eclipse-temurin:11-jdk-jammy",
        "filename":  "solution.scala",
        "setup_cmd": "apt-get update -qq && apt-get install -yqq scala",
        "run_cmd":   "scala solution.scala",
        "assemble":  _scala,
        "mem_limit": "1g",
    },
    "ruby": {
        "image":     "ruby:3.2-slim",
        "filename":  "solution.rb",
        "setup_cmd": None,
        "run_cmd":   "ruby solution.rb",
        "assemble":  _ruby,
    },
    "php": {
        "image":     "php:8.2-cli",
        "filename":  "solution.php",
        "setup_cmd": None,
        "run_cmd":   "php solution.php",
        "assemble":  _php,
    },
    "swift": {
        "image":     "swift:5.9",
        "filename":  "solution.swift",
        "setup_cmd": None,
        "run_cmd":   "swift solution.swift",
        "assemble":  _brace,
    },
    "perl": {
        "image":     "perl:5.38-slim",
        "filename":  "solution.pl",
        "setup_cmd": (
            "apt-get update -qq && apt-get install -yqq build-essential "
            "&& cpanm --notest --force Clone File::Find::Rule Data::Compare"
        ),
        "run_cmd":   "perl solution.pl",
        "assemble":  _perl,
    },
    "csharp": {
        "image":     "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
        "filename":  "solution.cs",
        "setup_cmd": (
            "dotnet new console -o /tmp/csproject -f net8.0 --force "
            "&& cd /tmp/csproject "
            "&& dotnet add package CompareNETObjects "
            "&& dotnet build -c Release -o /tmp/csproject/bin"
        ),
        "run_cmd":   "cp /tmp/solution.cs /tmp/csproject/Program.cs && dotnet run --project /tmp/csproject -c Release",
        "assemble":  _csharp,
    },
}


# =============================================================================
# Sanitize (from testing/sanitize.py)
# =============================================================================

def sanitize(completion: str, entry_point: str, language: str) -> tuple:
    """Return (code, use_prompt).

    use_prompt=True  → code is the function body; assembler prepends the prompt.
    use_prompt=False → code is a self-contained function definition.
    """
    if language == "python":
        return _python_body(completion, entry_point), True
    if language in ("ruby", "perl"):
        return _keyword_body(completion, entry_point, language)
    return _brace_body(completion, entry_point), True


def _python_body(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == entry_point:
                lines = code.splitlines(keepends=True)
                body_lines = lines[node.body[0].lineno - 1 : node.end_lineno]
                return "".join(body_lines)
    except SyntaxError:
        pass

    lines = code.splitlines(keepends=True)
    lines = [l for l in lines if not (not l.startswith(' ') and l.lstrip().startswith('#'))]
    body_started = False
    truncated = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            body_started = True
        elif not body_started:
            truncated.append(line)
            continue
        if body_started and not line.startswith(' ') and re.match(r'(def|class)\s', line):
            break
        truncated.append(line)
    code = ''.join(truncated)
    dedented = textwrap.dedent(code)
    content_lines = [l for l in dedented.splitlines() if l.strip()]
    if content_lines:
        indents = [len(l) - len(l.lstrip()) for l in content_lines]
        if min(indents) == 0 and max(indents) > 0:
            code = textwrap.indent(dedented, '    ')
    return code


def _brace_body(code: str, entry_point: str) -> str:
    ep = re.escape(entry_point)
    pattern = re.compile(
        rf'(?:'
        rf'function\s+{ep}\s*[(<]'
        rf'|(?:const|let|var)\s+{ep}\s*='
        rf'|\b{ep}\s*[(<]'
        rf')',
        re.MULTILINE,
    )
    match = pattern.search(code)
    if match:
        brace_pos = code.find('{', match.end())
        if brace_pos != -1:
            return _extract_inner(code, brace_pos)

    stripped = code.lstrip()
    if stripped.startswith('{'):
        brace_pos = len(code) - len(stripped)
        return _extract_inner(code, brace_pos)

    depth = 0
    for i, ch in enumerate(code):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                return code[:i].rstrip()
    return code


def _extract_inner(code: str, brace_pos: int) -> str:
    depth = 0
    for i, ch in enumerate(code[brace_pos:], start=brace_pos):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return code[brace_pos + 1 : i]
    return code[brace_pos + 1:]


_RUBY_BLOCK_OPEN = re.compile(
    r'^(if|unless|while|until|for|begin|case|def|class|module)\b'
)
_RUBY_BLOCK_DO = re.compile(r'\bdo\s*(\|[^|]*\|)?\s*$')
_RUBY_BLOCK_END = re.compile(r'^end\b')


def _ruby_extract_function(code: str, entry_point: str) -> str:
    match = re.search(rf'\bdef\s+{re.escape(entry_point)}\b', code)
    if not match:
        return code
    start = code.rfind('\n', 0, match.start()) + 1
    lines = code[start:].splitlines(keepends=True)
    depth = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if _RUBY_BLOCK_OPEN.match(s):
            depth += 1
        elif _RUBY_BLOCK_DO.search(s):
            depth += 1
        elif _RUBY_BLOCK_END.match(s):
            depth -= 1
            if depth == 0:
                return ''.join(lines[:i + 1]).rstrip()
    return ''.join(lines).rstrip()


def _ruby_strip_closing_end(code: str) -> str:
    depth = 0
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        s = line.strip()
        if _RUBY_BLOCK_OPEN.match(s):
            depth += 1
        elif _RUBY_BLOCK_DO.search(s):
            depth += 1
        elif _RUBY_BLOCK_END.match(s):
            if depth == 0:
                return ''.join(lines[:i]).rstrip()
            depth -= 1
    return code.rstrip()


def _keyword_body(code: str, entry_point: str, language: str) -> tuple:
    if language == "ruby":
        if re.search(rf'\bdef\s+{re.escape(entry_point)}\b', code):
            return _ruby_extract_function(code, entry_point), False
        return _ruby_strip_closing_end(code), True
    elif language == "perl":
        match = re.search(rf'\bsub\s+{re.escape(entry_point)}\b', code)
        if match:
            brace_pos = code.find('{', match.end())
            if brace_pos != -1:
                line_start = code.rfind('\n', 0, match.start()) + 1
                depth = 0
                for i in range(brace_pos, len(code)):
                    if code[i] == '{':
                        depth += 1
                    elif code[i] == '}':
                        depth -= 1
                        if depth == 0:
                            return code[line_start:i + 1], False
            return code[match.start():], False
        depth = 0
        for i, ch in enumerate(code):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth < 0:
                    return code[:i].rstrip(), True
    return code, True


# =============================================================================
# Docker runner (from testing/runner.py)
# =============================================================================

TIMEOUT = 30  # seconds per test case
_docker = docker.from_env()

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


def extract_code(completion: str) -> str:
    match = re.search(r"```(?:\w+)?\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
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
    try:
        _pull_with_progress(lang, config["image"])
    except Exception as e:
        return None, f"image pull failed: {e}"

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


def _run_language(
    lang: str,
    records: list,
    container,
    log_dir: Path,
    pbar: tqdm,
) -> dict:
    from daos.model_task_result_dao import update_test_result

    config = LANGUAGES[lang]
    summary = {"lang": lang, "total": len(records), "passed": 0, "failed": 0, "skipped": 0}

    with open(log_dir / f"{lang}.log", "w") as log:
        def logline(msg: str):
            log.write(msg + "\n")
            log.flush()

        for row, problem in records:
            task_id = row.task_id
            extracted = extract_code(row.result or "")

            if container is None:
                summary["skipped"] += 1
                logline(f"[SKIP] {task_id} — no container")
                update_test_result(task_id, row.model_name, extracted, False)
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
                logline(f"[{status}] {task_id} ({row.model_name})")
                logline(f"--- extracted body ---\n{body}")
                logline(f"--- output ---\n{output}" if output else "--- output --- (none)")

                summary["passed" if passed else "failed"] += 1
                update_test_result(task_id, row.model_name, extracted, passed)

            except docker.errors.NotFound:
                tqdm.write(f"[{lang}] container died — skipping remaining cases")
                logline(f"CONTAINER DIED on {task_id} — skipping remaining cases")
                summary["skipped"] += 1
                container = None
                update_test_result(task_id, row.model_name, extracted, False)

            except Exception as e:
                tqdm.write(f"[{lang}] ERROR on {task_id}: {e}")
                logline(f"ERROR on {task_id}: {e}")
                summary["failed"] += 1
                update_test_result(task_id, row.model_name, extracted, False)

            pbar.update(1)

    _deregister(container)
    try:
        container.stop()
    except Exception:
        pass

    return summary


def run_tests(model_name: str, split_id: int, is_test: bool = False) -> None:
    """Evaluate untested inference results against Docker test cases for all languages.

    Reads pending rows (passed IS NULL) from model_task_result and writes
    extracted_code and passed back to the same table when done.

    Args:
        model_name: HuggingFace model ID.
        split_id:   DB split id.
        is_test:    Whether to evaluate the test partition (default: train).
    """
    from daos import tasks_dao, model_task_result_dao

    pending = model_task_result_dao.get_untested_for_model(model_name, split_id, is_test)
    if not pending:
        print("All tasks already tested, skipping.")
        return

    all_tasks = tasks_dao.get_all_for_split(split_id, is_test=is_test)
    problems = {t.id: t for t in all_tasks}

    by_lang: dict[str, list] = defaultdict(list)
    for row in pending:
        problem = problems.get(row.task_id)
        if problem:
            by_lang[problem.programming_language].append((row, problem))

    total = sum(len(v) for v in by_lang.values())
    langs_sorted = sorted(by_lang)

    w = 60
    print("=" * w)
    print(f"  {total} problems  ·  {len(by_lang)} languages: {', '.join(langs_sorted)}")
    print("=" * w)

    print("\n── Pulling images ──────────────────────────────────────")
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

    print("── Evaluating ──────────────────────────────────────────")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    pbars = {
        lang: tqdm(total=len(by_lang[lang]), desc=f"{lang:<14}", position=i, leave=True)
        for i, lang in enumerate(langs_sorted)
    }

    summaries = []
    with ThreadPoolExecutor(max_workers=len(langs_sorted)) as pool:
        futures = {
            pool.submit(
                _run_language, lang, by_lang[lang], containers.get(lang),
                log_dir, pbars[lang]
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
