"""
runner.py - Run inference results against test cases using Docker.

One container per language is started at the beginning, kept alive for all
test cases of that language, then stopped at the end. Code is copied into
the running container via the Docker API — no restarts between test cases.

Usage:
    python testing/runner.py --results results.jsonl --output eval.jsonl
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import docker

sys.path.insert(0, str(Path(__file__).parent.parent))
import dataset
from languages import LANGUAGES

TIMEOUT = 30  # seconds per test case
_docker = docker.from_env()


def _write_file_to_container(container, filename: str, source: str) -> None:
    """Copy source code into a running container via tar archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        encoded = source.encode()
        info = tarfile.TarInfo(name=filename)
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
    buf.seek(0)
    container.put_archive("/tmp", buf)


def _start_container(config: dict):
    """Start a persistent container and run any one-time setup commands."""
    container = _docker.containers.run(
        image=config["image"],
        command="sleep infinity",
        detach=True,
        auto_remove=True,
        mem_limit="256m",
    )
    if config["setup_cmd"]:
        result = container.exec_run(config["setup_cmd"], workdir="/tmp")
        if result.exit_code != 0:
            print(f"  Warning: setup failed: {result.output.decode()}")
    return container


def run_tests(results_path: Path, output_path: Path) -> None:
    problems = {p.task_id: p for p in dataset.load()}

    # Collect which languages are actually needed
    needed_langs = set()
    records = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            problem = problems.get(rec["task_id"])
            if problem:
                needed_langs.add(problem.programming_language)
                records.append((rec, problem))

    # Start one container per needed language
    containers = {}
    for lang in needed_langs:
        config = LANGUAGES.get(lang)
        if config is None:
            print(f"No runner configured for: {lang}")
            continue
        print(f"Starting container for {lang}...")
        containers[lang] = _start_container(config)
    print(f"Started {len(containers)} container(s)\n")

    try:
        with open(output_path, "w") as f_out:
            for rec, problem in records:
                lang = problem.programming_language
                config = LANGUAGES.get(lang)
                container = containers.get(lang)

                if container is None:
                    rec["passed"] = False
                    f_out.write(json.dumps(rec) + "\n")
                    continue

                source = config["assemble"](
                    problem.prompt, rec["completion"], problem.test, problem.entry_point
                )
                _write_file_to_container(container, config["filename"], source)

                run_cmd = config["run_cmd"]
                result = container.exec_run(
                    ["sh", "-c", f"timeout {TIMEOUT} {run_cmd}"],
                    workdir="/tmp",
                )
                passed = result.exit_code == 0
                rec["passed"] = passed
                f_out.write(json.dumps(rec) + "\n")
                print(f"[{'PASS' if passed else 'FAIL'}] {rec['task_id']}  ({rec.get('chosen_model', '?')})")
    finally:
        for lang, container in containers.items():
            container.stop()
        print("\nAll containers stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()
    run_tests(Path(args.results), Path(args.output))
