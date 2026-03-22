# Testing

Runs generated completions against test cases across all 12 languages using Docker. No language runtimes need to be installed locally — only Docker Desktop.

`runner.py` scans the results file to see which languages are needed, starts exactly one container per language, then runs all test cases for that language inside the same container — no restarts between problems. Code is copied in via the Docker API (`put_archive`). All containers are stopped when evaluation is done, or immediately if the script is interrupted.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Inference results file (JSONL) produced by `main.py`

## Usage

```bash
python testing/runner.py --results results.jsonl --output eval.jsonl
```

`results.jsonl` — output from inference, one record per line with `task_id`, `completion`, and `chosen_model`.

`eval.jsonl` — same records with a `passed` field added (`true`/`false`).

## How it works

1. All unique languages in the results file are identified
2. One Docker container per language is started (`sleep infinity`)
3. Any language-specific dependencies are installed once (e.g. `lodash` for JS, `Data::Compare` for Perl)
4. For each test case: source is copied into the container, executed with a 30s timeout, pass/fail recorded
5. All containers are stopped and removed on exit
