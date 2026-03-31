# Testing

Runs generated completions against test cases across all 12 languages using Docker. No language runtimes need to be installed locally — only Docker Desktop.

`runner.py` scans the results file to see which languages are needed, starts exactly one container per language, then runs all test cases for that language inside the same container — no restarts between problems. Code is copied in via the Docker API (`put_archive`). All containers are stopped when evaluation is done, or immediately if the script is interrupted.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed AND running
- Inference results file (JSONL) produced by `main.py`

## Usage

```bash
python3 testing/runner.py --results results_original_1.jsonl
```

or py or python... IDK I just do them all until one works

`results.jsonl` — output from inference, one record per line with `task_id`, `completion`, and `chosen_model`.

`eval.jsonl` — same records with `passed` and `extracted_code` fields added.

`testing_results.jsonl` — written alongside `eval.jsonl`; contains only `task_id` and `passed` for quick result inspection.

## How it works

1. All unique languages in the results file are identified
2. One Docker container per language is started (`sleep infinity`)
3. Any language-specific dependencies are installed once (e.g. `lodash` for JS, `Data::Compare` for Perl)
4. For each test case:
   - Markdown fences are stripped from the completion (`extract_code`)
   - The extracted code is sanitized to function-body-only (`sanitize`): for Python this uses the `ast` module to locate the target function and extract its body; for other languages a regex detects a duplicate signature and slices it off
   - The body is assembled with the prompt and test harness into a complete runnable file
   - The file is copied into the container, executed with a 30s timeout, pass/fail recorded
5. All containers are stopped and removed on exit
