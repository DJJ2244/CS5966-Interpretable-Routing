"""
split_data.py - Stratified 80/20 train/test split by programming_language.

Each language is split independently so both splits have ~equal language
proportions (~80% train, ~20% test per language).

Output:
  data/humaneval_xl_english_train.jsonl
  data/humaneval_xl_english_test.jsonl
"""

import json
import random
from collections import defaultdict
from pathlib import Path

DATA_PATH  = Path("data/humaneval_xl_english.jsonl")
TRAIN_PATH = Path("data/humaneval_xl_english_train.jsonl")
TEST_PATH  = Path("data/humaneval_xl_english_test.jsonl")

SEED      = 42
TEST_SIZE = 0.2


def split():
    # Group records by programming language
    buckets = defaultdict(list)
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                buckets[rec.get("programming_language", "unknown")].append(rec)

    rng = random.Random(SEED)

    train_records, test_records = [], []

    print(f"{'Language':<20} {'Total':>6} {'Train':>6} {'Test':>5} {'Test%':>6}")
    print("-" * 48)

    for lang in sorted(buckets):
        recs = buckets[lang][:]
        rng.shuffle(recs)
        split_idx = int(len(recs) * (1 - TEST_SIZE))
        train_records.extend(recs[:split_idx])
        test_records.extend(recs[split_idx:])
        pct = len(recs[split_idx:]) / len(recs) * 100
        print(f"{lang:<20} {len(recs):>6} {split_idx:>6} {len(recs) - split_idx:>5} {pct:>5.1f}%")

    # Shuffle the final splits so languages are interleaved
    rng.shuffle(train_records)
    rng.shuffle(test_records)

    def write_jsonl(path: Path, data: list):
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(TEST_PATH, test_records)

    print("-" * 48)
    print(f"{'TOTAL':<20} {len(train_records) + len(test_records):>6} {len(train_records):>6} {len(test_records):>5}")
    print(f"\nTrain -> {TRAIN_PATH}")
    print(f"Test  -> {TEST_PATH}")


if __name__ == "__main__":
    split()
