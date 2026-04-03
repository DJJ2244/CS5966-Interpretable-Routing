"""
split_data.py - Splits humaneval_xl_english.jsonl into train/test JSONL files.

Output:
  data/humaneval_xl_english_train.jsonl
  data/humaneval_xl_english_test.jsonl
"""

import json
import random
from pathlib import Path

DATA_PATH = Path("data/humaneval_xl_english.jsonl")
TRAIN_PATH = Path("data/humaneval_xl_english_train.jsonl")
TEST_PATH = Path("data/humaneval_xl_english_test.jsonl")

SEED = 42
TEST_SIZE = 0.2


def split():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - TEST_SIZE))
    train_records = shuffled[:split_idx]
    test_records = shuffled[split_idx:]

    def write_jsonl(path: Path, data: list):
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(TEST_PATH, test_records)

    print(f"Total : {len(records)}")
    print(f"Train : {len(train_records)} -> {TRAIN_PATH}")
    print(f"Test  : {len(test_records)} -> {TEST_PATH}")


if __name__ == "__main__":
    split()
