import hashlib
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "derived" / "pilot_1000_seed42.parquet"
TARGET = ROOT / "data" / "derived" / "smoke_100_seed2026.parquet"
PROFILE = ROOT / "outputs" / "smoke_100_sample_profile.md"
SEED = 2026


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table_block(title, counter, total):
    lines = [f"## {title}", "", "| 값 | 빈도 | 비율 |", "| --- | ---: | ---: |"]
    for value, count in sorted(counter.items(), key=lambda x: str(x[0])):
        lines.append(f"| {value} | {count} | {count / total:.1%} |")
    return lines + [""]


before = digest(SOURCE)
pilot = pq.read_table(SOURCE)
if pilot.num_rows != 1000:
    raise RuntimeError(f"Expected 1,000 pilot rows, found {pilot.num_rows}")

indices = random.Random(SEED).sample(range(pilot.num_rows), 100)
sample = pilot.take(pa.array(indices, type=pa.int64()))
TARGET.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(sample, TARGET, compression="zstd")

after = digest(SOURCE)
if before != after:
    raise RuntimeError("pilot_1000_seed42.parquet changed during sampling")
if sample.num_rows != 100 or len(set(sample["uuid"].to_pylist())) != 100:
    raise RuntimeError("Smoke sample row or UUID validation failed")

lines = [
    "# Smoke test 100명 표본",
    "",
    "- Source: `data/derived/pilot_1000_seed42.parquet`",
    "- Method: Python `random.Random(2026).sample(range(1000), 100)`",
    "- 앞 100행이 아닌 비복원 단순무작위추출",
    "- 레코드: 100",
    "- UUID 중복: 0",
    "- 원본 pilot SHA-256 전후 동일: PASS",
    "",
]
for column in ["sex", "age_group", "province"]:
    lines.extend(table_block(column, Counter(sample[column].to_pylist()), sample.num_rows))
PROFILE.parent.mkdir(parents=True, exist_ok=True)
PROFILE.write_text("\n".join(lines), encoding="utf-8")

print(f"sample_rows={sample.num_rows}")
print(f"source_unchanged={before == after}")
print(f"target={TARGET}")
