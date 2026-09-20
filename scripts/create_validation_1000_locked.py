from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "Nemotron-Personas-Korea"
SAMPLE_PATH = ROOT / "data" / "derived" / "validation_1000_locked_seed42.parquet"
OUT = ROOT / "outputs"
LOCKED = OUT / "validation_1000_locked"
SAMPLING = LOCKED / "sampling"
QUOTA_CSV = OUT / "validation_1000_quota_table.csv"
REPORT_MD = OUT / "validation_1000_sampling_report.md"
SEED = 42

SEXES = ["남성", "여성"]
AGES = ["19-29", "30-39", "40-49", "50-59", "60-69", "70+"]
REGIONS = ["서울", "인천·경기", "대전·세종·충청", "광주·전라", "대구·경북", "부산·울산·경남", "강원·제주"]
SEX_TARGET = {"남성": 495, "여성": 505}
SEX_SOURCE_TO_TARGET = {"남자": "남성", "여자": "여성"}
AGE_TARGET = {"19-29": 146, "30-39": 150, "40-49": 169, "50-59": 193, "60-69": 179, "70+": 163}
REGION_TARGET = {"서울": 186, "인천·경기": 325, "대전·세종·충청": 108, "광주·전라": 96,
                 "대구·경북": 95, "부산·울산·경남": 147, "강원·제주": 43}

# Exact source strings, verified against all one million source records before mapping.
PROVINCE_TO_REGION = {
    "서울": "서울",
    "인천": "인천·경기", "경기": "인천·경기",
    "대전": "대전·세종·충청", "세종": "대전·세종·충청", "충청남": "대전·세종·충청", "충청북": "대전·세종·충청",
    "광주": "광주·전라", "전라남": "광주·전라", "전북": "광주·전라",
    "대구": "대구·경북", "경상북": "대구·경북",
    "부산": "부산·울산·경남", "울산": "부산·울산·경남", "경상남": "부산·울산·경남",
    "강원": "강원·제주", "제주": "강원·제주",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest(files: list[Path]) -> dict:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}


def age_group(age: int) -> str:
    if 19 <= age <= 29: return "19-29"
    if 30 <= age <= 39: return "30-39"
    if 40 <= age <= 49: return "40-49"
    if 50 <= age <= 59: return "50-59"
    if 60 <= age <= 69: return "60-69"
    if age >= 70: return "70+"
    return "UNDER_19"


def ipf(observed: np.ndarray) -> np.ndarray:
    x = observed.astype(float) + 1e-12
    targets = [np.array([SEX_TARGET[s] for s in SEXES]), np.array([AGE_TARGET[a] for a in AGES]), np.array([REGION_TARGET[r] for r in REGIONS])]
    for _ in range(10000):
        old = x.copy()
        x *= (targets[0] / x.sum(axis=(1, 2)))[:, None, None]
        x *= (targets[1] / x.sum(axis=(0, 2)))[None, :, None]
        x *= (targets[2] / x.sum(axis=(0, 1)))[None, None, :]
        if np.max(np.abs(x - old)) < 1e-12: break
    return x


def integerize(expected: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    n = expected.size
    # Variables: integer cell quotas x, plus continuous positive/negative L1 deviations.
    c = np.r_[np.zeros(n), np.ones(2 * n)]
    integrality = np.r_[np.ones(n), np.zeros(2 * n)]
    lower = np.zeros(3 * n)
    upper = np.r_[capacities.ravel(), np.full(2 * n, np.inf)]
    rows = 2 + 6 + 7 + n
    A = lil_matrix((rows, 3 * n), dtype=float)
    b = []
    row = 0
    for si in range(2):
        for ai in range(6):
            for ri in range(7): A[row, np.ravel_multi_index((si, ai, ri), expected.shape)] = 1
        b.append(SEX_TARGET[SEXES[si]]); row += 1
    for ai in range(6):
        for si in range(2):
            for ri in range(7): A[row, np.ravel_multi_index((si, ai, ri), expected.shape)] = 1
        b.append(AGE_TARGET[AGES[ai]]); row += 1
    for ri in range(7):
        for si in range(2):
            for ai in range(6): A[row, np.ravel_multi_index((si, ai, ri), expected.shape)] = 1
        b.append(REGION_TARGET[REGIONS[ri]]); row += 1
    for j in range(n):
        A[row, j] = 1; A[row, n + j] = -1; A[row, 2 * n + j] = 1
        b.append(float(expected.ravel()[j])); row += 1
    b = np.array(b, dtype=float)
    # Margins are exact; deviation equations equal the continuous IPF targets.
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=LinearConstraint(A.tocsr(), b, b),
                  options={"time_limit": 120, "mip_rel_gap": 0.0})
    if not result.success:
        raise RuntimeError(f"quota MILP failed: {result.message}")
    quota = np.rint(result.x[:n]).astype(int).reshape(expected.shape)
    return quota


def compute_selection(dataset: ds.Dataset, quota: np.ndarray) -> list[int]:
    table = dataset.to_table(columns=["sex", "age", "province"])
    rows = table.to_pylist()
    cell_indices: dict[tuple[int, int, int], list[int]] = {(s, a, r): [] for s in range(2) for a in range(6) for r in range(7)}
    for idx, row in enumerate(rows):
        sex = row["sex"]; age = int(row["age"]); province = row["province"]
        ag = age_group(age)
        if ag == "UNDER_19": continue
        key = (SEXES.index(SEX_SOURCE_TO_TARGET[sex]), AGES.index(ag), REGIONS.index(PROVINCE_TO_REGION[province]))
        cell_indices[key].append(idx)
    rng = np.random.default_rng(SEED)
    selected = []
    for key in sorted(cell_indices):
        need = int(quota[key]); available = cell_indices[key]
        if need > len(available): raise RuntimeError(f"insufficient source cell {key}: need {need}, have {len(available)}")
        if need: selected.extend(int(x) for x in rng.choice(available, size=need, replace=False))
    return sorted(selected)


def counts_from_sample(rows: list[dict]) -> tuple[Counter, Counter, Counter]:
    return (Counter(SEX_SOURCE_TO_TARGET[r["sex"]] for r in rows), Counter(r["age_group"] for r in rows), Counter(r["region"] for r in rows))


def main() -> None:
    SAMPLING.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE_DIR.rglob("*.parquet"))
    if not files: raise FileNotFoundError(SOURCE_DIR)
    before = source_manifest(files)
    dataset = ds.dataset([str(p) for p in files], format="parquet")
    base = dataset.to_table(columns=["sex", "age", "province"])
    base_rows = base.to_pylist()
    actual_provinces = sorted({r["province"] for r in base_rows})
    if set(actual_provinces) != set(PROVINCE_TO_REGION):
        raise RuntimeError(f"province mapping mismatch: actual={actual_provinces}, mapped={sorted(PROVINCE_TO_REGION)}")
    actual_sexes = sorted({r["sex"] for r in base_rows})
    if set(actual_sexes) != set(SEX_SOURCE_TO_TARGET): raise RuntimeError(f"sex values changed: {actual_sexes}")
    under19 = sum(int(r["age"]) < 19 for r in base_rows)
    observed = np.zeros((2, 6, 7), dtype=int)
    for r in base_rows:
        ag = age_group(int(r["age"]))
        if ag == "UNDER_19": continue
        observed[SEXES.index(SEX_SOURCE_TO_TARGET[r["sex"]]), AGES.index(ag), REGIONS.index(PROVINCE_TO_REGION[r["province"]])] += 1
    expected = ipf(observed)
    quota = integerize(expected, observed)
    if not (quota.sum() == 1000 and np.array_equal(quota.sum(axis=(1,2)), [SEX_TARGET[s] for s in SEXES])
            and np.array_equal(quota.sum(axis=(0,2)), [AGE_TARGET[a] for a in AGES])
            and np.array_equal(quota.sum(axis=(0,1)), [REGION_TARGET[r] for r in REGIONS])):
        raise RuntimeError("integer quotas do not satisfy exact margins")
    selected = compute_selection(dataset, quota)
    if selected != compute_selection(dataset, quota): raise RuntimeError("seed=42 selection is not reproducible")
    sample = dataset.take(pa.array(selected, type=pa.int64()))
    rows = sample.to_pylist()
    for row in rows:
        row["age_group"] = age_group(int(row["age"]))
        row["region"] = PROVINCE_TO_REGION[row["province"]]
    schema = sample.schema.append(pa.field("age_group", pa.string())).append(pa.field("region", pa.string()))
    result = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(result, SAMPLE_PATH, compression="zstd")
    written = pq.read_table(SAMPLE_PATH).to_pylist()
    sex_c, age_c, region_c = counts_from_sample(written)
    uuids = [str(r["uuid"]) for r in written]
    source_uuid_set = set(str(x) for x in dataset.to_table(columns=["uuid"])["uuid"].combine_chunks().to_pylist())
    checks = {
        "record_count_1000": len(written) == 1000,
        "uuid_duplicates_zero": len(uuids) == len(set(uuids)),
        "all_uuids_in_source": set(uuids) <= source_uuid_set,
        "sex_exact": dict(sex_c) == SEX_TARGET,
        "age_exact": dict(age_c) == AGE_TARGET,
        "region_exact": dict(region_c) == REGION_TARGET,
        "seed42_reproducible": selected == compute_selection(dataset, quota),
    }
    after = source_manifest(files)
    checks["source_unchanged"] = before == after
    if not all(checks.values()): raise RuntimeError(f"sampling validation failed: {checks}")
    with QUOTA_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["sex", "age_group", "region", "source_count", "ipf_expected", "integer_quota", "quota_minus_ipf"])
        for si, sex in enumerate(SEXES):
            for ai, ag in enumerate(AGES):
                for ri, region in enumerate(REGIONS):
                    w.writerow([sex, ag, region, int(observed[si,ai,ri]), float(expected[si,ai,ri]), int(quota[si,ai,ri]), float(quota[si,ai,ri]-expected[si,ai,ri])])
    mapping_path = SAMPLING / "province_to_region_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["source_province_exact", "validation_region"]); w.writerows(PROVINCE_TO_REGION.items())
    manifest = {"source_files": before, "sample": {"path": str(SAMPLE_PATH.relative_to(ROOT)).replace("\\","/"), "bytes": SAMPLE_PATH.stat().st_size,
                                                          "sha256": sha256_file(SAMPLE_PATH)}, "checks": checks, "under_19_source_records": under19,
                "actual_province_values": actual_provinces, "selected_source_row_indices_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
    (SAMPLING / "sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Validation 1,000 locked sampling report", "", "## Outcome", "",
             "- New sample drawn from the 1,000,000-row Nemotron source; the earlier pilot was not used.",
             f"- Sample SHA-256: `{manifest['sample']['sha256']}`", "- IPF fitted the source sex × age_group × region joint distribution to the three target margins.",
             "- Mixed-integer L1 minimization produced the closest integer table subject to all exact margins and source-cell capacities.",
             "- Sampling was without replacement within each cell using NumPy `default_rng(42)`.", "",
             "## Exact target checks", "", "| Margin | Target and observed counts | Exact |", "| --- | --- | --- |",
             "| Sex | " + ", ".join(f"{k} {sex_c[k]}" for k in SEXES) + " | yes |",
             "| Age | " + ", ".join(f"{k} {age_c[k]}" for k in AGES) + " | yes |",
             "| Region | " + ", ".join(f"{k} {region_c[k]}" for k in REGIONS) + " | yes |", "",
             "## Integrity", ""] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in checks.items()] + ["", "## Limitation", "",
             "- The human survey's youngest group is 18–29, whereas Nemotron's minimum age is 19. The synthetic target is therefore 19–29; no 18-year-old synthetic respondent exists.",
             f"- Source records under 19: {under19}. No source row was modified or deleted.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"sample_sha256": manifest["sample"]["sha256"], "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
