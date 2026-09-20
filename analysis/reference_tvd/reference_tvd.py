from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "analysis" / "reference_tvd"
OUTPUT_CSV = OUTPUT_DIR / "reference_tvd.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "reference_tvd_summary.md"

PRIMARY_ANALYSIS = ROOT / "scripts" / "analyze_validation_1000_locked.py"
HUMAN_BENCHMARK = (
    ROOT
    / "outputs"
    / "validation_1000_locked"
    / "benchmark"
    / "human_benchmark_subgroups_20260818.json"
)
INTERFACE_COMPARISON = (
    ROOT
    / "outputs"
    / "consistency_audit_20260904"
    / "tables"
    / "HC8_interface_comparison_A_vs_B_v1.csv"
)
QUESTIONS = ("Q2", "Q3", "Q4", "Q5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_primary_distance_functions():
    """Import, without copying, the frozen Primary norm and TVD functions."""
    # The frozen analysis module imports pyarrow for its main analysis. This
    # derived script only needs norm/tvd, so provide an inert import placeholder
    # when pyarrow is absent rather than installing or changing dependencies.
    try:
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        pyarrow_module = types.ModuleType("pyarrow")
        parquet_module = types.ModuleType("pyarrow.parquet")
        pyarrow_module.parquet = parquet_module
        sys.modules.setdefault("pyarrow", pyarrow_module)
        sys.modules.setdefault("pyarrow.parquet", parquet_module)

    spec = importlib.util.spec_from_file_location(
        "locked_primary_analysis_for_reference_tvd", PRIMARY_ANALYSIS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Primary analysis module: {PRIMARY_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.norm, module.tvd


norm, tvd = load_primary_distance_functions()


def normalized_percent_vector(values: list[float]) -> np.ndarray:
    vector = norm(np.asarray(values, dtype=float))
    if not np.isclose(vector.sum(), 1.0, atol=1e-12):
        raise AssertionError("Normalized human vector does not sum to one")
    return vector


def append_value_row(
    rows: list[dict],
    *,
    family: str,
    metric: str,
    question: str,
    comparison: str,
    tvd_value: float,
    k: int | str,
    detail: str,
    scope: str,
    source: Path,
) -> None:
    rows.append(
        {
            "family": family,
            "metric": metric,
            "question": question,
            "comparison": comparison,
            "tvd": tvd_value,
            "tvd_min": "",
            "tvd_max": "",
            "k": k,
            "detail": detail,
            "aggregation_scope": scope,
            "source_file": source.relative_to(ROOT).as_posix(),
        }
    )


def append_range_row(
    rows: list[dict],
    *,
    family: str,
    metric: str,
    question: str,
    comparison: str,
    tvd_min: float,
    tvd_max: float,
    k: int | str,
    detail: str,
    scope: str,
    source: Path,
) -> None:
    rows.append(
        {
            "family": family,
            "metric": metric,
            "question": question,
            "comparison": comparison,
            "tvd": "",
            "tvd_min": tvd_min,
            "tvd_max": tvd_max,
            "k": k,
            "detail": detail,
            "aggregation_scope": scope,
            "source_file": source.relative_to(ROOT).as_posix(),
        }
    )


def calculate_reference_rows(benchmark: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    collected: dict[str, dict[str, float | tuple[float, float]]] = {
        "uniform": {},
        "single_choice": {},
        "sex": {},
        "age": {},
        "region_max": {},
    }

    # (1) Uniform and (2) one-option concentration reference points.
    for question in QUESTIONS:
        human = normalized_percent_vector(benchmark["overall"][question])
        k = len(human)
        uniform_value = tvd(np.full(k, 1.0 / k), human)
        append_value_row(
            rows,
            family="reference_baseline",
            metric="uniform_vs_human",
            question=question,
            comparison="uniform distribution vs overall human distribution",
            tvd_value=uniform_value,
            k=k,
            detail="Uniform probability 1/K for every response option",
            scope="one question",
            source=HUMAN_BENCHMARK,
        )
        collected["uniform"][question] = uniform_value

        concentrated = []
        for option_index in range(k):
            one_hot = np.zeros(k, dtype=float)
            one_hot[option_index] = 1.0
            concentrated.append(tvd(one_hot, human))
        min_index = int(np.argmin(concentrated))
        max_index = int(np.argmax(concentrated))
        min_value = float(concentrated[min_index])
        max_value = float(concentrated[max_index])
        labels = benchmark["choice_labels"][question]
        append_range_row(
            rows,
            family="reference_baseline",
            metric="single_choice_concentration_vs_human",
            question=question,
            comparison="100% on one option vs overall human distribution",
            tvd_min=min_value,
            tvd_max=max_value,
            k=k,
            detail=(
                f"minimum at option {min_index + 1} ({labels[min_index]}); "
                f"maximum at option {max_index + 1} ({labels[max_index]})"
            ),
            scope=f"minimum and maximum across {k} one-hot option distributions",
            source=HUMAN_BENCHMARK,
        )
        collected["single_choice"][question] = (min_value, max_value)

    uniform_mean = float(np.mean(list(collected["uniform"].values())))
    single_min_mean = float(
        np.mean([value[0] for value in collected["single_choice"].values()])
    )
    single_max_mean = float(
        np.mean([value[1] for value in collected["single_choice"].values()])
    )
    append_value_row(
        rows,
        family="reference_baseline",
        metric="uniform_vs_human",
        question="MEAN",
        comparison="uniform distribution vs overall human distribution",
        tvd_value=uniform_mean,
        k="4/5",
        detail="Equal-weight mean of Q2, Q3, Q4, and Q5",
        scope="four question-level TVDs, equal question weights",
        source=HUMAN_BENCHMARK,
    )
    append_range_row(
        rows,
        family="reference_baseline",
        metric="single_choice_concentration_vs_human",
        question="MEAN",
        comparison="100% on one option vs overall human distribution",
        tvd_min=single_min_mean,
        tvd_max=single_max_mean,
        k="4/5",
        detail="Mean of question-specific minima and mean of question-specific maxima",
        scope="four question-level minimum/maximum TVDs, equal question weights",
        source=HUMAN_BENCHMARK,
    )

    # (3) Published human-survey subgroup contrasts.
    subgroups = benchmark.get("subgroups", {})
    required = {"sex", "age", "region"}
    if not required.issubset(subgroups):
        missing = sorted(required.difference(subgroups))
        raise RuntimeError(f"Published subgroup cross-tabs unavailable: {missing}")

    sex_pairs = [("남성", "여성")]
    age_pairs = [("19-29", "70+")]
    for family, pairs in (("sex", sex_pairs), ("age", age_pairs)):
        for question in QUESTIONS:
            level_a, level_b = pairs[0]
            vector_a = normalized_percent_vector(subgroups[family][level_a][question])
            vector_b = normalized_percent_vector(subgroups[family][level_b][question])
            value = tvd(vector_a, vector_b)
            detail = f"{level_a} vs {level_b}"
            if family == "age":
                detail = "18–29 years (published; synthetic key 19-29) vs 70 years or older"
            append_value_row(
                rows,
                family="human_subgroup",
                metric=f"{family}_pair_tvd",
                question=question,
                comparison=detail,
                tvd_value=value,
                k=len(vector_a),
                detail="Published weighted row percentages, normalized to sum to one",
                scope="one prespecified human subgroup pair within one question",
                source=HUMAN_BENCHMARK,
            )
            collected[family][question] = value

        mean_value = float(np.mean(list(collected[family].values())))
        append_value_row(
            rows,
            family="human_subgroup",
            metric=f"{family}_pair_tvd",
            question="MEAN",
            comparison=("male vs female" if family == "sex" else "youngest vs oldest age group"),
            tvd_value=mean_value,
            k="4/5",
            detail="Equal-weight mean of Q2, Q3, Q4, and Q5",
            scope="four question-level human subgroup TVDs, equal question weights",
            source=HUMAN_BENCHMARK,
        )

    region_names = list(subgroups["region"])
    if len(region_names) < 2:
        raise RuntimeError("At least two published region rows are required")
    for question in QUESTIONS:
        pair_values = []
        for region_a, region_b in combinations(region_names, 2):
            vector_a = normalized_percent_vector(subgroups["region"][region_a][question])
            vector_b = normalized_percent_vector(subgroups["region"][region_b][question])
            pair_values.append((tvd(vector_a, vector_b), region_a, region_b))
        value, region_a, region_b = max(pair_values, key=lambda item: item[0])
        append_value_row(
            rows,
            family="human_subgroup",
            metric="region_max_pair_tvd",
            question=question,
            comparison=f"{region_a} vs {region_b}",
            tvd_value=value,
            k=len(benchmark["overall"][question]),
            detail=f"Maximum across {len(pair_values)} published region pairs",
            scope="maximum pairwise region TVD within one question",
            source=HUMAN_BENCHMARK,
        )
        collected["region_max"][question] = value

    region_mean = float(np.mean(list(collected["region_max"].values())))
    append_value_row(
        rows,
        family="human_subgroup",
        metric="region_max_pair_tvd",
        question="MEAN",
        comparison="mean of question-specific maximum region pairs",
        tvd_value=region_mean,
        k="4/5",
        detail="The maximizing region pair may differ by question",
        scope="equal-weight mean of four question-specific maximum pairwise region TVDs",
        source=HUMAN_BENCHMARK,
    )

    return rows, collected


def read_existing_result_ranges() -> dict[str, tuple[float, float]]:
    with INTERFACE_COMPARISON.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    primary = [
        float(row["primary_probability_mean_tvd"])
        for row in records
        if row["primary_probability_mean_tvd"].strip()
    ]
    hard_choice = [float(row["HC_B"]) for row in records if row["HC_B"].strip()]
    if len(primary) != 6 or len(hard_choice) != 8:
        raise AssertionError(
            f"Unexpected existing-result counts: Primary={len(primary)}, HC_B={len(hard_choice)}"
        )
    return {
        "primary": (min(primary), max(primary)),
        "hard_choice": (min(hard_choice), max(hard_choice)),
    }


def add_existing_ranges(rows: list[dict], ranges: dict[str, tuple[float, float]]) -> None:
    append_range_row(
        rows,
        family="existing_synthetic_human_result",
        metric="primary_probability_mean_tvd_range",
        question="MEAN",
        comparison="six model × persona-condition means",
        tvd_min=ranges["primary"][0],
        tvd_max=ranges["primary"][1],
        k="4/5",
        detail="Qwen, EXAONE, Gemma × two persona conditions",
        scope="original-order Q2–Q5 equal-weight mean TVD; range across six conditions",
        source=INTERFACE_COMPARISON,
    )
    append_range_row(
        rows,
        family="existing_synthetic_human_result",
        metric="hard_choice_mean_tvd_range",
        question="MEAN",
        comparison="eight model × persona-condition means",
        tvd_min=ranges["hard_choice"][0],
        tvd_max=ranges["hard_choice"][1],
        k="4/5",
        detail="Qwen, EXAONE, Gemma, Motif run1 × two persona conditions",
        scope="original-order Q2–Q5 equal-weight mean TVD; range across eight conditions",
        source=INTERFACE_COMPARISON,
    )


def write_csv(rows: list[dict]) -> None:
    fields = [
        "family",
        "metric",
        "question",
        "comparison",
        "tvd",
        "tvd_min",
        "tvd_max",
        "k",
        "detail",
        "aggregation_scope",
        "source_file",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def comparison_table(
    collected: dict[str, dict[str, float | tuple[float, float]]],
    ranges: dict[str, tuple[float, float]],
) -> list[tuple[str, str, str]]:
    uniform = float(np.mean(list(collected["uniform"].values())))
    single_min = float(np.mean([v[0] for v in collected["single_choice"].values()]))
    single_max = float(np.mean([v[1] for v in collected["single_choice"].values()]))
    sex = float(np.mean(list(collected["sex"].values())))
    age = float(np.mean(list(collected["age"].values())))
    region = float(np.mean(list(collected["region_max"].values())))
    return [
        ("Uniform distribution vs human", fmt(uniform), "equal-weight mean of Q2–Q5"),
        (
            "100% single-option concentration vs human",
            f"{fmt(single_min)}–{fmt(single_max)}",
            "mean of question minima – mean of question maxima",
        ),
        ("Human male vs female", fmt(sex), "equal-weight mean of Q2–Q5"),
        ("Human youngest vs oldest age group", fmt(age), "equal-weight mean of Q2–Q5"),
        (
            "Human maximum region pair",
            fmt(region),
            "mean of four question-specific maxima",
        ),
        (
            "Existing Primary synthetic–human",
            f"{fmt(ranges['primary'][0])}–{fmt(ranges['primary'][1])}",
            "range across 3 models × 2 persona conditions",
        ),
        (
            "Existing hard-choice synthetic–human",
            f"{fmt(ranges['hard_choice'][0])}–{fmt(ranges['hard_choice'][1])}",
            "range across 4 models × 2 persona conditions; Motif run1",
        ),
    ]


def markdown_question_table(
    collected: dict[str, dict[str, float | tuple[float, float]]]
) -> list[str]:
    lines = [
        "| Reference | Q2 | Q3 | Q4 | Q5 | Q2–Q5 mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    uniform = collected["uniform"]
    lines.append(
        "| Uniform vs human | "
        + " | ".join(fmt(float(uniform[q])) for q in QUESTIONS)
        + f" | {fmt(float(np.mean(list(uniform.values()))))} |"
    )
    single = collected["single_choice"]
    single_cells = [f"{fmt(single[q][0])}–{fmt(single[q][1])}" for q in QUESTIONS]
    single_mean = (
        float(np.mean([v[0] for v in single.values()])),
        float(np.mean([v[1] for v in single.values()])),
    )
    lines.append(
        "| 100% single-option concentration vs human (min–max) | "
        + " | ".join(single_cells)
        + f" | {fmt(single_mean[0])}–{fmt(single_mean[1])} |"
    )
    for key, label in (
        ("sex", "Human male vs female"),
        ("age", "Human youngest vs oldest age group"),
        ("region_max", "Human maximum region pair"),
    ):
        values = collected[key]
        lines.append(
            f"| {label} | "
            + " | ".join(fmt(float(values[q])) for q in QUESTIONS)
            + f" | {fmt(float(np.mean(list(values.values()))))} |"
        )
    return lines


def write_summary(
    benchmark: dict,
    rows: list[dict],
    collected: dict[str, dict[str, float | tuple[float, float]]],
    ranges: dict[str, tuple[float, float]],
) -> None:
    table = comparison_table(collected, ranges)
    region_rows = [
        row
        for row in rows
        if row["metric"] == "region_max_pair_tvd" and row["question"] in QUESTIONS
    ]
    source_pdf = ROOT / benchmark["source"]["pdf_local"]
    lines = [
        "# Reference TVD summary",
        "",
        "**Verification status:** ANALYZED — deterministic derived analysis of preserved results; no LLM inference was run.",
        "",
        "## Inputs and method",
        "",
        f"- Human benchmark and subgroup transcription: `{HUMAN_BENCHMARK.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256(HUMAN_BENCHMARK)}`)",
        f"- Official survey report: `{source_pdf.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256(source_pdf)}`; subgroup cross-tabs on PDF pages 10–13 were visually checked during this audit)",
        f"- Existing Primary and corrected hard-choice means: `{INTERFACE_COMPARISON.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256(INTERFACE_COMPARISON)}`)",
        f"- Reused functions: `norm` and `tvd` imported from `{PRIMARY_ANALYSIS.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256(PRIMARY_ANALYSIS)}`)",
        "- Normalization: every published percentage vector was normalized to sum to one with the imported Primary `norm` function. All distances were evaluated with the imported Primary `tvd` function.",
        "- Question summary: Q2–Q5 were weighted equally. Q2, Q3, and Q5 have five options; Q4 has four.",
        "",
        "## Question-level reference points",
        "",
        *markdown_question_table(collected),
        "",
        "For the single-option reference, the mean interval is the mean of the four question-specific minima through the mean of the four question-specific maxima; it is not a confidence interval.",
        "",
        "## Region pairs producing the question-specific maxima",
        "",
        "| Question | Region pair | TVD |",
        "| --- | --- | ---: |",
    ]
    for row in region_rows:
        lines.append(f"| {row['question']} | {row['comparison']} | {fmt(float(row['tvd']))} |")

    lines += [
        "",
        "## Final comparison table",
        "",
        "| Reference or preserved result | Q2–Q5 mean TVD or range | Scope |",
        "| --- | ---: | --- |",
    ]
    for label, value, scope in table:
        lines.append(f"| {label} | {value} | {scope} |")

    lines += [
        "",
        "## Human subgroup availability",
        "",
        "The official report publishes weighted row percentages for sex, age group, and region for every item Q2–Q5. Accordingly, male–female, youngest–oldest, and maximum pairwise region TVDs were calculable without imputation. Human microdata and joint subgroup distributions were not used or reconstructed.",
        "",
        "## Interpretation boundaries",
        "",
        "- These values are descriptive scale references, not prespecified thresholds of acceptable validity and not tests of statistical equivalence.",
        "- The human subgroup contrasts are the most substantively comparable within-survey reference available here, but they use rounded published percentages and do not include sampling, nonresponse, or weighting uncertainty.",
        "- The maximum-region value is selected from all published region pairs separately for each question, so it is an upper descriptive reference by construction; the maximizing pair may change across questions.",
        "- The synthetic–human ranges are preserved results, not recalculated model estimates. Primary and hard-choice are different measurement configurations and should not be treated as the same estimand.",
        "- No causal claim, subgroup-to-individual inference, population-truth claim, or generalization beyond these four items and this survey is licensed by this calculation.",
        "- All Q2–Q5 items and the prespecified sex, age-extreme, and region-maximum comparisons were retained; no favorable subset was selected.",
        "",
        "## Reproducibility",
        "",
        f"Run `{Path(__file__).name}` from any working directory. It writes only `{OUTPUT_DIR.relative_to(ROOT).as_posix()}/` and does not modify frozen Primary or hard-choice artifacts.",
        "",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def print_comparison_table(table: list[tuple[str, str, str]]) -> None:
    print("\nFinal reference TVD comparison")
    print("=" * 106)
    print(f"{'Reference/result':<48} {'Mean TVD or range':>20}  Scope")
    print("-" * 106)
    for label, value, scope in table:
        print(f"{label:<48} {value:>20}  {scope}")
    print("=" * 106)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(HUMAN_BENCHMARK.read_text(encoding="utf-8"))
    if tuple(benchmark["overall"].keys()) != QUESTIONS:
        raise AssertionError("Human benchmark question order/content is not Q2–Q5")
    rows, collected = calculate_reference_rows(benchmark)
    ranges = read_existing_result_ranges()
    add_existing_ranges(rows, ranges)
    write_csv(rows)
    write_summary(benchmark, rows, collected, ranges)
    print_comparison_table(comparison_table(collected, ranges))
    print(f"\nWrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
