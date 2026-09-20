from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "validation_1000_locked" / "validation_1000_human_comparison.csv"
HARD_CHOICE = (
    ROOT
    / "outputs"
    / "hard_choice_extension"
    / "full_run_v1"
    / "analysis_final"
    / "tables"
    / "HC1_four_model_human_original_order.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def primary_means() -> dict[tuple[str, str], float]:
    # TVD is repeated once per response category in the released long table.
    unique: dict[tuple[str, str, str], float] = {}
    names: dict[str, str] = {}
    for row in read_csv(PRIMARY):
        if row["primary"].lower() != "true":
            continue
        model_key = row["model_key"]
        names[model_key] = row["model"]
        key = (model_key, row["persona_condition"], row["question"])
        unique[key] = float(row["tvd"])

    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for (model_key, condition, _question), value in unique.items():
        grouped[(names[model_key], condition)].append(value)
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def hard_choice_means() -> dict[tuple[str, str], float]:
    return {
        (row["model"], row["persona_condition"]): float(row["mean_q2_q5_tvd"])
        for row in read_csv(HARD_CHOICE)
    }


def main() -> None:
    primary = primary_means()
    hard_choice = hard_choice_means()

    primary_values = list(primary.values())
    hard_choice_values = list(hard_choice.values())
    assert 0.526 <= min(primary_values) <= 0.527
    assert 0.576 <= max(primary_values) <= 0.578
    assert 0.464 <= min(hard_choice_values) <= 0.465
    assert 0.623 <= max(hard_choice_values) <= 0.624

    print("Primary mean Q2-Q5 TVD")
    for (model, condition), value in sorted(primary.items()):
        print(f"  {model} | {condition}: {value:.6f}")

    print("\nHard-choice mean Q2-Q5 TVD")
    for (model, condition), value in sorted(hard_choice.items()):
        print(f"  {model} | {condition}: {value:.6f}")

    print("\nReleased headline ranges verified.")


if __name__ == "__main__":
    main()
