from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "data" / "derived" / "validation_1000_locked_seed42.parquet"
OUT = ROOT / "outputs" / "hard_choice_extension"
PILOT_OUT = OUT / "technical_pilot_10"
AUDIT_OUT = OUT / "audits"
RUNTIME = ROOT / "tools" / "llama.cpp" / "b10514"
SERVER = RUNTIME / "llama-server.exe"
EXTENSION_NAME = "Supplementary hard-choice model comparison"
CONDITIONS = ("demographics_only", "demographics_plus_persona")
ORDER_CONDITIONS = ("original_order", "counterbalanced_order")
QUESTION_IDS = ("Q2", "Q3", "Q4", "Q5")
LABELS = tuple("ABCDE")
EXPECTED_PILOT_OBSERVATIONS = 4 * 10 * 2 * 2 * 4
EXPECTED_FULL_PER_MODEL = 1000 * 2 * 2 * 4
EXPECTED_FULL_TOTAL = 4 * EXPECTED_FULL_PER_MODEL

SYSTEM_PROMPT = """제공된 합성 인물의 관점에서 설문에 응답한다.
제공되지 않은 새로운 생애사, 사건, 경험 등의 사실을 만들어내지 않는다.
그러나 프로필에 해당 정책에 대한 의견이 명시되어 있지 않다는 이유만으로 자동으로 '잘 모르겠다'를 선택하지 않는다.
실제 사람이 설문에 응답하듯 제공된 특성을 바탕으로 가장 자연스러운 정책 선호를 판단한다.
'잘 모르겠다'는 정상적인 선택지이며 실제로 판단이 불확실한 경우 선택할 수 있다.
특정 응답으로 유도하지 않는다.
이유나 chain-of-thought를 생성하지 않는다.
현재 문항의 선택지 코드 하나만 출력한다."""

QUESTIONS = {
    "Q2": {"text": "선택적 모병제는 현행 징병제를 유지하면서, 인공지능과 드론 등 전문 기술 분야에 한해 보수와 혜택을 강화한 기술집약형 부사관을 도입하는 징병제와 모병제의 혼합 방식입니다. 선생님께서는 선택적 모병제 도입에 대해 어떻게 생각하십니까?", "choices": {1: "매우 찬성한다", 2: "찬성하는 편이다", 3: "반대하는 편이다", 4: "매우 반대한다", 5: "잘 모르겠다"}},
    "Q3": {"text": "장교와 부사관 지원이 정원에 미치지 못하는 상황이 이어지고 있습니다. 선생님께서는 선택적 모병제가 병력 충원 문제 해결에 도움이 될 것으로 보십니까?", "choices": {1: "크게 도움될 것이다", 2: "어느 정도 도움될 것이다", 3: "별로 도움되지 않을 것이다", 4: "전혀 도움되지 않을 것이다", 5: "잘 모르겠다"}},
    "Q4": {"text": "앞으로 우리나라 병역제도가 어떤 방향으로 가야 한다고 생각하십니까?", "choices": {1: "현행 징병제 유지", 2: "징병제와 모병제 병행", 3: "전면 모병제 전환", 4: "잘 모르겠다"}},
    "Q5": {"text": "병역 자원이 줄어드는 상황에서 여성도 징병 대상에 포함하자는 주장이 있습니다. 선생님께서는 여성 징병제에 대해 어떻게 생각하십니까?", "choices": {1: "매우 찬성한다", 2: "찬성하는 편이다", 3: "반대하는 편이다", 4: "매우 반대한다", 5: "잘 모르겠다"}},
}

MODELS = {
    "qwen": {"display": "Qwen3-8B", "path": ROOT / "models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", "port": 8201, "parallel": 4, "kind": "local"},
    "exaone": {"display": "EXAONE-3.5-7.8B-Instruct", "path": ROOT / "models/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf", "sha256": "2f4aff8d555b82500d4de2723810586e83e358003388fdd08bd52e6091e6ddd1", "port": 8202, "parallel": 4, "kind": "local"},
    "gemma": {"display": "Gemma-3-12B", "path": ROOT / "models/google/gemma-3-12b-it-qat-q4_0-gguf/gemma-3-12b-it-q4_0.gguf", "sha256": "dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd", "port": 8203, "parallel": 2, "kind": "local"},
    "motif": {"display": "Motif 3", "model_id": "motif/motif-3", "base_url": "https://api-cbt.morphfactory.io/v1", "kind": "api"},
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_sample() -> list[dict[str, Any]]:
    rows = pq.read_table(SAMPLE_PATH).to_pylist()
    if len(rows) != 1000 or len({str(r["uuid"]) for r in rows}) != 1000:
        raise RuntimeError("locked sample integrity check failed")
    return rows


def select_pilot10(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Parquet row order is frozen by the locked sample hash and is the predeclared selection rule.
    return rows[:10]


def rotation_assignments(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    assignments: dict[tuple[str, str], int] = {}
    for qid in QUESTION_IDS:
        k = len(QUESTIONS[qid]["choices"])
        uuids = sorted((str(r["uuid"]) for r in rows), key=lambda u: hashlib.sha256(f"2026|{qid}|{u}".encode()).hexdigest())
        for rank, uuid in enumerate(uuids):
            assignments[(uuid, qid)] = rank % k
    return assignments


def display_mapping(qid: str, rotation: int) -> tuple[list[int], dict[str, int]]:
    semantic = list(QUESTIONS[qid]["choices"])
    order = semantic[rotation:] + semantic[:rotation]
    mapping = {LABELS[position]: code for position, code in enumerate(order)}
    return order, mapping


def profile_system(row: dict[str, Any], condition: str) -> str:
    fields = [("sex", row["sex"]), ("age", row["age"]), ("age_group", row["age_group"]), ("region/province", f"{row['region']} ({row['province']})"), ("marital_status", row["marital_status"]), ("education_level", row["education_level"]), ("occupation", row["occupation"])]
    if condition == "demographics_plus_persona":
        fields.append(("persona", row["persona"]))
    return SYSTEM_PROMPT + "\n\n합성 응답자 정보:\n" + "\n".join(f"- {name}: {value}" for name, value in fields)


def build_messages(row: dict[str, Any], condition: str, qid: str, rotation: int) -> tuple[list[dict[str, str]], list[int], dict[str, int]]:
    order, mapping = display_mapping(qid, rotation)
    visible = LABELS[:len(order)]
    option_text = "\n".join(f"{label} = {QUESTIONS[qid]['choices'][mapping[label]]}" for label in visible)
    instruction = f"위 선택지 중 가장 적절한 하나를 선택하세요. 답변은 {', '.join(visible)} 중 하나만 출력하세요. 설명이나 다른 문장은 출력하지 마세요."
    user = f"{QUESTIONS[qid]['text']}\n\n{option_text}\n\n{instruction}"
    return [{"role": "system", "content": profile_system(row, condition)}, {"role": "user", "content": user}], order, mapping


def parse_hard_choice(raw: str | None, qid: str) -> tuple[str | None, bool]:
    parsed = (raw or "").strip()
    valid = set(LABELS[:len(QUESTIONS[qid]["choices"])])
    return (parsed if parsed in valid else None), parsed in valid


def mapping_json(mapping: dict[str, int]) -> str:
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
