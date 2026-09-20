from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_choice_common import *
from model_adapters import LocalAdapter, post_json, sanitize

RUN_ID = "pilot10_v2_completion_allowance_1024"
MAX_COMPLETION = 1024
CHECKPOINT = PILOT_OUT / "pilot10_observations_v2.jsonl"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def primary_files() -> list[Path]:
    candidates = [
        SAMPLE_PATH,
        ROOT / "outputs/validation_1000_locked/analysis_plan_locked.json",
        ROOT / "outputs/validation_1000_locked/validation_1000_report.md",
        ROOT / "outputs/validation_1000_locked/validation_1000_primary_distribution.csv",
        ROOT / "outputs/validation_1000_locked/probabilities/qwen_validation_1000_choice_probabilities.csv",
        ROOT / "outputs/validation_1000_locked/probabilities/exaone_validation_1000_choice_probabilities.csv",
        ROOT / "outputs/validation_1000_locked/probabilities/gemma_validation_1000_choice_probabilities.csv",
        ROOT / "outputs/manuscript_draft/manuscript_with_abstract.md",
    ]
    return [p for p in candidates if p.exists()]


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in paths}


def parse_response(model_key: str, response, latency, error, status, task):
    row, condition, order_condition, qid, rotation = task
    data = response or {}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw = message.get("content")
    reasoning = message.get("reasoning")
    parsed, valid = parse_hard_choice(raw, qid)
    _, order, mapping = build_messages(row, condition, qid, rotation)
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    finish_reason = choice.get("finish_reason")
    api_failure = status != 200 or error is not None
    truncated = finish_reason == "length" or (raw is None and completion_tokens is not None and int(completion_tokens) >= MAX_COMPLETION)
    semantic = mapping.get(parsed) if parsed else None
    visible = LABELS[:len(order)]
    return {
        "run_id": RUN_ID, "model": MODELS[model_key]["display"], "model_key": model_key,
        "uuid": str(row["uuid"]), "question_id": qid, "persona_condition": condition,
        "order_condition": order_condition, "rotation_id": rotation,
        "displayed_semantic_order": ",".join(map(str, order)), "label_to_semantic_mapping": mapping_json(mapping),
        "displayed_options": " | ".join(f"{label}={QUESTIONS[qid]['choices'][mapping[label]]}" for label in visible),
        "max_completion_allowance": MAX_COMPLETION, "actual_completion_tokens": completion_tokens,
        "finish_reason": finish_reason, "raw_model_response": raw, "parsed_label": parsed,
        "semantic_choice": semantic, "valid_response": valid, "truncated": truncated,
        "api_failure": api_failure, "http_status": status, "error_sanitized": sanitize(error, os.environ.get("MORPH_API_KEY", "")) if error else None,
        "latency_seconds": round(latency, 6), "retry_count": 0,
        "returned_model_id": data.get("model"), "reasoning_present": bool(reasoning),
        "reasoning_character_count": len(reasoning or ""),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion_tokens, "total_tokens": usage.get("total_tokens"),
        "temperature": 0, "top_p": 1.0, "top_k": 1 if model_key != "motif" else None,
        "seed": 2026, "seed_support": "local llama.cpp" if model_key != "motif" else "accepted by API; application not echoed",
        "chat_template": "embedded GGUF template" if model_key != "motif" else "provider-managed OpenAI-compatible chat template",
        "timestamp": now(), "api_key_recorded": False,
    }


def local_call(model_key: str, task):
    row, condition, _, qid, rotation = task
    messages, _, _ = build_messages(row, condition, qid, rotation)
    cfg = MODELS[model_key]
    payload = {"model": cfg["display"], "messages": messages, "temperature": 0, "top_p": 1.0, "top_k": 1, "seed": 2026, "max_tokens": MAX_COMPLETION, "stream": False, "cache_prompt": False}
    response, latency, error, status = post_json(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions", payload)
    return parse_response(model_key, response, latency, error, status, task)


def motif_call(task):
    key = os.environ.get("MORPH_API_KEY")
    if not key:
        raise RuntimeError("MORPH_API_KEY missing")
    row, condition, _, qid, rotation = task
    messages, _, _ = build_messages(row, condition, qid, rotation)
    cfg = MODELS["motif"]
    payload = {"model": cfg["model_id"], "messages": messages, "temperature": 0, "top_p": 1.0, "seed": 2026, "max_completion_tokens": MAX_COMPLETION, "stream": False}
    response, latency, error, status = post_json(cfg["base_url"] + "/chat/completions", payload, {"Authorization": "Bearer " + key}, timeout=240)
    return parse_response("motif", response, latency, error, status, task)


def append(row):
    with CHECKPOINT.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_checkpoint():
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()] if CHECKPOINT.exists() else []


def tasks_for(model_key, pilot, assignments, existing_keys):
    tasks = []
    for row in pilot:
        uuid = str(row["uuid"])
        for condition in CONDITIONS:
            for order_condition in ORDER_CONDITIONS:
                for qid in QUESTION_IDS:
                    key = (model_key, uuid, condition, order_condition, qid)
                    if key in existing_keys: continue
                    rotation = 0 if order_condition == "original_order" else assignments[(uuid, qid)]
                    tasks.append((row, condition, order_condition, qid, rotation))
    return tasks


def export(rows):
    ordered = sorted(rows, key=lambda r: (list(MODELS).index(r["model_key"]), r["uuid"], CONDITIONS.index(r["persona_condition"]), ORDER_CONDITIONS.index(r["order_condition"]), QUESTION_IDS.index(r["question_id"])))
    with (PILOT_OUT / "pilot10_observations_v2.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (PILOT_OUT / "pilot10_observations_v2.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0])); writer.writeheader(); writer.writerows(ordered)
    pq.write_table(pa.Table.from_pylist(ordered), PILOT_OUT / "pilot10_observations.parquet", compression="zstd")


def run_repro(pilot):
    records = []
    for model_key in MODELS:
        log = PILOT_OUT / f"{model_key}_repro_v2_server.log"
        context = LocalAdapter(model_key, log) if model_key != "motif" else None
        if context: context.__enter__()
        try:
            for row in pilot[:2]:
                runs = []
                task = (row, "demographics_only", "original_order", "Q2", 0)
                for repeat in (1, 2):
                    result = local_call(model_key, task) if model_key != "motif" else motif_call(task)
                    runs.append({"repeat": repeat, "raw_content": result["raw_model_response"], "parsed_label": result["parsed_label"], "semantic_choice": result["semantic_choice"], "valid": result["valid_response"], "finish_reason": result["finish_reason"], "completion_tokens": result["completion_tokens"], "reasoning_present": result["reasoning_present"], "reasoning_character_count": result["reasoning_character_count"], "latency_seconds": result["latency_seconds"]})
                records.append({"model": MODELS[model_key]["display"], "model_key": model_key, "uuid": str(row["uuid"]), "question_id": "Q2", "persona_condition": "demographics_only", "order_condition": "original_order", "runs": runs, "exact_label_identical": runs[0]["parsed_label"] == runs[1]["parsed_label"], "semantic_choice_identical": runs[0]["semantic_choice"] == runs[1]["semantic_choice"], "raw_content_identical": runs[0]["raw_content"] == runs[1]["raw_content"], "finish_reason_identical": runs[0]["finish_reason"] == runs[1]["finish_reason"], "completion_tokens_identical": runs[0]["completion_tokens"] == runs[1]["completion_tokens"]})
        finally:
            if context: context.__exit__(None, None, None)
    write_json(PILOT_OUT / "reproducibility_check_v2.json", {"run_id": RUN_ID, "timestamp": now(), "request_count": 16, "records": records, "human_benchmark_accessed": False, "api_key_recorded": False})


def main():
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    all_rows = load_locked_sample(); pilot = select_pilot10(all_rows); assignments = rotation_assignments(all_rows)
    if sha256_file(SAMPLE_PATH) != "6650945363b9cd2d7ca702ec98e900aa16cfada714642af3521a2c0e895a27f7": raise RuntimeError("sample hash mismatch")
    for model_key, cfg in MODELS.items():
        if cfg["kind"] == "local" and sha256_file(cfg["path"]) != cfg["sha256"]: raise RuntimeError(f"{model_key} model hash mismatch")
    preserved = primary_files(); before = snapshot(preserved)
    existing = read_checkpoint()
    keys = {(r["model_key"], r["uuid"], r["persona_condition"], r["order_condition"], r["question_id"]) for r in existing}
    if len(keys) != len(existing): raise RuntimeError("duplicate checkpoint rows")
    for model_key in MODELS:
        pending = tasks_for(model_key, pilot, assignments, keys)
        if not pending: continue
        log = PILOT_OUT / f"{model_key}_pilot_v2_server.log"
        context = LocalAdapter(model_key, log) if model_key != "motif" else None
        if context: context.__enter__()
        try:
            worker = (lambda task: local_call(model_key, task)) if model_key != "motif" else motif_call
            workers = MODELS[model_key].get("parallel", 4) if model_key != "motif" else 4
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(worker, pending):
                    append(result); existing.append(result)
                    if model_key == "motif" and result["returned_model_id"] != MODELS["motif"]["model_id"]:
                        raise RuntimeError("unexpected Motif returned model ID")
        finally:
            if context: context.__exit__(None, None, None)
        print(json.dumps({"model": model_key, "completed": sum(r["model_key"] == model_key for r in existing), "expected": 160}), flush=True)
    if len(existing) != EXPECTED_PILOT_OBSERVATIONS: raise RuntimeError(f"expected 640 rows, found {len(existing)}")
    export(existing)
    run_repro(pilot)
    after = snapshot(preserved)
    modified = sorted(k for k in before if before[k] != after[k])
    write_json(PILOT_OUT / "primary_preservation_v2.json", {"before": before, "after": after, "modified": modified, "pass": not modified})
    print(json.dumps({"rows": len(existing), "primary_modified": len(modified), "repro_requests": 16, "api_key_exposure": 0}))


if __name__ == "__main__": main()
