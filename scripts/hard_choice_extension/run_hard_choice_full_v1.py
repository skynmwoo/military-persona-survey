from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_choice_common import *
from model_adapters import LocalAdapter, post_json, sanitize
from run_final_stability_audit import semantic_parse
import run_hard_choice_pilot10_v2 as prior

FULL = OUT / "full_run_v1"
RAW = FULL / "raw"
AUDITS = FULL / "audits"
PROTOCOL = OUT / "hard_choice_protocol_frozen_v1.md"
PROTOCOL_SHA256 = "5b0f6ddefc3e637d7eb74217aa8fee18a7d438c5bae00edc479bde90b3e2348d"
SAMPLE_SHA256 = "6650945363b9cd2d7ca702ec98e900aa16cfada714642af3521a2c0e895a27f7"
MAX_COMPLETION = 32
TARGETS = ("qwen", "exaone", "gemma", "motif_run1", "motif_run2")
RESUME_LOG = AUDITS / "resume_log.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def target_paths(target: str) -> tuple[Path, Path]:
    directory = RAW / ("motif_replicate2" if target == "motif_run2" else target)
    return directory / "checkpoint.jsonl", directory / "observations.parquet"


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_rate_limit(row: dict[str, Any]) -> bool:
    text = str(row.get("error_sanitized") or "").lower()
    return row.get("http_status") == 429 or "rate limit" in text or "rate_limit" in text or "too many requests" in text


def update_resume_event(target: str, **values) -> None:
    data = json.loads(RESUME_LOG.read_text(encoding="utf-8")) if RESUME_LOG.exists() else {"events": []}
    active = next((e for e in reversed(data["events"]) if e.get("target") == target and e.get("status") == "running"), None)
    if active and values.get("status") == "running":
        active["interruption_detected_at"] = now()
        active["status"] = "interrupted"
        active = None
    if values.get("status") == "running":
        data["events"].append({"target": target, **values})
    else:
        active = next((e for e in reversed(data["events"]) if e.get("target") == target and e.get("status") == "running"), None)
        if active: active.update(values)
        else: data["events"].append({"target": target, **values})
    write_json(RESUME_LOG, data)


def all_tasks(sample, target: str):
    assignments = rotation_assignments(sample)
    tasks = []
    for row in sample:
        uuid = str(row["uuid"])
        for condition in CONDITIONS:
            for order_condition in ORDER_CONDITIONS:
                for qid in QUESTION_IDS:
                    rotation = 0 if order_condition == "original_order" else assignments[(uuid, qid)]
                    tasks.append((row, condition, order_condition, qid, rotation))
    return tasks


def key_for(task) -> tuple[str, str, str, str]:
    row, condition, order_condition, qid, _ = task
    return str(row["uuid"]), condition, order_condition, qid


def parse_result(model_key: str, run_label: str, response, latency: float, error, status, task):
    row, condition, order_condition, qid, rotation = task
    data = response or {}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    parsed = semantic_parse(raw, qid)
    _, order, mapping = build_messages(row, condition, qid, rotation)
    usage = data.get("usage") or {}
    completion = usage.get("completion_tokens")
    finish = choice.get("finish_reason")
    semantic = mapping.get(parsed["semantic_label"]) if parsed["semantic_label"] else None
    return {
        "experiment": EXTENSION_NAME, "run_label": run_label, "model": MODELS[model_key]["display"], "model_key": model_key,
        "uuid": str(row["uuid"]), "sex": row["sex"], "age_group": row["age_group"], "region": row["region"],
        "question_id": qid, "persona_condition": condition, "order_condition": order_condition, "rotation_id": rotation,
        "displayed_semantic_order": ",".join(map(str, order)), "label_to_semantic_mapping": mapping_json(mapping),
        "mapping_bijection": len(mapping) == len(set(mapping.values())) == len(QUESTIONS[qid]["choices"]),
        "raw_response": raw, "parsed_label": parsed["semantic_label"], "semantic_choice": semantic,
        "semantic_valid": parsed["semantic_valid"], "format_compliant": parsed["format_compliant"], "parser_reason": parsed["parser_reason"],
        "latency_seconds": round(latency, 6), "finish_reason": finish, "completion_tokens": completion,
        "temperature": 0, "top_p": 1.0, "top_k": 1 if model_key != "motif" else None, "seed": 2026,
        "max_completion_allowance": MAX_COMPLETION, "cache_prompt": False if model_key != "motif" else None,
        "enable_thinking": False if model_key == "motif" else None, "reasoning_present": bool(reasoning),
        "returned_model_id": data.get("model"), "prompt_tokens": usage.get("prompt_tokens"), "total_tokens": usage.get("total_tokens"),
        "http_status": status, "api_failure": status != 200 or error is not None,
        "error_sanitized": sanitize(error, os.environ.get("MORPH_API_KEY", "")) if error else None,
        "truncated": finish == "length" or (raw is None and completion is not None and int(completion) >= MAX_COMPLETION),
        "retry_count": 0, "timestamp": now(), "api_key_recorded": False,
    }


def local_call(model_key: str, run_label: str, task):
    row, condition, _, qid, rotation = task
    messages, _, _ = build_messages(row, condition, qid, rotation)
    cfg = MODELS[model_key]
    payload = {"model": cfg["display"], "messages": messages, "temperature": 0, "top_p": 1.0, "top_k": 1, "seed": 2026, "max_tokens": MAX_COMPLETION, "stream": False, "cache_prompt": False}
    response, latency, error, status = post_json(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions", payload, timeout=180)
    return parse_result(model_key, run_label, response, latency, error, status, task)


def motif_call(run_label: str, task):
    key = os.environ.get("MORPH_API_KEY")
    if not key:
        raise RuntimeError("MORPH_API_KEY missing")
    row, condition, _, qid, rotation = task
    messages, _, _ = build_messages(row, condition, qid, rotation)
    cfg = MODELS["motif"]
    payload = {"model": cfg["model_id"], "messages": messages, "temperature": 0, "top_p": 1.0, "seed": 2026, "max_completion_tokens": MAX_COMPLETION, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    response, latency, error, status = post_json(cfg["base_url"] + "/chat/completions", payload, {"Authorization": "Bearer " + key}, timeout=120)
    return parse_result("motif", run_label, response, latency, error, status, task)


def initialize_preservation() -> None:
    path = AUDITS / "primary_preservation_before.json"
    if not path.exists():
        write_json(path, {"timestamp": now(), "files": prior.snapshot(prior.primary_files())})


def run_target(target: str) -> None:
    FULL.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True); AUDITS.mkdir(parents=True, exist_ok=True)
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    if sha256_file(SAMPLE_PATH) != SAMPLE_SHA256:
        raise RuntimeError("locked sample hash mismatch")
    sample = load_locked_sample()
    for model_key in ("qwen", "exaone", "gemma"):
        if sha256_file(MODELS[model_key]["path"]) != MODELS[model_key]["sha256"]:
            raise RuntimeError(f"{model_key} checkpoint hash mismatch")
    initialize_preservation()
    checkpoint, parquet_path = target_paths(target)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing_all = load_checkpoint(checkpoint)
    failed_existing = [r for r in existing_all if r.get("api_failure")]
    existing = [r for r in existing_all if not r.get("api_failure")]
    if failed_existing:
        archive = checkpoint.parent / "failed_attempts.jsonl"
        with archive.open("a", encoding="utf-8", newline="\n") as handle:
            for row in failed_existing: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        with checkpoint.open("w", encoding="utf-8", newline="\n") as handle:
            for row in existing: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    existing_keys = {(r["uuid"], r["persona_condition"], r["order_condition"], r["question_id"]) for r in existing}
    if len(existing_keys) != len(existing):
        raise RuntimeError(f"duplicate checkpoint rows for {target}")
    tasks = [task for task in all_tasks(sample, target) if key_for(task) not in existing_keys]
    model_key = "motif" if target.startswith("motif") else target
    run_label = "motif_replicate2" if target == "motif_run2" else "primary_supplementary_run1"
    update_resume_event(target, status="running", resume_time=now(), previously_saved=len(existing_all), successful_saved=len(existing), failed_keys_requeued=len(failed_existing), rate_limit_failures_requeued=sum(is_rate_limit(r) for r in failed_existing), pending_keys=len(tasks), last_saved_observation_timestamp=(existing[-1].get("timestamp") if existing else None))
    context = LocalAdapter(model_key, checkpoint.parent / "llama_server.log") if model_key != "motif" and tasks else None
    if context:
        context.__enter__()
    try:
        worker = (lambda task: local_call(model_key, run_label, task)) if model_key != "motif" else (lambda task: motif_call(run_label, task))
        workers = MODELS[model_key].get("parallel", 4) if model_key != "motif" else 4
        with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, result in enumerate(pool.map(worker, tasks), start=len(existing) + 1):
                    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    existing.append(result)
                    if model_key == "motif" and result["returned_model_id"] not in (None, MODELS["motif"]["model_id"]):
                        raise RuntimeError("unexpected Motif returned model ID")
                    if index % 250 == 0 or index == 16000:
                        print(json.dumps({"target": target, "completed": index, "expected": 16000}), flush=True)
    finally:
        if context:
            context.__exit__(None, None, None)
    if len(existing) != 16000:
        raise RuntimeError(f"{target}: expected 16000, found {len(existing)}")
    pq.write_table(pa.Table.from_pylist(existing), parquet_path, compression="zstd")
    write_json(checkpoint.parent / "run_summary.json", {"target": target, "timestamp": now(), "observations": len(existing), "semantic_invalid": sum(not r["semantic_valid"] for r in existing), "format_noncompliant": sum(not r["format_compliant"] for r in existing), "api_failures": sum(r["api_failure"] for r in existing), "reasoning_occurrences": sum(r["reasoning_present"] for r in existing), "truncations": sum(r["truncated"] for r in existing), "prompt_tokens": sum((r["prompt_tokens"] or 0) for r in existing), "completion_tokens": sum((r["completion_tokens"] or 0) for r in existing), "total_tokens": sum((r["total_tokens"] or 0) for r in existing), "checkpoint_sha256": sha256_file(checkpoint), "parquet_sha256": sha256_file(parquet_path), "api_key_exposure": 0})
    update_resume_event(target, status="completed", completion_time=now(), final_successful_keys=len(existing), rate_limit_failures_in_final=sum(is_rate_limit(r) for r in existing), duplicate_keys=len(existing)-len({(r['uuid'],r['question_id'],r['persona_condition'],r['order_condition']) for r in existing}))
    print(json.dumps({"target": target, "complete": len(existing), "parquet": str(parquet_path), "api_key_exposure": 0}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    args = parser.parse_args()
    run_target(args.target)


if __name__ == "__main__":
    main()
