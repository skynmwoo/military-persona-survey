from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_choice_common import *
from model_adapters import LocalAdapter, post_json, sanitize
import run_hard_choice_pilot10_v2 as prior

AUDIT_DIR = OUT / "final_stability_audit"
PRIOR_PARQUET = OUT / "technical_pilot_10_nonthinking" / "pilot10_nonthinking_observations.parquet"
AMENDMENT = OUT / "hard_choice_protocol_amendment_03_parser_and_stability.md"
RUN_ID = "final_format_stability_audit_v1"
MAX_COMPLETION = 32


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_parse(raw: str | None, qid: str) -> dict[str, Any]:
    text = (raw or "").strip()
    valid_labels = set(LABELS[:len(QUESTIONS[qid]["choices"])])
    if text in valid_labels:
        return {"semantic_label": text, "semantic_valid": True, "format_compliant": True, "parser_reason": "exact_single_label"}
    candidate = None
    direct = re.match(r"^([A-E])(?=$|[\s\.:=\)])", text)
    wrapped = re.match(r"^[\(\[\{<]\s*([A-E])(?:\s*[\)\]\}>])?", text)
    if direct:
        candidate = direct.group(1)
    elif wrapped:
        candidate = wrapped.group(1)
    if candidate not in valid_labels:
        return {"semantic_label": None, "semantic_valid": False, "format_compliant": False, "parser_reason": "no_valid_leading_label"}
    tokens = re.findall(r"(?<![A-Za-z0-9])([A-E])(?![A-Za-z0-9])", text)
    conflicting = {token for token in tokens if token != candidate}
    if conflicting:
        return {"semantic_label": None, "semantic_valid": False, "format_compliant": False, "parser_reason": "conflicting_labels"}
    return {"semantic_label": candidate, "semantic_valid": True, "format_compliant": False, "parser_reason": "label_leading_nonexact"}


def reanalyze_prior() -> dict[str, Any]:
    rows = pq.read_table(PRIOR_PARQUET).to_pylist()
    by_model = {}
    enriched = []
    for row in rows:
        parsed = semantic_parse(row.get("raw_model_response"), row["question_id"])
        enriched.append((row, parsed))
    for model in [MODELS[k]["display"] for k in MODELS]:
        subset = [(r, p) for r, p in enriched if r["model"] == model]
        by_model[model] = {
            "total": len(subset),
            "semantic_valid_count": sum(p["semantic_valid"] for _, p in subset),
            "semantic_valid_rate": sum(p["semantic_valid"] for _, p in subset) / len(subset),
            "format_compliant_count": sum(p["format_compliant"] for _, p in subset),
            "format_compliant_rate": sum(p["format_compliant"] for _, p in subset) / len(subset),
            "newly_semantic_valid_count": sum(p["semantic_valid"] and not r["valid_response"] for r, p in subset),
            "remaining_invalid_count": sum(not p["semantic_valid"] for _, p in subset),
        }
    return {
        "source": str(PRIOR_PARQUET.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(PRIOR_PARQUET),
        "total": len(rows),
        "semantic_valid_count": sum(p["semantic_valid"] for _, p in enriched),
        "semantic_valid_rate": sum(p["semantic_valid"] for _, p in enriched) / len(rows),
        "format_compliant_count": sum(p["format_compliant"] for _, p in enriched),
        "format_compliant_rate": sum(p["format_compliant"] for _, p in enriched) / len(rows),
        "newly_semantic_valid_count": sum(p["semantic_valid"] and not r["valid_response"] for r, p in enriched),
        "remaining_invalid_count": sum(not p["semantic_valid"] for _, p in enriched),
        "models": by_model,
        "inference_rerun": False,
        "human_benchmark_accessed": False,
    }


def select_configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: hashlib.sha256(("stability-v1|" + str(r["uuid"])).encode()).hexdigest())[:50]
    assignments = rotation_assignments(rows)
    configs = []
    for index, row in enumerate(ordered):
        qid = QUESTION_IDS[index % 4]
        condition = CONDITIONS[index % 2]
        order_condition = ORDER_CONDITIONS[(index // 2) % 2]
        rotation = 0 if order_condition == "original_order" else assignments[(str(row["uuid"]), qid)]
        config_id = f"cfg_{index + 1:03d}"
        configs.append({"configuration_id": config_id, "row": row, "uuid": str(row["uuid"]), "question_id": qid, "persona_condition": condition, "order_condition": order_condition, "rotation_id": rotation})
    return configs


def parse_response_v2(model_key: str, response, latency: float, error, status, config, repeat_id: int) -> dict[str, Any]:
    data = response or {}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    parsed = semantic_parse(raw, config["question_id"])
    _, order, mapping = build_messages(config["row"], config["persona_condition"], config["question_id"], config["rotation_id"])
    semantic_choice = mapping.get(parsed["semantic_label"]) if parsed["semantic_label"] else None
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason")
    completion = usage.get("completion_tokens")
    return {
        "run_id": RUN_ID, "model": MODELS[model_key]["display"], "model_key": model_key,
        "configuration_id": config["configuration_id"], "repeat_id": repeat_id,
        "uuid": config["uuid"], "question_id": config["question_id"], "persona_condition": config["persona_condition"],
        "order_condition": config["order_condition"], "rotation_id": config["rotation_id"],
        "displayed_semantic_order": ",".join(map(str, order)), "label_to_semantic_mapping": mapping_json(mapping),
        "mapping_bijection": len(mapping) == len(set(mapping.values())) == len(QUESTIONS[config["question_id"]]["choices"]),
        "raw_content": raw, "semantic_label": parsed["semantic_label"], "semantic_choice": semantic_choice,
        "semantic_valid": parsed["semantic_valid"], "format_compliant": parsed["format_compliant"], "parser_reason": parsed["parser_reason"],
        "completion_tokens": completion, "finish_reason": finish, "latency_seconds": round(latency, 6),
        "returned_model_id": data.get("model"), "reasoning_present": bool(reasoning),
        "prompt_tokens": usage.get("prompt_tokens"), "total_tokens": usage.get("total_tokens"),
        "http_status": status, "api_failure": status != 200 or error is not None,
        "error_sanitized": sanitize(error, os.environ.get("MORPH_API_KEY", "")) if error else None,
        "truncated": finish == "length" or (raw is None and completion is not None and int(completion) >= MAX_COMPLETION),
        "temperature": 0, "top_p": 1.0, "top_k": 1 if model_key != "motif" else None, "seed": 2026,
        "max_completion_allowance": MAX_COMPLETION, "enable_thinking": False if model_key == "motif" else None,
        "timestamp": now(), "api_key_recorded": False,
    }


def call_local(model_key: str, config, repeat_id: int) -> dict[str, Any]:
    messages, _, _ = build_messages(config["row"], config["persona_condition"], config["question_id"], config["rotation_id"])
    cfg = MODELS[model_key]
    payload = {"model": cfg["display"], "messages": messages, "temperature": 0, "top_p": 1.0, "top_k": 1, "seed": 2026, "max_tokens": MAX_COMPLETION, "stream": False, "cache_prompt": False}
    response, latency, error, status = post_json(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions", payload)
    return parse_response_v2(model_key, response, latency, error, status, config, repeat_id)


def call_motif(config, repeat_id: int) -> dict[str, Any]:
    key = os.environ.get("MORPH_API_KEY")
    if not key:
        raise RuntimeError("MORPH_API_KEY missing")
    messages, _, _ = build_messages(config["row"], config["persona_condition"], config["question_id"], config["rotation_id"])
    cfg = MODELS["motif"]
    payload = {"model": cfg["model_id"], "messages": messages, "temperature": 0, "top_p": 1.0, "seed": 2026, "max_completion_tokens": MAX_COMPLETION, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    response, latency, error, status = post_json(cfg["base_url"] + "/chat/completions", payload, {"Authorization": "Bearer " + key}, timeout=120)
    return parse_response_v2("motif", response, latency, error, status, config, repeat_id)


def percentile_nearest(values: list[int | float], percentile: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize_model(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration_id"]].append(row)
    exact_agree = semantic_agree = disagreements = 0
    modal_props = []
    unique_counts = {}
    for config_id, group in grouped.items():
        raw_values = [r["raw_content"] for r in group]
        semantic_values = [r["semantic_choice"] for r in group]
        exact_agree += len(set(raw_values)) == 1
        valid_semantics = [v for v in semantic_values if v is not None]
        semantic_same = len(valid_semantics) == repeats and len(set(valid_semantics)) == 1
        semantic_agree += semantic_same
        disagreements += not semantic_same
        counts = Counter(valid_semantics)
        modal_props.append((max(counts.values()) / repeats) if counts else 0.0)
        unique_counts[config_id] = len(set(valid_semantics))
    completion = [int(r["completion_tokens"]) for r in rows if r["completion_tokens"] is not None]
    latency = [float(r["latency_seconds"]) for r in rows]
    return {
        "requests": len(rows), "configurations": len(grouped),
        "semantic_valid_count": sum(r["semantic_valid"] for r in rows), "semantic_valid_rate": sum(r["semantic_valid"] for r in rows) / len(rows),
        "format_compliant_count": sum(r["format_compliant"] for r in rows), "format_compliant_rate": sum(r["format_compliant"] for r in rows) / len(rows),
        "within_configuration_exact_agreement_count": exact_agree, "within_configuration_exact_agreement_rate": exact_agree / len(grouped),
        "within_configuration_semantic_agreement_count": semantic_agree, "within_configuration_semantic_agreement_rate": semantic_agree / len(grouped),
        "mean_modal_choice_proportion": statistics.mean(modal_props), "configurations_with_any_semantic_disagreement": disagreements,
        "semantic_disagreement_configuration_rate": disagreements / len(grouped), "unique_semantic_choice_count_by_configuration": unique_counts,
        "completion_tokens": {"mean": statistics.mean(completion), "median": statistics.median(completion), "p95": percentile_nearest(completion, .95), "max": max(completion)},
        "latency_seconds": {"mean": statistics.mean(latency), "median": statistics.median(latency), "p95": percentile_nearest(latency, .95), "max": max(latency)},
        "finish_reason_distribution": dict(Counter(r["finish_reason"] for r in rows)),
        "reasoning_occurrence_count": sum(r["reasoning_present"] for r in rows), "content_none_count": sum(r["raw_content"] is None for r in rows),
        "truncation_count": sum(r["truncated"] for r in rows), "api_failure_count": sum(r["api_failure"] for r in rows),
        "mapping_failure_count": sum(not r["mapping_bijection"] for r in rows),
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if not AMENDMENT.exists():
        raise RuntimeError("Amendment 03 missing")
    sample = load_locked_sample()
    if sha256_file(SAMPLE_PATH) != "6650945363b9cd2d7ca702ec98e900aa16cfada714642af3521a2c0e895a27f7":
        raise RuntimeError("locked sample hash mismatch")
    for key, cfg in MODELS.items():
        if cfg["kind"] == "local" and sha256_file(cfg["path"]) != cfg["sha256"]:
            raise RuntimeError(f"{key} model hash mismatch")
    prior_result = reanalyze_prior()
    write_json(AUDIT_DIR / "parser_v2_reanalysis.json", prior_result)
    configs = select_configurations(sample)
    public_configs = [{k: v for k, v in c.items() if k != "row"} for c in configs]
    config_hash = hashlib.sha256(canonical_json(public_configs).encode("utf-8")).hexdigest()
    preserved = prior.primary_files()
    before = prior.snapshot(preserved)
    observations = []
    for model_key in ("qwen", "exaone", "gemma"):
        cfg = MODELS[model_key]
        with LocalAdapter(model_key, AUDIT_DIR / f"{model_key}_server.log"):
            jobs = [(config, repeat) for config in configs for repeat in (1, 2)]
            with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
                observations.extend(pool.map(lambda job: call_local(model_key, job[0], job[1]), jobs))
        print(json.dumps({"model": model_key, "requests": 100}), flush=True)
    motif_jobs = [(config, repeat) for config in configs for repeat in range(1, 6)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations.extend(pool.map(lambda job: call_motif(job[0], job[1]), motif_jobs))
    print(json.dumps({"model": "motif", "requests": 250}), flush=True)
    if len(observations) != 550:
        raise RuntimeError(f"expected 550 observations, found {len(observations)}")
    pq.write_table(pa.Table.from_pylist(observations), AUDIT_DIR / "stability_observations.parquet", compression="zstd")
    summaries = {}
    for model_key in MODELS:
        subset = [r for r in observations if r["model_key"] == model_key]
        summaries[MODELS[model_key]["display"]] = summarize_model(subset, 5 if model_key == "motif" else 2)
    motif = summaries[MODELS["motif"]["display"]]
    structural_pass = motif["semantic_valid_rate"] >= .99 and motif["reasoning_occurrence_count"] == 0 and motif["truncation_count"] == 0 and motif["api_failure_count"] == 0 and motif["mapping_failure_count"] == 0
    rate = motif["semantic_disagreement_configuration_rate"]
    if not structural_pass or rate > .20:
        decision = "NO-GO"
    elif rate > .05:
        decision = "GO WITH STOCHASTICITY LIMITATION"
    else:
        decision = "GO"
    after = prior.snapshot(preserved)
    modified = sorted(key for key in before if before[key] != after[key])
    if modified:
        decision = "NO-GO"
    usage = {"prompt_tokens": sum((r["prompt_tokens"] or 0) for r in observations if r["model_key"] == "motif"), "completion_tokens": sum((r["completion_tokens"] or 0) for r in observations if r["model_key"] == "motif"), "total_tokens": sum((r["total_tokens"] or 0) for r in observations if r["model_key"] == "motif")}
    summary = {"run_id": RUN_ID, "experiment": EXTENSION_NAME, "configuration_selection_rule": "first 50 UUIDs after SHA256('stability-v1|' + uuid) sort; question cycles Q2-Q5; persona condition alternates; order condition alternates in two-person blocks", "configuration_count": 50, "configuration_sha256": config_hash, "configuration_axis_counts": {"question": dict(Counter(c["question_id"] for c in configs)), "persona_condition": dict(Counter(c["persona_condition"] for c in configs)), "order_condition": dict(Counter(c["order_condition"] for c in configs))}, "expected_requests": 550, "actual_requests": len(observations), "models": summaries, "motif_token_usage": usage, "decision_thresholds": {"go_max_disagreement_rate": .05, "go_with_limitation_max_disagreement_rate": .20, "minimum_semantic_valid_rate": .99}, "decision": decision, "full_run_executed": False, "human_benchmark_accessed": False, "primary_files_modified": modified, "api_key_exposure": 0}
    write_json(AUDIT_DIR / "stability_summary.json", summary)
    motif_lines = ["# Motif Stability Report", "", "No human benchmark was read or compared.", "", f"- Configurations / requests: 50 / 250", f"- Semantic-valid rate: {motif['semantic_valid_rate']:.1%}", f"- Format-compliant rate: {motif['format_compliant_rate']:.1%}", f"- Configurations with semantic disagreement: {motif['configurations_with_any_semantic_disagreement']}/50 ({rate:.1%})", f"- Mean modal-choice proportion: {motif['mean_modal_choice_proportion']:.1%}", f"- Reasoning / truncation / API failure: {motif['reasoning_occurrence_count']} / {motif['truncation_count']} / {motif['api_failure_count']}", f"- Decision: **{decision}**"]
    (AUDIT_DIR / "motif_stability_report.md").write_text("\n".join(motif_lines) + "\n", encoding="utf-8")
    table = ["# Four-Model Stability Report", "", "Technical stability only; no human-benchmark metrics were calculated.", "", "| Model | Semantic valid | Format compliant | Exact agreement configs | Semantic agreement configs | Mean modal proportion |", "|---|---:|---:|---:|---:|---:|"]
    for model, value in summaries.items():
        table.append(f"| {model} | {value['semantic_valid_rate']:.1%} | {value['format_compliant_rate']:.1%} | {value['within_configuration_exact_agreement_count']}/50 | {value['within_configuration_semantic_agreement_count']}/50 | {value['mean_modal_choice_proportion']:.1%} |")
    table.extend(["", f"Final technical decision: **{decision}**."])
    (AUDIT_DIR / "four_model_stability_report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    script_paths = [Path(__file__), Path(__file__).parent / "hard_choice_common.py", Path(__file__).parent / "model_adapters.py"]
    manifest = {"run_id": RUN_ID, "timestamp": now(), "amendment_sha256": sha256_file(AMENDMENT), "locked_sample_sha256": sha256_file(SAMPLE_PATH), "prior_pilot_sha256": sha256_file(PRIOR_PARQUET), "configuration_sha256": config_hash, "scripts_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in script_paths}, "model_identifiers": {k: (v.get("sha256") or v.get("model_id")) for k, v in MODELS.items()}, "request_count": len(observations), "motif_request_count": 250, "decision": decision, "primary_files_modified": modified, "human_benchmark_accessed": False, "api_key_exposure": 0}
    write_json(AUDIT_DIR / "run_manifest.json", manifest)
    print(json.dumps({"requests": len(observations), "decision": decision, "motif_disagreement_configs": motif["configurations_with_any_semantic_disagreement"], "human_benchmark_accessed": False, "api_key_exposure": 0}), flush=True)


if __name__ == "__main__":
    main()
