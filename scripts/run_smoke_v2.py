from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "derived" / "smoke_100_seed2026.parquet"
MODEL_PATH = ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"
RUNTIME_DIR = ROOT / "tools" / "llama.cpp" / "b10514"
SERVER_EXE = RUNTIME_DIR / "llama-server.exe"
OUT = ROOT / "outputs"
BENCHMARK_PATH = OUT / "human_benchmark_20260818.json"
RESPONSES_JSONL = OUT / "smoke_v2_responses.jsonl"
RESPONSES_CSV = OUT / "smoke_v2_responses.csv"
REPRO_JSONL = OUT / "smoke_v2_reproducibility.jsonl"
REPORT_PATH = OUT / "smoke_v2_report.md"
CONFIG_PATH = OUT / "smoke_v2_inference_config.json"
ENV_PATH = OUT / "smoke_v2_model_environment.md"
SERVER_LOG = OUT / "smoke_v2_llama_server.log"
PRESERVATION_PATH = OUT / "smoke_v2_preservation_manifest.json"
DIST_PATH = OUT / "smoke_v2_distribution.csv"
BENCHMARK_COMPARISON_PATH = OUT / "smoke_v2_benchmark_comparison.csv"
AGGREGATED_PATH = OUT / "smoke_v2_aggregated_comparison.csv"
CONTEXT_PATH = OUT / "smoke_v2_context_effect.csv"
PERSONA_EFFECT_PATH = OUT / "smoke_v2_persona_effect.csv"

PORT = 8088
CONTEXT_LENGTH = 4096
GPU_LAYERS = 99
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"

PERSONA_CONDITIONS = ["demographics_only", "demographics_plus_persona"]
PROTOCOLS = {
    "full_questionnaire": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "policy_only": ["Q2", "Q3", "Q4", "Q5"],
}

QUESTIONS = {
    "Q1": {
        "text": "이재명 대통령은 최근 저출생에 따른 병력 감소에 대응해, 군을 첨단 기술 중심으로 바꾸는 선택적 모병제를 내년부터 시행하겠다고 밝혔습니다. 선생님께서는 이 발표를 알고 계십니까?",
        "choices": {
            1: "잘 알고 있다",
            2: "들어본 적은 있다",
            3: "잘 모르겠다",
        },
    },
    "Q2": {
        "text": "선택적 모병제는 현행 징병제를 유지하면서, 인공지능과 드론 등 전문 기술 분야에 한해 보수와 혜택을 강화한 기술집약형 부사관을 도입하는 징병제와 모병제의 혼합 방식입니다. 선생님께서는 선택적 모병제 도입에 대해 어떻게 생각하십니까?",
        "choices": {
            1: "매우 찬성한다",
            2: "찬성하는 편이다",
            3: "반대하는 편이다",
            4: "매우 반대한다",
            5: "잘 모르겠다",
        },
    },
    "Q3": {
        "text": "장교와 부사관 지원이 정원에 미치지 못하는 상황이 이어지고 있습니다. 선생님께서는 선택적 모병제가 병력 충원 문제 해결에 도움이 될 것으로 보십니까?",
        "choices": {
            1: "크게 도움될 것이다",
            2: "어느 정도 도움될 것이다",
            3: "별로 도움되지 않을 것이다",
            4: "전혀 도움되지 않을 것이다",
            5: "잘 모르겠다",
        },
    },
    "Q4": {
        "text": "앞으로 우리나라 병역제도가 어떤 방향으로 가야 한다고 생각하십니까?",
        "choices": {
            1: "현행 징병제 유지",
            2: "징병제와 모병제 병행",
            3: "전면 모병제 전환",
            4: "잘 모르겠다",
        },
    },
    "Q5": {
        "text": "병역 자원이 줄어드는 상황에서 여성도 징병 대상에 포함하자는 주장이 있습니다. 선생님께서는 여성 징병제에 대해 어떻게 생각하십니까?",
        "choices": {
            1: "매우 찬성한다",
            2: "찬성하는 편이다",
            3: "반대하는 편이다",
            4: "매우 반대한다",
            5: "잘 모르겠다",
        },
    },
}

SYSTEM_PRINCIPLES = """제공된 합성 인물의 관점에서 설문에 응답한다.
각 문항에서 제공된 선택지 중 하나를 선택한다.
선택 가능한 '잘 모르겠다'를 회피하도록 강요하지 않는다.
특정 정책에 찬성 또는 반대하도록 설득하지 않는다.
제공되지 않은 개인적 사실이나 경험을 만들어내지 않는다.
대한민국 전체 국민을 대표한다고 가정하지 않는다.
설명, 정책 권고, 이유, chain-of-thought를 생성하지 않는다.
각 문항에는 해당 문항의 응답 JSON 객체만 출력한다."""

INFERENCE_CONFIG = {
    "model_repository": "Qwen/Qwen3-8B-GGUF",
    "model_filename": "Qwen3-8B-Q4_K_M.gguf",
    "model_sha256": MODEL_SHA256,
    "quantization": "Q4_K_M",
    "llama_cpp": {"build": 10514, "commit": "7221e24f5"},
    "server": {
        "context_length": CONTEXT_LENGTH,
        "gpu_layers": GPU_LAYERS,
        "flash_attention": True,
        "parallel": 1,
        "reasoning": "off",
        "reasoning_budget": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "request": {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "max_tokens": 32,
        "seed": 2026,
        "cache_prompt": False,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "conversation_protocols": PROTOCOLS,
    "conversation_independence": "Each uuid × persona_condition × survey_protocol starts with a new message list.",
    "question_visibility": "Only the current question is appended; future questions are not shown.",
    "source_sample": "data/derived/smoke_100_seed2026.parquet",
}

PRESERVE_FILES = [
    ROOT / "data" / "derived" / "pilot_1000_seed42.parquet",
    ROOT / "data" / "derived" / "smoke_100_seed2026.parquet",
    OUT / "smoke_100_responses.jsonl",
    OUT / "smoke_100_responses.csv",
    OUT / "smoke_100_reproducibility.jsonl",
    OUT / "smoke_100_sample_profile.md",
    OUT / "smoke_test_report.md",
    OUT / "model_environment.md",
    OUT / "inference_config.json",
    OUT / "llama_server.log",
    ROOT / "scripts" / "run_smoke_test.py",
    ROOT / "scripts" / "create_smoke_sample.py",
]


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_preservation_manifest():
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in PRESERVE_FILES
        if path.exists()
    }


def initialize_or_verify_preservation():
    current = current_preservation_manifest()
    if PRESERVATION_PATH.exists():
        saved = json.loads(PRESERVATION_PATH.read_text(encoding="utf-8"))
        if saved["files"] != current:
            raise RuntimeError("A preserved smoke/pilot file changed since the v2 baseline")
    else:
        PRESERVATION_PATH.write_text(
            json.dumps({"created_for": "smoke_v2", "files": current}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return current


def run_text(command):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags
    ).stdout.strip()


def nvidia_query(fields):
    return [
        part.strip()
        for part in run_text(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).split(",")
    ]


class GpuMonitor:
    def __init__(self):
        name, total, used = nvidia_query("name,memory.total,memory.used")
        self.name = name
        self.total_mib = int(total)
        self.baseline_mib = int(used)
        self.peak_mib = self.baseline_mib
        self.peak_utilization = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                used, utilization = nvidia_query("memory.used,utilization.gpu")
                self.peak_mib = max(self.peak_mib, int(used))
                self.peak_utilization = max(self.peak_utilization, int(utilization))
            except Exception:
                pass
            self.stop_event.wait(0.25)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)


def server_command():
    return [
        str(SERVER_EXE),
        "--model", str(MODEL_PATH),
        "--ctx-size", str(CONTEXT_LENGTH),
        "--n-gpu-layers", str(GPU_LAYERS),
        "--flash-attn", "on",
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--no-webui",
        "--metrics",
        "--reasoning", "off",
        "--reasoning-budget", "0",
        "--chat-template-kwargs", json.dumps({"enable_thinking": False}, separators=(",", ":")),
        "--seed", "2026",
    ]


def start_server(log_handle):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        server_command(), cwd=RUNTIME_DIR, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags
    )


def http_json(path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}; see {SERVER_LOG}")
        try:
            if http_json("/health", timeout=5).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Timed out waiting for llama-server")


def respondent_system_prompt(row, condition):
    fields = [
        ("sex", row["sex"]),
        ("age", row["age"]),
        ("age_group", row["age_group"]),
        ("province", row["province"]),
        ("marital_status", row["marital_status"]),
        ("education_level", row["education_level"]),
        ("occupation", row["occupation"]),
    ]
    if condition == "demographics_plus_persona":
        fields.append(("persona", row["persona"]))
    info = "\n".join(f"- {name}: {value}" for name, value in fields)
    return f"{SYSTEM_PRINCIPLES}\n\n합성 응답자 정보:\n{info}"


def question_prompt(question):
    spec = QUESTIONS[question]
    choices = "\n".join(f"{code} = {label}" for code, label in spec["choices"].items())
    key = question.lower()
    return f"{spec['text']}\n\n{choices}\n\n다음 JSON 객체만 출력하십시오: {{\"{key}\": 1}}"


def response_schema(question):
    key = question.lower()
    return {
        "type": "object",
        "properties": {key: {"type": "integer", "enum": list(QUESTIONS[question]["choices"]) }},
        "required": [key],
        "additionalProperties": False,
    }


def request_payload(messages, question):
    cfg = INFERENCE_CONFIG["request"]
    return {
        "model": "Qwen3-8B-Q4_K_M",
        "messages": messages,
        "temperature": cfg["temperature"],
        "top_k": cfg["top_k"],
        "top_p": cfg["top_p"],
        "min_p": cfg["min_p"],
        "max_tokens": cfg["max_tokens"],
        "seed": cfg["seed"],
        "cache_prompt": cfg["cache_prompt"],
        "stream": cfg["stream"],
        "reasoning_effort": cfg["reasoning_effort"],
        "chat_template_kwargs": cfg["chat_template_kwargs"],
        "response_format": {"type": "json_object", "schema": response_schema(question)},
    }


def parse_output(question, content):
    key = question.lower()
    try:
        parsed = json.loads(content)
        code = parsed[key]
        if set(parsed) == {key} and type(code) is int and code in QUESTIONS[question]["choices"]:
            return code, True
    except Exception:
        pass
    return None, False


def infer(messages, question):
    last_error = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            response = http_json("/v1/chat/completions", request_payload(messages, question), timeout=180)
            elapsed = time.perf_counter() - started
            message = response["choices"][0]["message"]
            content = message.get("content", "")
            code, success = parse_output(question, content)
            return {
                "response_code": code,
                "response_label": QUESTIONS[question]["choices"].get(code),
                "raw_model_output": content,
                "parse_success": success,
                "inference_time_seconds": round(elapsed, 6),
                "reasoning_content_present": bool(message.get("reasoning_content")),
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt)
    return {
        "response_code": None,
        "response_label": None,
        "raw_model_output": "",
        "parse_success": False,
        "inference_time_seconds": None,
        "reasoning_content_present": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "attempts": 3,
        "error": last_error,
    }


def load_sample():
    table = pq.read_table(SAMPLE_PATH)
    if table.num_rows != 100 or len(set(table["uuid"].to_pylist())) != 100:
        raise RuntimeError("The preserved smoke sample is not exactly 100 unique UUIDs")
    return table.to_pylist()


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def conversation_key(row, condition, protocol):
    return str(row["uuid"]), condition, protocol


def run_conversation(row, condition, protocol, completed_by_key=None, checkpoint=True):
    questions = PROTOCOLS[protocol]
    messages = [{"role": "system", "content": respondent_system_prompt(row, condition)}]
    results = []
    existing = completed_by_key.get(conversation_key(row, condition, protocol), {}) if completed_by_key else {}
    existing_questions = [question for question in questions if question in existing]
    if existing_questions != questions[: len(existing_questions)]:
        raise RuntimeError(f"Checkpoint is not a question prefix for {conversation_key(row, condition, protocol)}")

    for question in questions:
        prompt = question_prompt(question)
        messages.append({"role": "user", "content": prompt})
        if question in existing:
            record = existing[question]
            messages.append({"role": "assistant", "content": record["raw_model_output"]})
            results.append(record)
            continue
        inference = infer(messages, question)
        record = {
            "uuid": str(row["uuid"]),
            "persona_condition": condition,
            "survey_protocol": protocol,
            "question": question,
            **inference,
        }
        if checkpoint:
            append_jsonl(RESPONSES_JSONL, record)
        messages.append({"role": "assistant", "content": inference["raw_model_output"]})
        results.append(record)
    return results


def completed_index(records):
    index = defaultdict(dict)
    for record in records:
        key = (record["uuid"], record["persona_condition"], record["survey_protocol"])
        question = record["question"]
        if question in index[key]:
            raise RuntimeError(f"Duplicate response in checkpoint: {key} {question}")
        index[key][question] = record
    return index


def export_csv(path, records, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_distribution(records):
    counters = Counter()
    totals = Counter()
    for record in records:
        if record["parse_success"]:
            key = (record["persona_condition"], record["survey_protocol"], record["question"])
            counters[key + (record["response_code"],)] += 1
            totals[key] += 1
    rows = []
    for condition in PERSONA_CONDITIONS:
        for protocol, questions in PROTOCOLS.items():
            for question in questions:
                key = (condition, protocol, question)
                for code, label in QUESTIONS[question]["choices"].items():
                    count = counters[key + (code,)]
                    rows.append({
                        "persona_condition": condition,
                        "survey_protocol": protocol,
                        "question": question,
                        "response_code": code,
                        "response_label": label,
                        "n": count,
                        "percent": count / totals[key] * 100 if totals[key] else 0,
                    })
    return rows


def dist_lookup(distribution):
    return {
        (row["persona_condition"], row["survey_protocol"], row["question"], int(row["response_code"])): float(row["percent"])
        for row in distribution
    }


def build_benchmark_comparison(distribution, benchmark):
    lookup = dist_lookup(distribution)
    rows = []
    for condition in PERSONA_CONDITIONS:
        for protocol in PROTOCOLS:
            for question in ["Q2", "Q3", "Q4", "Q5"]:
                differences = []
                for code, human_spec in benchmark["overall"][question].items():
                    synthetic = lookup[(condition, protocol, question, int(code))]
                    difference = synthetic - human_spec["percent"]
                    differences.append(abs(difference))
                    rows.append({
                        "persona_condition": condition,
                        "survey_protocol": protocol,
                        "question": question,
                        "response_code": int(code),
                        "response_label": human_spec["label"],
                        "human_percent": human_spec["percent"],
                        "synthetic_percent": synthetic,
                        "percentage_point_difference": difference,
                        "absolute_percentage_point_difference": abs(difference),
                    })
                mae = statistics.fmean(differences)
                tvd = sum(differences) / 200
                for row in rows[-len(differences):]:
                    row["mae_percentage_points"] = mae
                    row["total_variation_distance"] = tvd
    return rows


AGGREGATIONS = {
    "Q2": {"찬성": [1, 2], "반대": [3, 4], "모름": [5]},
    "Q3": {"도움됨": [1, 2], "도움 안 됨": [3, 4], "모름": [5]},
    "Q4": {"현행 징병제 유지": [1], "징병제와 모병제 병행": [2], "전면 모병제 전환": [3], "잘 모르겠다": [4]},
    "Q5": {"찬성": [1, 2], "반대": [3, 4], "모름": [5]},
}


def build_aggregated(distribution, benchmark):
    lookup = dist_lookup(distribution)
    rows = []
    for condition in PERSONA_CONDITIONS:
        for protocol in PROTOCOLS:
            for question, groups in AGGREGATIONS.items():
                for group, codes in groups.items():
                    synthetic = sum(lookup[(condition, protocol, question, code)] for code in codes)
                    human = sum(benchmark["overall"][question][str(code)]["percent"] for code in codes)
                    rows.append({
                        "persona_condition": condition,
                        "survey_protocol": protocol,
                        "question": question,
                        "aggregated_category": group,
                        "human_percent": human,
                        "synthetic_percent": synthetic,
                        "percentage_point_difference": synthetic - human,
                    })
    return rows


def build_pairwise_effect(distribution, dimension):
    lookup = dist_lookup(distribution)
    rows = []
    if dimension == "protocol":
        for condition in PERSONA_CONDITIONS:
            for question in ["Q2", "Q3", "Q4", "Q5"]:
                diffs = []
                start = len(rows)
                for code, label in QUESTIONS[question]["choices"].items():
                    full = lookup[(condition, "full_questionnaire", question, code)]
                    policy = lookup[(condition, "policy_only", question, code)]
                    diff = full - policy
                    diffs.append(abs(diff))
                    rows.append({
                        "persona_condition": condition,
                        "question": question,
                        "response_code": code,
                        "response_label": label,
                        "full_questionnaire_percent": full,
                        "policy_only_percent": policy,
                        "percentage_point_difference_full_minus_policy": diff,
                    })
                for row in rows[start:]:
                    row["mae_percentage_points"] = statistics.fmean(diffs)
                    row["total_variation_distance"] = sum(diffs) / 200
    else:
        for protocol in PROTOCOLS:
            for question in PROTOCOLS[protocol]:
                diffs = []
                start = len(rows)
                for code, label in QUESTIONS[question]["choices"].items():
                    demographic = lookup[("demographics_only", protocol, question, code)]
                    persona = lookup[("demographics_plus_persona", protocol, question, code)]
                    diff = persona - demographic
                    diffs.append(abs(diff))
                    rows.append({
                        "survey_protocol": protocol,
                        "question": question,
                        "response_code": code,
                        "response_label": label,
                        "demographics_only_percent": demographic,
                        "demographics_plus_persona_percent": persona,
                        "percentage_point_difference_persona_minus_demographics": diff,
                    })
                for row in rows[start:]:
                    row["mae_percentage_points"] = statistics.fmean(diffs)
                    row["total_variation_distance"] = sum(diffs) / 200
    return rows


def select_repro_conversations(sample_rows):
    selected = []
    for condition in PERSONA_CONDITIONS:
        for protocol in PROTOCOLS:
            seed_material = f"2026|{condition}|{protocol}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            for row in random.Random(seed).sample(sample_rows, 5):
                selected.append((row, condition, protocol))
    return selected


def run_reproducibility(sample_rows, original_records):
    original = {
        (r["uuid"], r["persona_condition"], r["survey_protocol"], r["question"]): r
        for r in original_records
    }
    rows = []
    for row, condition, protocol in select_repro_conversations(sample_rows):
        repeated = run_conversation(row, condition, protocol, checkpoint=False)
        for record in repeated:
            key = (record["uuid"], condition, protocol, record["question"])
            source = original[key]
            rows.append({
                "uuid": record["uuid"],
                "persona_condition": condition,
                "survey_protocol": protocol,
                "question": record["question"],
                "original_response_code": source["response_code"],
                "repeated_response_code": record["response_code"],
                "response_code_identical": source["response_code"] == record["response_code"],
                "raw_output_identical": source["raw_model_output"] == record["raw_model_output"],
                "repeated_parse_success": record["parse_success"],
                "repeated_inference_time_seconds": record["inference_time_seconds"],
            })
    with REPRO_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows


def markdown_table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(records, distribution, benchmark_rows, aggregated, context_rows, persona_rows, repro_rows, gpu, wall_seconds, total_wall_seconds):
    keys = [(r["uuid"], r["persona_condition"], r["survey_protocol"], r["question"]) for r in records]
    expected = 1800
    parse_failures = sum(not r["parse_success"] for r in records)
    invalid = sum(
        r["parse_success"] and r["response_code"] not in QUESTIONS[r["question"]]["choices"] for r in records
    )
    reasoning = sum(bool(r.get("reasoning_content_present")) for r in records)
    times = [r["inference_time_seconds"] for r in records if r["inference_time_seconds"] is not None]
    distribution_lookup = dist_lookup(distribution)
    benchmark_summary = {}
    for row in benchmark_rows:
        key = (row["persona_condition"], row["survey_protocol"], row["question"])
        benchmark_summary[key] = (row["mae_percentage_points"], row["total_variation_distance"])
    context_summary = {}
    for row in context_rows:
        key = (row["persona_condition"], row["question"])
        context_summary[key] = (row["mae_percentage_points"], row["total_variation_distance"])
    persona_summary = {}
    for row in persona_rows:
        key = (row["survey_protocol"], row["question"])
        persona_summary[key] = (row["mae_percentage_points"], row["total_variation_distance"])
    conv_repro = defaultdict(list)
    for row in repro_rows:
        conv_repro[(row["uuid"], row["persona_condition"], row["survey_protocol"])].append(row)
    code_question_match = sum(r["response_code_identical"] for r in repro_rows)
    raw_question_match = sum(r["raw_output_identical"] for r in repro_rows)
    code_conv_match = sum(all(r["response_code_identical"] for r in rows) for rows in conv_repro.values())
    raw_conv_match = sum(all(r["raw_output_identical"] for r in rows) for rows in conv_repro.values())

    lines = [
        "# Smoke test v2: 실제 질문지 순차 설문",
        "",
        "본 결과는 기술적·방법론적 파이프라인 검증용이다. 100명 합성 표본의 결과를 한국 국민의 여론이나 모델 성능 결론으로 해석하지 않는다.",
        "",
        "## 기술 검증",
        "",
        markdown_table(
            ["항목", "결과"],
            [
                ["독립 conversation", "400"],
                ["예상 question response", f"{expected:,}"],
                ["실제 question response", f"{len(records):,}"],
                ["JSON parse 성공", f"{len(records) - parse_failures:,} ({(len(records)-parse_failures)/len(records):.2%})"],
                ["범위 밖 응답", invalid],
                ["누락", max(0, expected - len(set(keys)))],
                ["중복", len(keys) - len(set(keys))],
                ["평균 inference time", f"{statistics.fmean(times):.4f}초"],
                ["중앙 inference time", f"{statistics.median(times):.4f}초"],
                ["본 실행 wall time", f"{wall_seconds:.3f}초 ({wall_seconds/60:.2f}분)"],
                ["모델 로드·본 실행·재현성 포함 total wall time", f"{total_wall_seconds:.3f}초 ({total_wall_seconds/60:.2f}분)"],
                ["GPU", f"{gpu.name}; peak {gpu.peak_mib:,} MiB; baseline 대비 +{gpu.peak_mib-gpu.baseline_mib:,} MiB"],
                ["reasoning output 감지", reasoning],
            ],
        ),
        "",
        "Q1은 실제 뉴스 인지도로 해석하지 않고 full_questionnaire의 조사 맥락과 순서를 재현하기 위해서만 포함했다.",
        "",
        "## 응답분포",
        "",
    ]
    for condition in PERSONA_CONDITIONS:
        for protocol, questions in PROTOCOLS.items():
            lines.extend([f"### {condition} × {protocol}", ""])
            for question in questions:
                rows = []
                for code, label in QUESTIONS[question]["choices"].items():
                    percent = distribution_lookup[(condition, protocol, question, code)]
                    rows.append([question, code, label, round(percent), f"{percent:.1f}%"])
                lines.extend([markdown_table(["문항", "코드", "선택지", "n", "%"], rows), ""])
    lines.extend(["## Human benchmark 비교: MAE와 TVD", ""])
    lines.append(markdown_table(
        ["persona condition", "protocol", "question", "MAE(pp)", "TVD"],
        [
            [condition, protocol, question, f"{benchmark_summary[(condition, protocol, question)][0]:.3f}", f"{benchmark_summary[(condition, protocol, question)][1]:.4f}"]
            for condition in PERSONA_CONDITIONS for protocol in PROTOCOLS for question in ["Q2", "Q3", "Q4", "Q5"]
        ],
    ))
    lines.extend(["", "범주별 human/synthetic 비율과 absolute percentage-point difference는 `smoke_v2_benchmark_comparison.csv`에 저장했다.", ""])
    lines.extend(["## Aggregated 지표", ""])
    lines.append(markdown_table(
        ["condition", "protocol", "question", "범주", "human %", "synthetic %", "차이(pp)"],
        [[r["persona_condition"], r["survey_protocol"], r["question"], r["aggregated_category"], f"{r['human_percent']:.1f}", f"{r['synthetic_percent']:.1f}", f"{r['percentage_point_difference']:+.1f}"] for r in aggregated],
    ))
    lines.extend(["", "## Survey context effect", ""])
    lines.append(markdown_table(
        ["persona condition", "question", "MAE(pp)", "TVD"],
        [[condition, question, f"{context_summary[(condition, question)][0]:.3f}", f"{context_summary[(condition, question)][1]:.4f}"] for condition in PERSONA_CONDITIONS for question in ["Q2", "Q3", "Q4", "Q5"]],
    ))
    lines.extend(["", "Q2의 범주별 full_questionnaire − policy_only 차이는 `smoke_v2_context_effect.csv`에서 직접 확인할 수 있다.", ""])
    lines.extend(["## Persona input effect", ""])
    lines.append(markdown_table(
        ["protocol", "question", "MAE(pp)", "TVD"],
        [[protocol, question, f"{persona_summary[(protocol, question)][0]:.3f}", f"{persona_summary[(protocol, question)][1]:.4f}"] for protocol, questions in PROTOCOLS.items() for question in questions],
    ))
    lines.extend([
        "", "## 재현성", "",
        f"- 반복 conversation: {len(conv_repro)}/20",
        f"- question-level response code 일치: {code_question_match}/{len(repro_rows)}",
        f"- question-level raw output 일치: {raw_question_match}/{len(repro_rows)}",
        f"- conversation 전체 response code 일치: {code_conv_match}/20",
        f"- conversation 전체 raw output 일치: {raw_conv_match}/20",
        "",
        "## 방법론적 주의", "",
        "- Q1 응답은 합성 persona의 실제 뉴스 인지도를 측정하지 않는다.",
        "- 100명은 분포 비교와 subgroup 추론에 작으므로 human benchmark와의 차이에 통계적 의미를 부여하지 않는다.",
        "- JSON schema는 형식 준수만 보장하며 응답의 인간 타당성을 보장하지 않는다.",
        "- 선택지 집중, '잘 모르겠다' 회피 또는 과다 선택 여부는 본 pilot 확대 전에 검토해야 한다.",
        "- 1,000명 실행은 별도 승인 전까지 시작하지 않는다.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_environment(gpu, model_load_seconds):
    lines = [
        "# Smoke v2 model environment",
        "",
        f"- GPU: {gpu.name}",
        f"- VRAM total: {gpu.total_mib:,} MiB",
        f"- VRAM baseline: {gpu.baseline_mib:,} MiB",
        f"- VRAM peak: {gpu.peak_mib:,} MiB",
        f"- VRAM peak increment: {gpu.peak_mib-gpu.baseline_mib:,} MiB",
        f"- Peak GPU utilization: {gpu.peak_utilization}%",
        "- llama.cpp: build 10514, commit `7221e24f5`",
        "- Model: `Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`",
        f"- Model SHA-256: `{MODEL_SHA256}`",
        "- Quantization: Q4_K_M",
        f"- Context length: {CONTEXT_LENGTH}",
        f"- GPU offload: {GPU_LAYERS} layers requested (all layers)",
        "- temperature=0, top_k=1, cache_prompt=false, seed=2026",
        "- reasoning=off, reasoning_budget=0, enable_thinking=false",
        f"- Model load time: {model_load_seconds:.3f} seconds",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    initialize_or_verify_preservation()
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Existing model SHA-256 mismatch")
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    CONFIG_PATH.write_text(json.dumps(INFERENCE_CONFIG, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample_rows = load_sample()
    gpu = GpuMonitor()
    gpu.start()
    server = None
    server_start = time.perf_counter()
    with SERVER_LOG.open("w", encoding="utf-8", errors="replace") as log_handle:
        try:
            server = start_server(log_handle)
            wait_for_server(server)
            model_load_seconds = time.perf_counter() - server_start
            if args.probe_only:
                probe = run_conversation(sample_rows[0], "demographics_only", "full_questionnaire", checkpoint=False)
                print(json.dumps(probe, ensure_ascii=False, indent=2))
                return

            existing = read_jsonl(RESPONSES_JSONL)
            index = completed_index(existing)
            main_start = time.perf_counter()
            conversation_number = 0
            for row in sample_rows:
                for condition in PERSONA_CONDITIONS:
                    for protocol in PROTOCOLS:
                        run_conversation(row, condition, protocol, index, checkpoint=True)
                        conversation_number += 1
                        if conversation_number % 10 == 0:
                            current = len(read_jsonl(RESPONSES_JSONL))
                            print(f"conversation={conversation_number}/400 responses={current}/1800", flush=True)
            wall_seconds = time.perf_counter() - main_start
            records = read_jsonl(RESPONSES_JSONL)
            repro_rows = run_reproducibility(sample_rows, records)
            total_wall_seconds = time.perf_counter() - server_start
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            gpu.stop()

    write_environment(gpu, model_load_seconds)
    if args.probe_only:
        return

    response_fields = [
        "uuid", "persona_condition", "survey_protocol", "question", "response_code", "response_label",
        "raw_model_output", "parse_success", "inference_time_seconds", "reasoning_content_present",
        "prompt_tokens", "completion_tokens", "attempts", "error",
    ]
    export_csv(RESPONSES_CSV, records, response_fields)
    distribution = build_distribution(records)
    export_csv(DIST_PATH, distribution, list(distribution[0]))
    benchmark_rows = build_benchmark_comparison(distribution, benchmark)
    export_csv(BENCHMARK_COMPARISON_PATH, benchmark_rows, list(benchmark_rows[0]))
    aggregated = build_aggregated(distribution, benchmark)
    export_csv(AGGREGATED_PATH, aggregated, list(aggregated[0]))
    context_rows = build_pairwise_effect(distribution, "protocol")
    export_csv(CONTEXT_PATH, context_rows, list(context_rows[0]))
    persona_rows = build_pairwise_effect(distribution, "persona")
    export_csv(PERSONA_EFFECT_PATH, persona_rows, list(persona_rows[0]))
    write_report(records, distribution, benchmark_rows, aggregated, context_rows, persona_rows, repro_rows, gpu, wall_seconds, total_wall_seconds)
    initialize_or_verify_preservation()
    print("SMOKE_V2_COMPLETE")


if __name__ == "__main__":
    main()
