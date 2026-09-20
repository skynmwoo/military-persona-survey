from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_smoke_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SAMPLE = ROOT / "data" / "derived" / "smoke_100_seed2026.parquet"
MODEL = ROOT / "models" / "google" / "gemma-3-12b-it-qat-q4_0-gguf" / "gemma-3-12b-it-q4_0.gguf"
RUNTIME = ROOT / "tools" / "llama.cpp" / "b10514"
SERVER = RUNTIME / "llama-server.exe"
BENCHMARK = OUT / "human_benchmark_20260818.json"
PORT = 8092

CHECKPOINT = OUT / "smoke_v5_gemma_choice_probabilities_checkpoint.jsonl"
REVERSE_CHECKPOINT = OUT / "smoke_v5_gemma_reverse_order_checkpoint.jsonl"
BALANCED_CHECKPOINT = OUT / "smoke_v5_gemma_balanced_order_checkpoint.jsonl"
CHOICE_JSONL = OUT / "smoke_v5_gemma_choice_probabilities.jsonl"
CHOICE_CSV = OUT / "smoke_v5_gemma_choice_probabilities.csv"
POPULATION_CSV = OUT / "smoke_v5_gemma_population_distribution.csv"
BENCHMARK_CSV = OUT / "smoke_v5_gemma_benchmark_comparison.csv"
ENTROPY_CSV = OUT / "smoke_v5_gemma_entropy_margin.csv"
SENSITIVITY_CSV = OUT / "smoke_v5_gemma_persona_sensitivity.csv"
ORDER_CSV = OUT / "smoke_v5_gemma_order_effect.csv"
POSITION_CSV = OUT / "smoke_v5_position_bias.csv"
CROSS_MODEL_CSV = OUT / "smoke_v5_three_model_comparison.csv"
REPORT = OUT / "smoke_v5_report.md"
TOKEN_REPORT = OUT / "smoke_v5_gemma_token_validation.md"
CONFIG = OUT / "smoke_v5_gemma_inference_config.json"
ENVIRONMENT = OUT / "smoke_v5_gemma_environment.md"
SERVER_LOG = OUT / "smoke_v5_gemma_llama_server.log"
PRESERVATION = OUT / "smoke_v5_preservation_manifest.json"

CONDITIONS = ["demographics_only", "demographics_plus_persona"]
QUESTIONS = ["Q2", "Q3", "Q4", "Q5"]
TOKEN_IDS = {}
RESPONSE_LABELS = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
MODEL_SHA = "dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd"

SYSTEM_PROMPT = """제공된 합성 인물의 관점에서 설문에 응답한다.
제공되지 않은 새로운 생애사, 사건, 경험 등의 사실을 만들어내지 않는다.
그러나 프로필에 해당 정책에 대한 의견이 명시되어 있지 않다는 이유만으로 자동으로 '잘 모르겠다'를 선택하지 않는다.
실제 사람이 설문에 응답하듯 제공된 특성을 바탕으로 가장 자연스러운 정책 선호를 판단한다.
'잘 모르겠다'는 정상적인 선택지이며 실제로 판단이 불확실한 경우 선택할 수 있다.
특정 응답으로 유도하지 않는다.
이유나 chain-of-thought를 생성하지 않는다.
현재 문항의 선택지 코드 하나만 JSON 값으로 선택한다."""

CONFIG_DATA = {
    "experiment": "smoke_v5_gemma_and_balanced_order",
    "sample": "data/derived/smoke_100_seed2026.parquet",
    "sample_size": 100,
    "persona_conditions": CONDITIONS,
    "questions": QUESTIONS,
    "question_context": "Each persona_condition × question is an independent fresh conversation; Q1 is excluded.",
    "model": "google/gemma-3-12b-it-qat-q4_0-gguf/gemma-3-12b-it-q4_0.gguf",
    "model_sha256": MODEL_SHA,
    "llama_cpp": {"build": 10514, "commit": "7221e24f5"},
    "server": {"context_length": 4096, "gpu_layers": 99, "flash_attention": True, "reasoning": "off"},
    "logprob_extraction": {
        "endpoint": "/completion",
        "chat_template_endpoint": "/apply-template",
        "assistant_prefix": "{\"qN\": ",
        "n_predict": 1,
        "temperature": -1,
        "samplers": ["temperature"],
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "n_probs_initial": 64,
        "n_probs_fallback": 512,
        "cache_prompt": False,
        "seed": 2026,
        "raw_logits_available": False,
        "logit_columns": "relative logits = raw server logprob minus maximum valid-choice logprob",
        "choice_probability": "softmax over valid-choice relative logits only",
    },
    "categorical_simulation": {"library": "NumPy", "rng": "default_rng", "seed": 2026, "draws_per_row": 1},
    "order_effect": {"sample_size": 100, "seed": 2026, "questions": QUESTIONS, "orders": ["original", "reversed", "uuid_assigned_cyclic_rotation"],
                     "assignment_rule": "For each persona_condition×question, sort all 100 UUIDs by SHA256('2026|condition|question|uuid'), then assign rotation=index modulo K; exactly 20 per position for K=5 and 25 per position for K=4."},
    "benchmark_usage": "evaluation only; not included in prompts or used for model/prompt selection",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preserved_paths():
    paths = [ROOT / "data" / "derived" / "pilot_1000_seed42.parquet", SAMPLE]
    paths += sorted(p for p in OUT.glob("smoke_100*") if p.is_file())
    paths += sorted(p for p in OUT.glob("smoke_v2*") if p.is_file())
    paths += sorted(p for p in OUT.glob("smoke_v3*") if p.is_file())
    paths += sorted(p for p in OUT.glob("smoke_v4*") if p.is_file())
    paths += [
        OUT / "smoke_test_report.md", OUT / "model_environment.md", OUT / "inference_config.json",
        OUT / "llama_server.log", BENCHMARK,
        ROOT / "scripts" / "run_smoke_test.py", ROOT / "scripts" / "create_smoke_sample.py",
        ROOT / "scripts" / "run_smoke_v2.py", ROOT / "scripts" / "probe_smoke_v3_api.py",
        ROOT / "scripts" / "run_smoke_v3_choice_probabilities.py",
        ROOT / "scripts" / "run_smoke_v4_cross_model.py",
    ]
    return list(dict.fromkeys(p for p in paths if p.exists() and not p.name.startswith("smoke_v5")))


def preservation_manifest():
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in preserved_paths()
    }


def init_or_verify_preservation():
    current = preservation_manifest()
    if PRESERVATION.exists():
        saved = json.loads(PRESERVATION.read_text(encoding="utf-8"))["files"]
        if saved != current:
            raise RuntimeError("A preserved smoke/v2/pilot file changed")
    else:
        PRESERVATION.write_text(json.dumps({"files": current}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(current)


def run_text(command):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags).stdout.strip()


def nvidia_query(fields):
    return [x.strip() for x in run_text(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).split(",")]


class GpuMonitor:
    def __init__(self):
        name, total, used = nvidia_query("name,memory.total,memory.used")
        self.name, self.total, self.baseline = name, int(total), int(used)
        self.peak, self.util = self.baseline, 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                used, util = nvidia_query("memory.used,utilization.gpu")
                self.peak, self.util = max(self.peak, int(used)), max(self.util, int(util))
            except Exception:
                pass
            self.stop_event.wait(0.25)

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=3)


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def start_server(log_handle):
    command = [
        str(SERVER), "--model", str(MODEL), "--ctx-size", "4096", "--n-gpu-layers", "99",
        "--flash-attn", "on", "--parallel", "1", "--host", "127.0.0.1", "--port", str(PORT),
        "--no-webui", "--reasoning", "off", "--reasoning-budget", "0",
        "--chat-template-kwargs", '{"enable_thinking":false}', "--seed", "2026",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(command, cwd=RUNTIME, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags)


def wait_server(process):
    deadline = time.time() + 240
    while time.time() < deadline:
        if process.poll() is not None: raise RuntimeError(f"server exited {process.returncode}")
        try:
            if get("/health").get("status") == "ok": return
        except Exception: pass
        time.sleep(1)
    raise TimeoutError("server load timeout")


def profile_system(row, condition):
    fields = [
        ("sex", row["sex"]), ("age", row["age"]), ("age_group", row["age_group"]),
        ("province", row["province"]), ("marital_status", row["marital_status"]),
        ("education_level", row["education_level"]), ("occupation", row["occupation"]),
    ]
    if condition == "demographics_plus_persona": fields.append(("persona", row["persona"]))
    return SYSTEM_PROMPT + "\n\n합성 응답자 정보:\n" + "\n".join(f"- {k}: {v}" for k, v in fields)


def question_text(question, order_codes=None):
    spec = v2.QUESTIONS[question]
    order_codes = order_codes or list(spec["choices"])
    items = [(code, spec["choices"][code]) for code in order_codes]
    choices = "\n".join(f"{RESPONSE_LABELS[code]} = {label}" for code, label in items)
    return f"{spec['text']}\n\n{choices}\n\n선택지 코드 하나를 JSON 값으로 선택하십시오. 이유는 쓰지 마십시오."


def validate_tokens():
    observed = {}
    for label in ["1", "2", "3", "4", "5", "A", "B", "C", "D", "E"]:
        observed[label] = post("/tokenize", {"content": label, "add_special": False, "with_pieces": True})["tokens"]
    numeric_ok = all(len(observed[str(code)]) == 1 for code in range(1, 6))
    alpha_ok = all(len(observed[label]) == 1 for label in "ABCDE")
    if not numeric_ok and not alpha_ok:
        raise RuntimeError(f"Neither numeric nor A-E labels are all single tokens: {observed}")
    if numeric_ok:
        RESPONSE_LABELS.update({code: str(code) for code in range(1, 6)})
    TOKEN_IDS.update({code: observed[RESPONSE_LABELS[code]][0]["id"] for code in range(1, 6)})
    lines = [
        "# Smoke v5 Gemma choice token validation", "",
        "- llama.cpp: b10514, commit `7221e24f5`", "- Model: `gemma-3-12b-it-q4_0.gguf`",
        "- `/tokenize` was called with `add_special=false`.",
        "- Numeric labels 1–5 each tokenize as a common space token plus a digit token and are not directly comparable as one token.",
        "- A–E are each one token and are used as a preregistered fallback with fixed mapping A=semantic category 1, ..., E=5.", "",
        "| Label | Token ID | Piece | Single token |", "| --- | ---: | --- | --- |",
    ]
    for code in range(1, 6):
        label = RESPONSE_LABELS[code]; token = observed[label][0]
        lines.append(f"| {label} (semantic {code}) | {token['id']} | `{token['piece']}` | yes |")
    lines += [
        "", "## API extraction", "",
        "- `/apply-template` applies the official chat template embedded in the Google Gemma GGUF; no Qwen/EXAONE template is supplied.",
        "- `/completion` generates exactly one A–E label token with a constrained grammar and returns `completion_probabilities.top_logprobs`.",
        "- Raw logits are not exposed by b10514 server. Direct server logprobs are stored; relative logits are exactly `choice logprob - max(valid choice logprob)`.",
        "- Choice probabilities are a fresh softmax over valid-choice relative logits only.", "",
    ]
    TOKEN_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return observed


def extract_choice(row, condition, question, order_type="original", order_codes=None, n_probs=64):
    valid_codes = list(v2.QUESTIONS[question]["choices"])
    order_codes = order_codes or valid_codes
    messages = [
        {"role": "system", "content": profile_system(row, condition)},
        {"role": "user", "content": question_text(question, order_codes)},
    ]
    start = time.perf_counter()
    rendered = post("/apply-template", {"messages": messages})["prompt"]
    prefix = rendered + '{"' + question.lower() + '":'
    allowed = "".join(RESPONSE_LABELS[code] for code in valid_codes)
    grammar = f'root ::= " " [{allowed}]'
    completion = post("/completion", {
        "prompt": prefix, "n_predict": 1, "temperature": -1, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "samplers": ["temperature"], "grammar": grammar, "n_probs": n_probs,
        "post_sampling_probs": False, "return_tokens": True, "cache_prompt": False, "seed": 2026,
    })
    elapsed = time.perf_counter() - start
    top = completion["completion_probabilities"][0]["top_logprobs"]
    by_id = {entry["id"]: entry for entry in top}
    missing = [code for code in valid_codes if TOKEN_IDS[code] not in by_id]
    if missing and n_probs < 512:
        return extract_choice(row, condition, question, order_type, order_codes, n_probs=512)
    if missing:
        raise RuntimeError(f"Choice tokens missing from top logprobs for {question}: {missing}")
    logprobs = [float(by_id[TOKEN_IDS[code]]["logprob"]) for code in valid_codes]
    max_lp = max(logprobs)
    relative = [lp - max_lp for lp in logprobs]
    weights = [math.exp(value) for value in relative]
    denom = sum(weights)
    probabilities = [value / denom for value in weights]
    probabilities[-1] = 1.0 - sum(probabilities[:-1])
    greedy_index = max(range(len(valid_codes)), key=lambda i: (probabilities[i], -valid_codes[i]))
    result = {
        "uuid": str(row["uuid"]), "model": "Gemma-3-12B-QAT-Q4_0", "persona_condition": condition, "question": question,
        "choice_order": order_type, "display_order": ",".join(map(str, order_codes)), "greedy_choice": valid_codes[greedy_index],
        "api_generated_token_id": completion["tokens"][0], "api_generated_content": completion["content"],
        "inference_time_seconds": round(elapsed, 6), "n_probs_used": n_probs,
    }
    for code in range(1, 6):
        if code in valid_codes:
            idx = valid_codes.index(code)
            result[f"token_id_choice_{code}"] = TOKEN_IDS[code]
            result[f"relative_logit_choice_{code}"] = relative[idx]
            result[f"logprob_choice_{code}"] = logprobs[idx]
            result[f"prob_choice_{code}"] = probabilities[idx]
        else:
            result[f"token_id_choice_{code}"] = None
            result[f"relative_logit_choice_{code}"] = None
            result[f"logprob_choice_{code}"] = None
            result[f"prob_choice_{code}"] = None
        result[f"display_position_choice_{code}"] = order_codes.index(code) + 1 if code in valid_codes else None
    result["max_probability"] = max(probabilities)
    result["entropy_nats"] = -sum(p * math.log(p) for p in probabilities if p > 0)
    result["top1_top2_margin"] = sorted(probabilities, reverse=True)[0] - sorted(probabilities, reverse=True)[1]
    return result


def read_jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"); f.flush(); os.fsync(f.fileno())


def export_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def finalize_choice_rows(rows):
    canonical = sorted(rows, key=lambda r: (r["uuid"], CONDITIONS.index(r["persona_condition"]), QUESTIONS.index(r["question"])))
    rng = np.random.default_rng(2026)
    for row in canonical:
        codes = list(v2.QUESTIONS[row["question"]]["choices"])
        probs = np.array([row[f"prob_choice_{code}"] for code in codes], dtype=float)
        row["sampled_choice_seed2026"] = int(rng.choice(codes, p=probs))
    with CHOICE_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for row in canonical: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    export_csv(CHOICE_CSV, canonical)
    return canonical


def population_distributions(rows):
    output = []
    for condition in CONDITIONS:
        for question in QUESTIONS:
            group = [r for r in rows if r["persona_condition"] == condition and r["question"] == question]
            for method in ["greedy", "probability", "sampled_seed2026"]:
                for code, label in v2.QUESTIONS[question]["choices"].items():
                    if method == "greedy": value = sum(r["greedy_choice"] == code for r in group) / len(group)
                    elif method == "probability": value = statistics.fmean(r[f"prob_choice_{code}"] for r in group)
                    else: value = sum(r["sampled_choice_seed2026"] == code for r in group) / len(group)
                    output.append({
                        "persona_condition": condition, "question": question, "method": method,
                        "response_code": code, "response_label": label, "proportion": value, "percent": value * 100,
                    })
    return output


def counterbalanced_distributions(rows):
    output = []
    for condition in CONDITIONS:
        for question in QUESTIONS:
            group = [r for r in rows if r["persona_condition"] == condition and r["question"] == question]
            for method in ["counterbalanced_greedy", "counterbalanced_probability"]:
                for code, label in v2.QUESTIONS[question]["choices"].items():
                    value = (sum(r["greedy_choice"] == code for r in group) / len(group) if method.endswith("greedy")
                             else statistics.fmean(r[f"prob_choice_{code}"] for r in group))
                    output.append({"persona_condition":condition,"question":question,"method":method,
                                   "response_code":code,"response_label":label,"proportion":value,"percent":value*100})
    return output


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return (kl(p, m) + kl(q, m)) / 2


def benchmark_comparison(population, benchmark):
    lookup = {(r["persona_condition"], r["question"], r["method"], int(r["response_code"])): r["percent"] for r in population}
    output = []
    for condition in CONDITIONS:
        for question in QUESTIONS:
            codes = list(v2.QUESTIONS[question]["choices"])
            human = [benchmark["overall"][question][str(c)]["percent"] for c in codes]
            for method in ["greedy", "probability", "sampled_seed2026"]:
                synthetic = [lookup[(condition, question, method, c)] for c in codes]
                differences = [s - h for s, h in zip(synthetic, human)]
                mae = statistics.fmean(abs(x) for x in differences)
                tvd = sum(abs(x) for x in differences) / 200
                jsd = js_divergence(np.array(synthetic) / 100, np.array(human) / 100)
                for c, h, s, diff in zip(codes, human, synthetic, differences):
                    output.append({
                        "persona_condition": condition, "question": question, "method": method,
                        "response_code": c, "response_label": v2.QUESTIONS[question]["choices"][c],
                        "human_percent": h, "synthetic_percent": s, "percentage_point_error": diff,
                        "absolute_percentage_point_error": abs(diff), "mae_percentage_points": mae,
                        "total_variation_distance": tvd, "jensen_shannon_divergence_nats": jsd,
                    })
    return output


def entropy_margin(rows):
    output = []
    for condition in CONDITIONS:
        for question in QUESTIONS:
            group = [r for r in rows if r["persona_condition"] == condition and r["question"] == question]
            entropies, maxima, margins = [], [], []
            for row in group:
                probs = [row[f"prob_choice_{c}"] for c in v2.QUESTIONS[question]["choices"]]
                ordered = sorted(probs, reverse=True)
                entropies.append(-sum(p * math.log(p) for p in probs if p > 0)); maxima.append(ordered[0]); margins.append(ordered[0] - ordered[1])
            record = {
                "persona_condition": condition, "question": question, "n": len(group),
                "mean_max_choice_probability": statistics.fmean(maxima), "median_max_choice_probability": statistics.median(maxima),
                "mean_entropy_nats": statistics.fmean(entropies), "max_entropy_nats": math.log(len(v2.QUESTIONS[question]["choices"])),
                "mean_top1_top2_margin": statistics.fmean(margins),
            }
            for code in range(1, 6):
                record[f"mean_prob_choice_{code}"] = statistics.fmean(r[f"prob_choice_{code}"] for r in group) if code in v2.QUESTIONS[question]["choices"] else None
            output.append(record)
    return output


def persona_sensitivity(rows):
    lookup = {(r["uuid"], r["persona_condition"], r["question"]): r for r in rows}
    individuals, summaries = [], []
    uuids = sorted(set(r["uuid"] for r in rows))
    for question in QUESTIONS:
        question_rows = []
        codes = list(v2.QUESTIONS[question]["choices"])
        for uuid in uuids:
            a, b = lookup[(uuid, "demographics_only", question)], lookup[(uuid, "demographics_plus_persona", question)]
            pa = [a[f"prob_choice_{c}"] for c in codes]; pb = [b[f"prob_choice_{c}"] for c in codes]
            l1 = sum(abs(x-y) for x, y in zip(pa, pb)); ha = -sum(p*math.log(p) for p in pa if p); hb = -sum(p*math.log(p) for p in pb if p)
            record = {"record_type": "individual", "uuid": uuid, "question": question, "l1_distance": l1, "tvd": l1/2,
                      "argmax_changed": a["greedy_choice"] != b["greedy_choice"], "entropy_demographics": ha,
                      "entropy_persona": hb, "entropy_change_persona_minus_demographics": hb-ha}
            individuals.append(record); question_rows.append(record)
        summaries.append({
            "record_type": "summary", "uuid": "ALL", "question": question,
            "mean_l1_distance": statistics.fmean(r["l1_distance"] for r in question_rows),
            "mean_tvd": statistics.fmean(r["tvd"] for r in question_rows),
            "argmax_change_rate": statistics.fmean(r["argmax_changed"] for r in question_rows),
            "mean_entropy_change_persona_minus_demographics": statistics.fmean(r["entropy_change_persona_minus_demographics"] for r in question_rows),
        })
    fields = ["record_type", "uuid", "question", "l1_distance", "tvd", "argmax_changed", "entropy_demographics", "entropy_persona",
              "entropy_change_persona_minus_demographics", "mean_l1_distance", "mean_tvd", "argmax_change_rate", "mean_entropy_change_persona_minus_demographics"]
    return individuals + summaries, fields


def balanced_assignments(sample):
    assignments = {}
    uuids = [str(row["uuid"]) for row in sample]
    for condition in CONDITIONS:
        for question in QUESTIONS:
            count = len(v2.QUESTIONS[question]["choices"])
            ordered = sorted(uuids, key=lambda uuid: hashlib.sha256(f"2026|{condition}|{question}|{uuid}".encode("utf-8")).digest())
            for index, uuid in enumerate(ordered): assignments[(uuid,condition,question)] = index % count
    return assignments


def cyclic_order(question, rotation):
    codes = list(v2.QUESTIONS[question]["choices"])
    return codes[rotation:] + codes[:rotation]


def order_effect(main_rows, reverse_rows, balanced_rows):
    main = {(r["uuid"], r["persona_condition"], r["question"]): r for r in main_rows}
    individuals, summaries = [], []
    for alternative in reverse_rows + balanced_rows:
        original = main[(alternative["uuid"], alternative["persona_condition"], alternative["question"])]
        codes = list(v2.QUESTIONS[alternative["question"]]["choices"])
        po = [original[f"prob_choice_{c}"] for c in codes]; pr = [alternative[f"prob_choice_{c}"] for c in codes]
        l1 = sum(abs(x-y) for x,y in zip(po,pr))
        record = {"record_type": "individual", "uuid": alternative["uuid"], "persona_condition": alternative["persona_condition"],
                  "question": alternative["question"], "comparison_order": alternative["choice_order"],
                  "display_order": alternative["display_order"], "l1_distance": l1, "tvd": l1/2,
                  "argmax_changed": original["greedy_choice"] != alternative["greedy_choice"],
                  "original_greedy_choice": original["greedy_choice"], "alternative_greedy_choice": alternative["greedy_choice"]}
        for c in range(1,6):
            record[f"original_prob_choice_{c}"] = original.get(f"prob_choice_{c}")
            record[f"alternative_prob_choice_{c}"] = alternative.get(f"prob_choice_{c}")
        individuals.append(record)
    for condition in CONDITIONS:
        for question in QUESTIONS:
            for comparison_order in ["reversed", "balanced_cyclic"]:
                group = [r for r in individuals if r["persona_condition"] == condition and r["question"] == question and r["comparison_order"] == comparison_order]
                summaries.append({"record_type": "summary", "uuid": "ALL", "persona_condition": condition, "question": question,
                                  "comparison_order": comparison_order,
                                  "mean_l1_distance": statistics.fmean(r["l1_distance"] for r in group),
                                  "mean_tvd": statistics.fmean(r["tvd"] for r in group),
                                  "argmax_change_rate": statistics.fmean(r["argmax_changed"] for r in group)})
    fields = ["record_type", "uuid", "persona_condition", "question", "comparison_order", "display_order", "l1_distance", "tvd", "argmax_changed",
              "original_greedy_choice", "alternative_greedy_choice"] + [f"original_prob_choice_{c}" for c in range(1,6)] + [f"alternative_prob_choice_{c}" for c in range(1,6)] + ["mean_l1_distance", "mean_tvd", "argmax_change_rate"]
    return individuals + summaries, fields


def position_bias(balanced_rows):
    output = []
    for condition in CONDITIONS:
        for question in QUESTIONS:
            codes = list(v2.QUESTIONS[question]["choices"]); final_position = len(codes)
            group = [r for r in balanced_rows if r["persona_condition"] == condition and r["question"] == question]
            for code in codes:
                by_position = {}
                for position in range(1, final_position + 1):
                    subset = [r for r in group if int(r[f"display_position_choice_{code}"]) == position]
                    by_position[position] = statistics.fmean(r[f"prob_choice_{code}"] for r in subset)
                    output.append({"record_type":"semantic_by_position","persona_condition":condition,"question":question,
                                   "semantic_choice":code,"is_unknown":code==final_position,"display_position":position,"n":len(subset),
                                   "mean_probability":by_position[position]})
                other_first = [value for pos,value in by_position.items() if pos != 1]
                other_final = [value for pos,value in by_position.items() if pos != final_position]
                output.append({"record_type":"position_effect_summary","persona_condition":condition,"question":question,
                               "semantic_choice":code,"is_unknown":code==final_position,"display_position":"ALL","n":len(group),
                               "mean_probability":statistics.fmean(by_position.values()),
                               "position_1_effect":by_position[1]-statistics.fmean(other_first),
                               "final_position_effect":by_position[final_position]-statistics.fmean(other_final)})
    return output


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cross_model_comparison(gemma_benchmark, gemma_entropy, gemma_order, gemma_sensitivity):
    qwen_benchmark = read_csv(OUT / "smoke_v3_benchmark_comparison.csv")
    qwen_entropy = read_csv(OUT / "smoke_v3_entropy_margin.csv")
    qwen_order = read_csv(OUT / "smoke_v3_order_effect.csv")
    qwen_sensitivity = read_csv(OUT / "smoke_v3_persona_sensitivity.csv")
    exaone_benchmark = read_csv(OUT / "smoke_v4_exaone_benchmark_comparison.csv")
    exaone_entropy = read_csv(OUT / "smoke_v4_exaone_entropy_margin.csv")
    exaone_order = read_csv(OUT / "smoke_v4_exaone_order_effect.csv")
    exaone_sensitivity = read_csv(OUT / "smoke_v4_exaone_persona_sensitivity.csv")
    sources = {
        "Qwen3-8B-Q4_K_M": (qwen_benchmark, qwen_entropy, qwen_order, qwen_sensitivity),
        "EXAONE-3.5-7.8B-Q4_K_M": (exaone_benchmark, exaone_entropy, exaone_order, exaone_sensitivity),
        "Gemma-3-12B-QAT-Q4_0": (gemma_benchmark, gemma_entropy, gemma_order, gemma_sensitivity),
    }
    output = []
    for model, (benchmark_rows, entropy_rows, order_rows, sensitivity_rows) in sources.items():
        for condition in CONDITIONS:
            for question in QUESTIONS:
                b = next(r for r in benchmark_rows if r["persona_condition"] == condition and r["question"] == question and r["method"] == "probability")
                e = next(r for r in entropy_rows if r["persona_condition"] == condition and r["question"] == question)
                unknown_code = "4" if question == "Q4" else "5"
                unknown = e[f"mean_prob_choice_{unknown_code}"]
                s = next(r for r in sensitivity_rows if r["record_type"] == "summary" and r["question"] == question)
                order_tvd = ""
                candidates = [r for r in order_rows if r["record_type"] == "summary" and r["persona_condition"] == condition and r["question"] == question]
                if model.startswith("Gemma"):
                    candidates = [r for r in candidates if r.get("comparison_order") == "reversed"]
                if candidates:
                    o = candidates[0]
                    order_tvd = o["mean_tvd"]
                output.append({
                    "model": model, "question": question, "persona_condition": condition,
                    "human_mae_percentage_points": b["mae_percentage_points"],
                    "human_tvd": b["total_variation_distance"],
                    "human_js_divergence_nats": b["jensen_shannon_divergence_nats"],
                    "mean_entropy_nats": e["mean_entropy_nats"],
                    "mean_max_probability": e["mean_max_choice_probability"],
                    "mean_top1_top2_margin": e["mean_top1_top2_margin"],
                    "unknown_probability": unknown, "persona_effect_tvd": s["mean_tvd"], "order_effect_tvd": order_tvd,
                })
    return output


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(population, benchmark_rows, entropy_rows, sensitivity_rows, order_rows, position_rows, cross_rows, gpu, total_seconds):
    pop = {(r["persona_condition"],r["question"],r["method"],int(r["response_code"])):r["percent"] for r in population}
    bench = {}; 
    for r in benchmark_rows: bench[(r["persona_condition"],r["question"],r["method"])]=(r["mae_percentage_points"],r["total_variation_distance"],r["jensen_shannon_divergence_nats"])
    sens = {r["question"]:r for r in sensitivity_rows if r["record_type"]=="summary"}
    order = {(r["persona_condition"],r["question"],r["comparison_order"]):r for r in order_rows if r["record_type"]=="summary"}
    lines = ["# Smoke v5: Gemma and balanced order", "", "Qwen/EXAONE과 동일한 표본·문항·semantic prompt를 사용한 Gemma 3 교차모델 진단이다. Human benchmark는 original order 평가에만 사용했고 prompt에는 제공하지 않았다.", "",
             "## Extraction", "", "- b10514 `/completion`의 direct next-token logprobs 추출: 성공", "- Raw logits API: 미제공",
             "- `relative_logit_choice_*`: direct logprob에서 valid-choice 최대 logprob를 뺀 정확한 상대 logit", f"- A–E fallback token IDs (semantic 1–5): {TOKEN_IDS}", f"- Total wall time: {total_seconds:.3f}초 ({total_seconds/60:.2f}분)",
             f"- GPU: {gpu.name}, peak {gpu.peak:,} MiB, baseline 대비 +{gpu.peak-gpu.baseline:,} MiB", "", "## Probability-aggregated distributions (%)", ""]
    dist_rows=[]
    for c in CONDITIONS:
        for q in QUESTIONS:
            original=[f"{pop[(c,q,'probability',code)]:.2f}" for code in v2.QUESTIONS[q]["choices"]]
            balanced=[f"{pop[(c,q,'counterbalanced_probability',code)]:.2f}" for code in v2.QUESTIONS[q]["choices"]]
            dist_rows.append([c,q," / ".join(original)," / ".join(balanced)])
    lines += [md_table(["condition","question","original choice 1..K (%)","counterbalanced choice 1..K (%)"],dist_rows),"", "## Original-order benchmark summary", ""]
    b_rows=[]
    for c in CONDITIONS:
        for q in QUESTIONS:
            for m in ["greedy","probability","sampled_seed2026"]:
                x=bench[(c,q,m)]; b_rows.append([c,q,m,f"{x[0]:.3f}",f"{x[1]:.4f}",f"{x[2]:.4f}"])
    lines += [md_table(["condition","question","method","MAE(pp)","TVD","JSD(nats)"],b_rows),"", "## Concentration diagnostics", ""]
    e_rows=[]
    for r in entropy_rows: e_rows.append([r["persona_condition"],r["question"],f"{r['mean_max_choice_probability']:.4f}",f"{r['median_max_choice_probability']:.4f}",f"{r['mean_entropy_nats']:.4f}",f"{r['mean_top1_top2_margin']:.4f}"])
    lines += [md_table(["condition","question","mean max P","median max P","mean entropy","mean margin"],e_rows),"", "## Persona sensitivity", ""]
    lines += [md_table(["question","mean L1","mean TVD","argmax change","entropy change"],[[q,f"{sens[q]['mean_l1_distance']:.4f}",f"{sens[q]['mean_tvd']:.4f}",f"{sens[q]['argmax_change_rate']:.1%}",f"{sens[q]['mean_entropy_change_persona_minus_demographics']:+.4f}"] for q in QUESTIONS]),"", "## Order effect (all 100 people)", ""]
    lines += [md_table(["condition","question","comparison","mean TVD","argmax change"],[[c,q,o,f"{order[(c,q,o)]['mean_tvd']:.4f}",f"{order[(c,q,o)]['argmax_change_rate']:.1%}"] for c in CONDITIONS for q in QUESTIONS for o in ["reversed","balanced_cyclic"]]),"",
              "## Unknown-option position effect", ""]
    unknown_rows=[r for r in position_rows if r["record_type"]=="semantic_by_position" and r["is_unknown"]]
    lines += [md_table(["condition","question","display position","n","mean unknown P"],[[r["persona_condition"],r["question"],r["display_position"],r["n"],f"{r['mean_probability']:.4f}"] for r in unknown_rows]),"",
              "## Three-model comparison (original-order probability aggregation)", "",
              md_table(["model","condition","question","MAE","TVD","JSD","entropy","max P","margin","unknown P","persona TVD","order TVD"], [[r["model"],r["persona_condition"],r["question"],f"{float(r['human_mae_percentage_points']):.3f}",f"{float(r['human_tvd']):.4f}",f"{float(r['human_js_divergence_nats']):.4f}",f"{float(r['mean_entropy_nats']):.4f}",f"{float(r['mean_max_probability']):.4f}",f"{float(r['mean_top1_top2_margin']):.4f}",f"{float(r['unknown_probability']):.4f}",f"{float(r['persona_effect_tvd']):.4f}",f"{float(r['order_effect_tvd']):.4f}" if r["order_effect_tvd"] != "" else "NA"] for r in cross_rows]), "",
              "## Interpretation limits","", "- 이 결과는 100명 기술 진단이며 한국 국민 여론이 아니다.", "- Benchmark는 최적화에 사용하지 않았다.", "- 1,000명 pilot은 별도 승인 전까지 실행하지 않는다.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--probe-only",action="store_true"); args=parser.parse_args()
    OUT.mkdir(exist_ok=True); init_or_verify_preservation()
    if sha256_file(MODEL)!=MODEL_SHA: raise RuntimeError("model hash mismatch")
    CONFIG.write_text(json.dumps(CONFIG_DATA,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    sample=pq.read_table(SAMPLE).to_pylist()
    if len(sample)!=100: raise RuntimeError("sample must be 100")
    gpu=GpuMonitor(); gpu.start(); process=None; server_props={}; start=time.perf_counter()
    with SERVER_LOG.open("w",encoding="utf-8") as log:
        try:
            process=start_server(log); wait_server(process); server_props=get("/props"); validate_tokens()
            if args.probe_only:
                probe = extract_choice(sample[0],CONDITIONS[0],"Q2")
                probe["probe_gpu_name"] = gpu.name; probe["probe_vram_total_mib"] = gpu.total
                probe["probe_vram_baseline_mib"] = gpu.baseline; probe["probe_vram_peak_mib"] = gpu.peak
                print(json.dumps(probe,ensure_ascii=True,indent=2)); return
            existing=read_jsonl(CHECKPOINT); keys={(r["uuid"],r["persona_condition"],r["question"]) for r in existing}
            if len(keys)!=len(existing): raise RuntimeError("duplicate main checkpoint")
            done=len(existing)
            for row in sample:
                for condition in CONDITIONS:
                    for question in QUESTIONS:
                        key=(str(row["uuid"]),condition,question)
                        if key in keys: continue
                        result=extract_choice(row,condition,question,"original"); append_jsonl(CHECKPOINT,result); keys.add(key); done+=1
                        if done%40==0: print(f"main={done}/800",flush=True)
            main_rows=read_jsonl(CHECKPOINT); main_lookup={(r["uuid"],r["persona_condition"],r["question"]):r for r in main_rows}
            row_by_uuid={str(r["uuid"]):r for r in sample}
            reverse_existing=read_jsonl(REVERSE_CHECKPOINT); reverse_keys={(r["uuid"],r["persona_condition"],r["question"]) for r in reverse_existing}
            reverse_done=len(reverse_existing)
            for uuid in sorted(row_by_uuid):
                for condition in CONDITIONS:
                    for question in QUESTIONS:
                        key=(uuid,condition,question)
                        if key in reverse_keys: continue
                        codes=list(reversed(list(v2.QUESTIONS[question]["choices"])))
                        result=extract_choice(row_by_uuid[uuid],condition,question,"reversed",codes); append_jsonl(REVERSE_CHECKPOINT,result); reverse_keys.add(key); reverse_done+=1
                        if reverse_done%40==0: print(f"reverse={reverse_done}/800",flush=True)
            reverse_rows=read_jsonl(REVERSE_CHECKPOINT)
            assignments=balanced_assignments(sample); balanced_existing=read_jsonl(BALANCED_CHECKPOINT); balanced_keys={(r["uuid"],r["persona_condition"],r["question"]) for r in balanced_existing}
            balanced_done=len(balanced_existing)
            for uuid in sorted(row_by_uuid):
                for condition in CONDITIONS:
                    for question in QUESTIONS:
                        key=(uuid,condition,question)
                        if key in balanced_keys: continue
                        rotation=assignments[key]; codes=cyclic_order(question,rotation)
                        if rotation==0:
                            result=dict(main_lookup[key]); result["choice_order"]="balanced_cyclic"; result["display_order"]=",".join(map(str,codes)); result["reused_primary"]=True
                        else:
                            result=extract_choice(row_by_uuid[uuid],condition,question,"balanced_cyclic",codes); result["reused_primary"]=False
                        result["cyclic_rotation"]=rotation; append_jsonl(BALANCED_CHECKPOINT,result); balanced_keys.add(key); balanced_done+=1
                        if balanced_done%40==0: print(f"balanced={balanced_done}/800",flush=True)
            balanced_rows=read_jsonl(BALANCED_CHECKPOINT); total_seconds=time.perf_counter()-start
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill()
            gpu.stop()
    rows=finalize_choice_rows(main_rows); primary_population=population_distributions(rows); population=primary_population+counterbalanced_distributions(balanced_rows); benchmark=json.loads(BENCHMARK.read_text(encoding="utf-8"))
    benchmark_rows=benchmark_comparison(primary_population,benchmark); entropy_rows=entropy_margin(rows); sensitivity_rows,sens_fields=persona_sensitivity(rows); order_rows,order_fields=order_effect(rows,reverse_rows,balanced_rows); position_rows=position_bias(balanced_rows)
    export_csv(POPULATION_CSV,population); export_csv(BENCHMARK_CSV,benchmark_rows); export_csv(ENTROPY_CSV,entropy_rows); export_csv(SENSITIVITY_CSV,sensitivity_rows,sens_fields); export_csv(ORDER_CSV,order_rows,order_fields); export_csv(POSITION_CSV,position_rows,["record_type","persona_condition","question","semantic_choice","is_unknown","display_position","n","mean_probability","position_1_effect","final_position_effect"])
    cross_rows = cross_model_comparison(benchmark_rows, entropy_rows, order_rows, sensitivity_rows); export_csv(CROSS_MODEL_CSV, cross_rows)
    write_report(population,benchmark_rows,entropy_rows,sensitivity_rows,order_rows,position_rows,cross_rows,gpu,total_seconds)
    embedded_template = server_props.get("chat_template", "See server /props; unavailable in response")
    ENVIRONMENT.write_text(f"# Smoke v5 Gemma environment\n\n- Primary repository: `google/gemma-3-12b-it-qat-q4_0-gguf`\n- Exact filename: `{MODEL.name}`\n- Fallback used: no\n- File size: {MODEL.stat().st_size:,} bytes\n- SHA-256: `{MODEL_SHA}`\n- Quantization: official QAT Q4_0\n- Model architecture: Gemma 3 (`gemma3`)\n- Parameter count: 11.77B (GGUF metadata; marketed size 12B)\n- Transformer blocks: 48\n- Chat template: official template embedded in GGUF and applied by `/apply-template`\n- GPU: {gpu.name}\n- VRAM total: {gpu.total:,} MiB\n- VRAM baseline: {gpu.baseline:,} MiB\n- VRAM peak: {gpu.peak:,} MiB\n- Peak increment: {gpu.peak-gpu.baseline:,} MiB\n- Peak utilization: {gpu.util}%\n- GPU offload: 49/49 layers (47 repeating + output; verbose metadata probe)\n- CUDA model buffer: 7,694.23 MiB\n- Context length: 4096\n- llama.cpp: b10514 commit 7221e24f5\n\n## Embedded chat template reported by llama.cpp\n\n```jinja\n{embedded_template}\n```\n",encoding="utf-8")
    init_or_verify_preservation(); print("SMOKE_V5_COMPLETE")


if __name__=="__main__": main()
