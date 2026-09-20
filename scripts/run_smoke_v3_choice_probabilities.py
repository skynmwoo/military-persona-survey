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
MODEL = ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"
RUNTIME = ROOT / "tools" / "llama.cpp" / "b10514"
SERVER = RUNTIME / "llama-server.exe"
BENCHMARK = OUT / "human_benchmark_20260818.json"
PORT = 8090

CHECKPOINT = OUT / "smoke_v3_choice_probabilities_checkpoint.jsonl"
ORDER_CHECKPOINT = OUT / "smoke_v3_order_effect_checkpoint.jsonl"
CHOICE_JSONL = OUT / "smoke_v3_choice_probabilities.jsonl"
CHOICE_CSV = OUT / "smoke_v3_choice_probabilities.csv"
POPULATION_CSV = OUT / "smoke_v3_population_distribution.csv"
BENCHMARK_CSV = OUT / "smoke_v3_benchmark_comparison.csv"
ENTROPY_CSV = OUT / "smoke_v3_entropy_margin.csv"
SENSITIVITY_CSV = OUT / "smoke_v3_persona_sensitivity.csv"
ORDER_CSV = OUT / "smoke_v3_order_effect.csv"
REPORT = OUT / "smoke_v3_report.md"
TOKEN_REPORT = OUT / "smoke_v3_token_validation.md"
CONFIG = OUT / "smoke_v3_inference_config.json"
ENVIRONMENT = OUT / "smoke_v3_model_environment.md"
SERVER_LOG = OUT / "smoke_v3_llama_server.log"
PRESERVATION = OUT / "smoke_v3_preservation_manifest.json"

CONDITIONS = ["demographics_only", "demographics_plus_persona"]
QUESTIONS = ["Q2", "Q3", "Q4", "Q5"]
TOKEN_IDS = {1: 16, 2: 17, 3: 18, 4: 19, 5: 20}
MODEL_SHA = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"

SYSTEM_PROMPT = """제공된 합성 인물의 관점에서 설문에 응답한다.
제공되지 않은 새로운 생애사, 사건, 경험 등의 사실을 만들어내지 않는다.
그러나 프로필에 해당 정책에 대한 의견이 명시되어 있지 않다는 이유만으로 자동으로 '잘 모르겠다'를 선택하지 않는다.
실제 사람이 설문에 응답하듯 제공된 특성을 바탕으로 가장 자연스러운 정책 선호를 판단한다.
'잘 모르겠다'는 정상적인 선택지이며 실제로 판단이 불확실한 경우 선택할 수 있다.
특정 응답으로 유도하지 않는다.
이유나 chain-of-thought를 생성하지 않는다.
현재 문항의 선택지 코드 하나만 JSON 값으로 선택한다."""

CONFIG_DATA = {
    "experiment": "smoke_v3_choice_probabilities",
    "sample": "data/derived/smoke_100_seed2026.parquet",
    "sample_size": 100,
    "persona_conditions": CONDITIONS,
    "questions": QUESTIONS,
    "question_context": "Each persona_condition × question is an independent fresh conversation; Q1 is excluded.",
    "model": "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
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
    "order_effect": {"sample_size": 20, "seed": 2026, "questions": ["Q2", "Q5"], "orders": ["original", "reversed"]},
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
    paths += [
        OUT / "smoke_test_report.md", OUT / "model_environment.md", OUT / "inference_config.json",
        OUT / "llama_server.log", BENCHMARK,
        ROOT / "scripts" / "run_smoke_test.py", ROOT / "scripts" / "create_smoke_sample.py",
        ROOT / "scripts" / "run_smoke_v2.py",
    ]
    return list(dict.fromkeys(p for p in paths if p.exists() and not p.name.startswith("smoke_v3")))


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


def question_text(question, reverse=False):
    spec = v2.QUESTIONS[question]
    items = list(spec["choices"].items())
    if reverse: items.reverse()
    choices = "\n".join(f"{code} = {label}" for code, label in items)
    return f"{spec['text']}\n\n{choices}\n\n선택지 코드 하나를 JSON 값으로 선택하십시오. 이유는 쓰지 마십시오."


def validate_tokens():
    observed = {}
    for label in ["1", "2", "3", "4", "5", "A", "B", "C", "D", "E"]:
        observed[label] = post("/tokenize", {"content": label, "add_special": False, "with_pieces": True})["tokens"]
    for code, token_id in TOKEN_IDS.items():
        tokens = observed[str(code)]
        if len(tokens) != 1 or tokens[0]["id"] != token_id:
            raise RuntimeError(f"Unexpected tokenization for {code}: {tokens}")
    lines = [
        "# Smoke v3 choice token validation", "",
        "- llama.cpp: b10514, commit `7221e24f5`", "- Model: `Qwen3-8B-Q4_K_M.gguf`",
        "- `/tokenize` was called with `add_special=false`.",
        "- Assistant prefix ends in one ASCII space; the next generated token is the bare digit token.",
        "- Numeric labels 1–5 are all one token and directly comparable, so A–E fallback was not used.", "",
        "| Label | Token ID | Piece | Single token |", "| --- | ---: | --- | --- |",
    ]
    for code in range(1, 6):
        token = observed[str(code)][0]
        lines.append(f"| {code} | {token['id']} | `{token['piece']}` | yes |")
    lines += [
        "", "## API extraction", "",
        "- `/apply-template` applies the embedded Qwen chat template with reasoning disabled.",
        "- `/completion` generates exactly one token with a digit-only grammar and returns `completion_probabilities.top_logprobs`.",
        "- Raw logits are not exposed by b10514 server. Direct server logprobs are stored; relative logits are exactly `choice logprob - max(valid choice logprob)`.",
        "- Choice probabilities are a fresh softmax over valid-choice relative logits only.", "",
    ]
    TOKEN_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return observed


def extract_choice(row, condition, question, reverse=False, n_probs=64):
    messages = [
        {"role": "system", "content": profile_system(row, condition)},
        {"role": "user", "content": question_text(question, reverse)},
    ]
    start = time.perf_counter()
    rendered = post("/apply-template", {"messages": messages})["prompt"]
    prefix = rendered + '{"' + question.lower() + '": '
    valid_codes = list(v2.QUESTIONS[question]["choices"])
    grammar = f"root ::= [{min(valid_codes)}-{max(valid_codes)}]"
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
        return extract_choice(row, condition, question, reverse, n_probs=512)
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
        "uuid": str(row["uuid"]), "persona_condition": condition, "question": question,
        "choice_order": "reversed" if reverse else "original", "greedy_choice": valid_codes[greedy_index],
        "api_generated_token_id": completion["tokens"][0], "api_generated_content": completion["content"],
        "inference_time_seconds": round(elapsed, 6), "n_probs_used": n_probs,
    }
    for code in range(1, 6):
        if code in valid_codes:
            idx = valid_codes.index(code)
            result[f"token_id_choice_{code}"] = TOKEN_IDS[code]
            result[f"logit_choice_{code}"] = relative[idx]
            result[f"logprob_choice_{code}"] = logprobs[idx]
            result[f"prob_choice_{code}"] = probabilities[idx]
        else:
            result[f"token_id_choice_{code}"] = None
            result[f"logit_choice_{code}"] = None
            result[f"logprob_choice_{code}"] = None
            result[f"prob_choice_{code}"] = None
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


def select_order_uuids(rows):
    uuids = sorted(str(r["uuid"]) for r in rows)
    return list(np.random.default_rng(2026).choice(uuids, size=20, replace=False))


def order_effect(main_rows, reverse_rows):
    main = {(r["uuid"], r["persona_condition"], r["question"]): r for r in main_rows}
    individuals, summaries = [], []
    for reverse in reverse_rows:
        original = main[(reverse["uuid"], reverse["persona_condition"], reverse["question"])]
        codes = list(v2.QUESTIONS[reverse["question"]]["choices"])
        po = [original[f"prob_choice_{c}"] for c in codes]; pr = [reverse[f"prob_choice_{c}"] for c in codes]
        l1 = sum(abs(x-y) for x,y in zip(po,pr))
        record = {"record_type": "individual", "uuid": reverse["uuid"], "persona_condition": reverse["persona_condition"],
                  "question": reverse["question"], "l1_distance": l1, "tvd": l1/2,
                  "argmax_changed": original["greedy_choice"] != reverse["greedy_choice"],
                  "original_greedy_choice": original["greedy_choice"], "reversed_greedy_choice": reverse["greedy_choice"]}
        for c in range(1,6):
            record[f"original_prob_choice_{c}"] = original.get(f"prob_choice_{c}")
            record[f"reversed_prob_choice_{c}"] = reverse.get(f"prob_choice_{c}")
        individuals.append(record)
    for condition in CONDITIONS:
        for question in ["Q2", "Q5"]:
            group = [r for r in individuals if r["persona_condition"] == condition and r["question"] == question]
            summaries.append({"record_type": "summary", "uuid": "ALL", "persona_condition": condition, "question": question,
                              "mean_l1_distance": statistics.fmean(r["l1_distance"] for r in group),
                              "mean_tvd": statistics.fmean(r["tvd"] for r in group),
                              "argmax_change_rate": statistics.fmean(r["argmax_changed"] for r in group)})
    fields = ["record_type", "uuid", "persona_condition", "question", "l1_distance", "tvd", "argmax_changed",
              "original_greedy_choice", "reversed_greedy_choice"] + [f"original_prob_choice_{c}" for c in range(1,6)] + [f"reversed_prob_choice_{c}" for c in range(1,6)] + ["mean_l1_distance", "mean_tvd", "argmax_change_rate"]
    return individuals + summaries, fields


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(population, benchmark_rows, entropy_rows, sensitivity_rows, order_rows, gpu, total_seconds):
    pop = {(r["persona_condition"],r["question"],r["method"],int(r["response_code"])):r["percent"] for r in population}
    bench = {}; 
    for r in benchmark_rows: bench[(r["persona_condition"],r["question"],r["method"])]=(r["mae_percentage_points"],r["total_variation_distance"],r["jensen_shannon_divergence_nats"])
    sens = {r["question"]:r for r in sensitivity_rows if r["record_type"]=="summary"}
    order = {(r["persona_condition"],r["question"]):r for r in order_rows if r["record_type"]=="summary"}
    lines = ["# Smoke v3 choice probabilities", "", "Q1을 제외한 독립 문항 진단이다. Human benchmark는 평가에만 사용했고 prompt에는 제공하지 않았다.", "",
             "## Extraction", "", "- b10514 `/completion`의 direct next-token logprobs 추출: 성공", "- Raw logits API: 미제공",
             "- `logit_choice_*`: direct logprob에서 valid-choice 최대 logprob를 뺀 정확한 상대 logit", "- 숫자 label token: 1–5 = token IDs 16–20", f"- Total wall time: {total_seconds:.3f}초 ({total_seconds/60:.2f}분)",
             f"- GPU: {gpu.name}, peak {gpu.peak:,} MiB, baseline 대비 +{gpu.peak-gpu.baseline:,} MiB", "", "## Probability-aggregated distributions (%)", ""]
    dist_rows=[]
    for c in CONDITIONS:
        for q in QUESTIONS:
            vals=[f"{pop[(c,q,'probability',code)]:.2f}" for code in v2.QUESTIONS[q]["choices"]]
            dist_rows.append([c,q," / ".join(vals)])
    lines += [md_table(["condition","question","choice 1..K (%)"],dist_rows),"", "## Benchmark summary", ""]
    b_rows=[]
    for c in CONDITIONS:
        for q in QUESTIONS:
            for m in ["greedy","probability","sampled_seed2026"]:
                x=bench[(c,q,m)]; b_rows.append([c,q,m,f"{x[0]:.3f}",f"{x[1]:.4f}",f"{x[2]:.4f}"])
    lines += [md_table(["condition","question","method","MAE(pp)","TVD","JSD(nats)"],b_rows),"", "## Concentration diagnostics", ""]
    e_rows=[]
    for r in entropy_rows: e_rows.append([r["persona_condition"],r["question"],f"{r['mean_max_choice_probability']:.4f}",f"{r['median_max_choice_probability']:.4f}",f"{r['mean_entropy_nats']:.4f}",f"{r['mean_top1_top2_margin']:.4f}"])
    lines += [md_table(["condition","question","mean max P","median max P","mean entropy","mean margin"],e_rows),"", "## Persona sensitivity", ""]
    lines += [md_table(["question","mean L1","mean TVD","argmax change","entropy change"],[[q,f"{sens[q]['mean_l1_distance']:.4f}",f"{sens[q]['mean_tvd']:.4f}",f"{sens[q]['argmax_change_rate']:.1%}",f"{sens[q]['mean_entropy_change_persona_minus_demographics']:+.4f}"] for q in QUESTIONS]),"", "## Order effect (20 people)", ""]
    lines += [md_table(["condition","question","mean L1","mean TVD","argmax change"],[[c,q,f"{order[(c,q)]['mean_l1_distance']:.4f}",f"{order[(c,q)]['mean_tvd']:.4f}",f"{order[(c,q)]['argmax_change_rate']:.1%}"] for c in CONDITIONS for q in ["Q2","Q5"]]),"",
              "## Interpretation limits","", "- 이 결과는 100명 기술 진단이며 한국 국민 여론이 아니다.", "- Benchmark는 최적화에 사용하지 않았다.", "- 1,000명 pilot은 별도 승인 전까지 실행하지 않는다.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--probe-only",action="store_true"); args=parser.parse_args()
    OUT.mkdir(exist_ok=True); init_or_verify_preservation()
    if sha256_file(MODEL)!=MODEL_SHA: raise RuntimeError("model hash mismatch")
    CONFIG.write_text(json.dumps(CONFIG_DATA,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    sample=pq.read_table(SAMPLE).to_pylist()
    if len(sample)!=100: raise RuntimeError("sample must be 100")
    gpu=GpuMonitor(); gpu.start(); process=None; start=time.perf_counter()
    with SERVER_LOG.open("w",encoding="utf-8") as log:
        try:
            process=start_server(log); wait_server(process); validate_tokens()
            if args.probe_only:
                print(json.dumps(extract_choice(sample[0],CONDITIONS[0],"Q2"),ensure_ascii=True,indent=2)); return
            existing=read_jsonl(CHECKPOINT); keys={(r["uuid"],r["persona_condition"],r["question"]) for r in existing}
            if len(keys)!=len(existing): raise RuntimeError("duplicate main checkpoint")
            done=len(existing)
            for row in sample:
                for condition in CONDITIONS:
                    for question in QUESTIONS:
                        key=(str(row["uuid"]),condition,question)
                        if key in keys: continue
                        result=extract_choice(row,condition,question); append_jsonl(CHECKPOINT,result); keys.add(key); done+=1
                        if done%40==0: print(f"main={done}/800",flush=True)
            main_rows=read_jsonl(CHECKPOINT)
            selected=set(select_order_uuids(sample)); order_existing=read_jsonl(ORDER_CHECKPOINT)
            order_keys={(r["uuid"],r["persona_condition"],r["question"]) for r in order_existing}
            order_done=len(order_existing)
            row_by_uuid={str(r["uuid"]):r for r in sample}
            for uuid in sorted(selected):
                for condition in CONDITIONS:
                    for question in ["Q2","Q5"]:
                        key=(uuid,condition,question)
                        if key in order_keys: continue
                        result=extract_choice(row_by_uuid[uuid],condition,question,reverse=True); append_jsonl(ORDER_CHECKPOINT,result); order_keys.add(key); order_done+=1
                        if order_done%20==0: print(f"order={order_done}/80",flush=True)
            reverse_rows=read_jsonl(ORDER_CHECKPOINT); total_seconds=time.perf_counter()-start
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill()
            gpu.stop()
    rows=finalize_choice_rows(main_rows); population=population_distributions(rows); benchmark=json.loads(BENCHMARK.read_text(encoding="utf-8"))
    benchmark_rows=benchmark_comparison(population,benchmark); entropy_rows=entropy_margin(rows); sensitivity_rows,sens_fields=persona_sensitivity(rows); order_rows,order_fields=order_effect(rows,reverse_rows)
    export_csv(POPULATION_CSV,population); export_csv(BENCHMARK_CSV,benchmark_rows); export_csv(ENTROPY_CSV,entropy_rows); export_csv(SENSITIVITY_CSV,sensitivity_rows,sens_fields); export_csv(ORDER_CSV,order_rows,order_fields)
    write_report(population,benchmark_rows,entropy_rows,sensitivity_rows,order_rows,gpu,total_seconds)
    ENVIRONMENT.write_text(f"# Smoke v3 environment\n\n- GPU: {gpu.name}\n- VRAM total: {gpu.total:,} MiB\n- VRAM baseline: {gpu.baseline:,} MiB\n- VRAM peak: {gpu.peak:,} MiB\n- Peak increment: {gpu.peak-gpu.baseline:,} MiB\n- Peak utilization: {gpu.util}%\n- llama.cpp: b10514 commit 7221e24f5\n- Model SHA-256: `{MODEL_SHA}`\n",encoding="utf-8")
    init_or_verify_preservation(); print("SMOKE_V3_COMPLETE")


if __name__=="__main__": main()
