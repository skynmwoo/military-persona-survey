from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_smoke_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "derived" / "validation_1000_locked_seed42.parquet"
LOCKED = ROOT / "outputs" / "validation_1000_locked"
RAW = LOCKED / "raw"
PROBS = LOCKED / "probabilities"
RUNTIME = ROOT / "tools" / "llama.cpp" / "b10514"
SERVER = RUNTIME / "llama-server.exe"
CONDITIONS = ["demographics_only", "demographics_plus_persona"]
QUESTIONS = ["Q2", "Q3", "Q4", "Q5"]
SYSTEM_PROMPT = """제공된 합성 인물의 관점에서 설문에 응답한다.
제공되지 않은 새로운 생애사, 사건, 경험 등의 사실을 만들어내지 않는다.
그러나 프로필에 해당 정책에 대한 의견이 명시되어 있지 않다는 이유만으로 자동으로 '잘 모르겠다'를 선택하지 않는다.
실제 사람이 설문에 응답하듯 제공된 특성을 바탕으로 가장 자연스러운 정책 선호를 판단한다.
'잘 모르겠다'는 정상적인 선택지이며 실제로 판단이 불확실한 경우 선택할 수 있다.
특정 응답으로 유도하지 않는다.
이유나 chain-of-thought를 생성하지 않는다.
현재 문항의 선택지 코드 하나만 JSON 값으로 선택한다."""

MODELS = {
    "qwen": {
        "display": "Qwen3-8B-Q4_K_M", "path": ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf",
        "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", "port": 8101, "parallel": 4,
        "labels": {1:"1",2:"2",3:"3",4:"4",5:"5"}, "prefix_space": True,
    },
    "exaone": {
        "display": "EXAONE-3.5-7.8B-Instruct-Q4_K_M", "path": ROOT / "models" / "LGAI-EXAONE" / "EXAONE-3.5-7.8B-Instruct-GGUF" / "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf",
        "sha256": "2f4aff8d555b82500d4de2723810586e83e358003388fdd08bd52e6091e6ddd1", "port": 8102, "parallel": 4,
        "labels": {1:"1",2:"2",3:"3",4:"4",5:"5"}, "prefix_space": True,
    },
    "gemma": {
        "display": "Gemma-3-12B-QAT-Q4_0", "path": ROOT / "models" / "google" / "gemma-3-12b-it-qat-q4_0-gguf" / "gemma-3-12b-it-q4_0.gguf",
        "sha256": "dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd", "port": 8103, "parallel": 2,
        "labels": {1:"A",2:"B",3:"C",4:"D",5:"E"}, "prefix_space": False,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def run_text(command):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags).stdout.strip()


def nvidia_query(fields):
    return [x.strip() for x in run_text(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).split(",")]


class GpuMonitor:
    def __init__(self):
        name,total,used=nvidia_query("name,memory.total,memory.used")
        self.name,self.total,self.baseline=name,int(total),int(used); self.peak=self.baseline; self.util=0
        self.stop_event=threading.Event(); self.thread=threading.Thread(target=self._loop,daemon=True)
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                used,util=nvidia_query("memory.used,utilization.gpu"); self.peak=max(self.peak,int(used)); self.util=max(self.util,int(util))
            except Exception: pass
            self.stop_event.wait(.25)
    def start(self): self.thread.start()
    def stop(self): self.stop_event.set(); self.thread.join(timeout=3)


class Client:
    def __init__(self, port): self.base=f"http://127.0.0.1:{port}"
    def post(self,path,payload,timeout=180):
        req=urllib.request.Request(self.base+path,data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=timeout) as r: return json.loads(r.read().decode("utf-8"))
    def get(self,path,timeout=5):
        with urllib.request.urlopen(self.base+path,timeout=timeout) as r: return json.loads(r.read().decode("utf-8"))


def start_server(cfg, log_handle):
    cmd=[str(SERVER),"--model",str(cfg["path"]),"--ctx-size","4096","--n-gpu-layers","99","--flash-attn","on","--parallel","1",
         "--host","127.0.0.1","--port",str(cfg["port"]),"--no-webui","--reasoning","off","--reasoning-budget","0",
         "--chat-template-kwargs",'{"enable_thinking":false}',"--seed","2026"]
    cmd[cmd.index("--parallel")+1]=str(cfg["parallel"])
    flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    return subprocess.Popen(cmd,cwd=RUNTIME,stdout=log_handle,stderr=subprocess.STDOUT,creationflags=flags)


def wait_server(client, process):
    deadline=time.time()+300
    while time.time()<deadline:
        if process.poll() is not None: raise RuntimeError(f"server exited {process.returncode}")
        try:
            if client.get("/health").get("status")=="ok": return
        except Exception: pass
        time.sleep(1)
    raise TimeoutError("server load timeout")


def profile_system(row, condition):
    fields=[("sex",row["sex"]),("age",row["age"]),("age_group",row["age_group"]),("region/province",f"{row['region']} ({row['province']})"),
            ("marital_status",row["marital_status"]),("education_level",row["education_level"]),("occupation",row["occupation"])]
    if condition=="demographics_plus_persona": fields.append(("persona",row["persona"]))
    return SYSTEM_PROMPT+"\n\n합성 응답자 정보:\n"+"\n".join(f"- {k}: {v}" for k,v in fields)


def rotation_assignments(sample):
    assignments={}
    for question in QUESTIONS:
        k=len(v2.QUESTIONS[question]["choices"])
        ordered=sorted((str(r["uuid"]) for r in sample),key=lambda u:hashlib.sha256(f"2026|{question}|{u}".encode()).hexdigest())
        for i,uuid in enumerate(ordered): assignments[(uuid,question)]=i%k
        counts={j:sum(assignments[(u,question)]==j for u in ordered) for j in range(k)}
        target=200 if k==5 else 250
        if set(counts.values())!={target}: raise RuntimeError(f"unbalanced assignment {question}: {counts}")
    return assignments


def question_text(question, cfg, order_codes):
    spec=v2.QUESTIONS[question]
    choices="\n".join(f"{cfg['labels'][code]} = {spec['choices'][code]}" for code in order_codes)
    return f"{spec['text']}\n\n{choices}\n\n선택지 코드 하나를 JSON 값으로 선택하십시오. 이유는 쓰지 마십시오."


def validate_tokens(client, model_key, cfg):
    observed={label:client.post("/tokenize",{"content":label,"add_special":False,"with_pieces":True})["tokens"] for label in set(cfg["labels"].values())}
    if not all(len(observed[cfg["labels"][code]])==1 for code in range(1,6)):
        raise RuntimeError(f"locked label tokenization changed: {observed}")
    ids={code:observed[cfg["labels"][code]][0]["id"] for code in range(1,6)}
    expected={"qwen":{1:16,2:17,3:18,4:19,5:20},"exaone":{1:378,2:379,3:380,4:381,5:382},"gemma":{1:562,2:603,3:565,4:622,5:645}}[model_key]
    if ids!=expected: raise RuntimeError(f"token IDs changed: {ids} vs {expected}")
    return ids, observed


def extract(client, cfg, token_ids, row, condition, question, design, rotation):
    valid=list(v2.QUESTIONS[question]["choices"])
    order=valid if design=="exact_replication_original_order" else valid[rotation:]+valid[:rotation]
    messages=[{"role":"system","content":profile_system(row,condition)},{"role":"user","content":question_text(question,cfg,order)}]
    start=time.perf_counter(); rendered=client.post("/apply-template",{"messages":messages})["prompt"]
    # Exact assistant prefix matches the previously validated model-family setup.
    prefix=rendered+'{"'+question.lower()+'":'+(" " if cfg["prefix_space"] else "")
    allowed="".join(cfg["labels"][c] for c in valid)
    grammar=(f"root ::= [{min(valid)}-{max(valid)}]" if model_key_global!="gemma" else f'root ::= " " [{allowed}]')
    payload={"prompt":prefix,"n_predict":1,"temperature":-1,"top_k":0,"top_p":1.0,"min_p":0.0,"samplers":["temperature"],
             "grammar":grammar,"n_probs":64,"post_sampling_probs":False,"return_tokens":True,"cache_prompt":False,"seed":2026}
    completion=client.post("/completion",payload); elapsed=time.perf_counter()-start
    top=completion["completion_probabilities"][0]["top_logprobs"]; by_id={e["id"]:e for e in top}
    missing=[c for c in valid if token_ids[c] not in by_id]
    if missing:
        payload["n_probs"]=512; completion=client.post("/completion",payload); top=completion["completion_probabilities"][0]["top_logprobs"]; by_id={e["id"]:e for e in top}
        missing=[c for c in valid if token_ids[c] not in by_id]
    if missing: raise RuntimeError(f"choice tokens missing {question}: {missing}")
    lps=[float(by_id[token_ids[c]]["logprob"]) for c in valid]; mx=max(lps); rel=[x-mx for x in lps]; weights=[math.exp(x) for x in rel]
    probs=[x/sum(weights) for x in weights]; probs[-1]=1-sum(probs[:-1]); greedy=valid[max(range(len(valid)),key=lambda i:(probs[i],-valid[i]))]
    out={"uuid":str(row["uuid"]),"model":cfg["display"],"persona_condition":condition,"question":question,"design":design,"rotation":rotation,
         "display_order":",".join(map(str,order)),"greedy_choice":greedy,"api_generated_token_id":completion["tokens"][0],"api_generated_content":completion["content"],
         "inference_time_seconds":round(elapsed,6),"n_probs_used":payload["n_probs"],"max_probability":max(probs),
         "entropy_nats":-sum(p*math.log(p) for p in probs if p>0),"top1_top2_margin":sorted(probs,reverse=True)[0]-sorted(probs,reverse=True)[1]}
    for code in range(1,6):
        if code in valid:
            i=valid.index(code); out[f"token_id_choice_{code}"]=token_ids[code]; out[f"relative_logit_choice_{code}"]=rel[i]; out[f"logprob_choice_{code}"]=lps[i]; out[f"prob_choice_{code}"]=probs[i]; out[f"display_position_choice_{code}"]=order.index(code)+1
        else:
            for name in ["token_id","relative_logit","logprob","prob","display_position"]: out[f"{name}_choice_{code}"]=None
    return out


def read_jsonl(path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path,row):
    with path.open("a",encoding="utf-8",newline="\n") as f: f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n"); f.flush()


def export(rows,path):
    ordered=sorted(rows,key=lambda r:(r["design"],r["uuid"],CONDITIONS.index(r["persona_condition"]),QUESTIONS.index(r["question"])))
    with path.with_suffix(".jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for r in ordered: f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    with path.with_suffix(".csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(ordered[0])); w.writeheader(); w.writerows(ordered)


def run_model(model_key, probe_only=False):
    global model_key_global
    model_key_global=model_key; cfg=MODELS[model_key]
    if sha256_file(cfg["path"])!=cfg["sha256"]: raise RuntimeError(f"{model_key} model hash mismatch")
    sample=pq.read_table(SAMPLE).to_pylist()
    if len(sample)!=1000 or len({str(r['uuid']) for r in sample})!=1000: raise RuntimeError("invalid locked sample")
    assignments=rotation_assignments(sample)
    RAW.mkdir(parents=True,exist_ok=True); PROBS.mkdir(parents=True,exist_ok=True)
    checkpoint=RAW/f"{model_key}_probability_checkpoint.jsonl"; server_log=RAW/f"{model_key}_llama_server.log"
    existing=read_jsonl(checkpoint); keys={(r["uuid"],r["persona_condition"],r["question"],r["design"]) for r in existing}
    if len(keys)!=len(existing): raise RuntimeError("duplicate checkpoint keys")
    gpu=GpuMonitor(); gpu.start(); process=None; started=time.perf_counter()
    with server_log.open("a",encoding="utf-8") as log:
        try:
            process=start_server(cfg,log); client=Client(cfg["port"]); wait_server(client,process); token_ids,observed=validate_tokens(client,model_key,cfg)
            if probe_only:
                print(json.dumps(extract(client,cfg,token_ids,sample[0],CONDITIONS[0],"Q2","exact_replication_original_order",0),ensure_ascii=False,indent=2)); return
            total=16000; done=len(existing); tasks=[]
            for design in ["exact_replication_original_order","counterbalanced_order"]:
                for row in sample:
                    uuid=str(row["uuid"])
                    for condition in CONDITIONS:
                        for question in QUESTIONS:
                            key=(uuid,condition,question,design)
                            if key in keys: continue
                            rotation=0 if design.startswith("exact") else assignments[(uuid,question)]
                            tasks.append((row,condition,question,design,rotation,key))
            def do_task(task):
                row,condition,question,design,rotation,key=task
                return key,extract(client,cfg,token_ids,row,condition,question,design,rotation)
            with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
                for key,result in pool.map(do_task,tasks):
                    append_jsonl(checkpoint,result); keys.add(key); done+=1
                    if done%100==0: print(f"{model_key}={done}/{total}",flush=True)
            wall=time.perf_counter()-started
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill()
            gpu.stop()
    rows=read_jsonl(checkpoint)
    if len(rows)!=16000: raise RuntimeError(f"{model_key}: expected 16000 rows, got {len(rows)}")
    export(rows,PROBS/f"{model_key}_validation_1000_choice_probabilities")
    env={"model":cfg["display"],"model_path":str(cfg["path"].relative_to(ROOT)).replace("\\","/"),"model_sha256":cfg["sha256"],
         "llama_cpp_build":10514,"llama_cpp_commit":"7221e24f5","chat_template":"GGUF embedded template via /apply-template","context_length":4096,
         "gpu_offload_layers_requested":99,"gpu":gpu.name,"vram_total_mib":gpu.total,"vram_baseline_mib":gpu.baseline,"peak_vram_mib":gpu.peak,
         "peak_vram_increment_mib":gpu.peak-gpu.baseline,"peak_gpu_utilization_percent":gpu.util,"parallel_independent_slots":cfg["parallel"],"wall_time_seconds":wall,"rows":len(rows),
         "token_ids":token_ids,"tokenization":observed,"settings":{"temperature":-1,"sampling":"disabled/greedy","top_k":0,"top_p":1.0,"cache_prompt":False,"seed":2026,"reasoning":"off"}}
    (RAW/f"{model_key}_run_environment.json").write_text(json.dumps(env,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"model":model_key,"rows":len(rows),"wall_seconds":wall,"peak_vram_mib":gpu.peak},indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["qwen","exaone","gemma","all"],required=True); p.add_argument("--probe-only",action="store_true"); args=p.parse_args()
    selected=list(MODELS) if args.model=="all" else [args.model]
    for model in selected: run_model(model,args.probe_only)


model_key_global=""
if __name__=="__main__": main()
