from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_smoke_v2 as v2
import run_validation_1000_locked as runner


ROOT=Path(__file__).resolve().parents[1]
LOCKED=ROOT/"outputs"/"validation_1000_locked"
PLAN=LOCKED/"analysis_plan_locked.json"
PLAN_HASH=LOCKED/"analysis_plan_locked.sha256"


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()


def preserved_manifest():
    paths=[]
    for pattern in ["smoke_100*","smoke_v2*","smoke_v3*","smoke_v4*","smoke_v5*"]:
        paths += [p for p in (ROOT/"outputs").glob(pattern) if p.is_file()]
    paths += [p for p in (ROOT/"scripts").glob("*smoke*") if p.is_file()]
    paths += [ROOT/"data"/"derived"/"smoke_100_seed2026.parquet",ROOT/"data"/"derived"/"pilot_1000_seed42.parquet"]
    return {str(p.relative_to(ROOT)).replace("\\","/"):{"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(set(paths)) if p.exists()}


def main():
    LOCKED.mkdir(parents=True,exist_ok=True)
    sample=ROOT/"data"/"derived"/"validation_1000_locked_seed42.parquet"
    benchmark=LOCKED/"benchmark"/"human_benchmark_subgroups_20260818.json"
    source_pdf=LOCKED/"benchmark"/"source"/"reform_institute_poll_2026-16_report.pdf"
    questions={q:v2.QUESTIONS[q] for q in runner.QUESTIONS}
    model_data={}
    for key,cfg in runner.MODELS.items():
        actual=sha256_file(cfg["path"])
        if actual!=cfg["sha256"]: raise RuntimeError(f"model changed: {key}")
        model_data[key]={"display":cfg["display"],"path":str(cfg["path"].relative_to(ROOT)).replace("\\","/"),"sha256":actual,
                         "labels":cfg["labels"],"chat_template":"model GGUF embedded template; /apply-template; exact smoke-validated model file",
                         "llama_cpp_build":10514,"llama_cpp_commit":"7221e24f5"}
    plan={
        "phase":"locked_validation","status":"locked_before_inference","locked_at_local_date":"2026-08-20",
        "sample":{"path":str(sample.relative_to(ROOT)).replace("\\","/"),"sha256":sha256_file(sample),"n":1000,"sampling_seed":42,
                  "targets":{"sex":{"남성":495,"여성":505},"age_group":{"19-29":146,"30-39":150,"40-49":169,"50-59":193,"60-69":179,"70+":163},
                             "region":{"서울":186,"인천·경기":325,"대전·세종·충청":108,"광주·전라":96,"대구·경북":95,"부산·울산·경남":147,"강원·제주":43}}},
        "models":model_data,
        "persona_conditions":runner.CONDITIONS,
        "persona_fields":{"demographics_only":["sex","age","age_group","region/province","marital_status","education_level","occupation"],
                           "demographics_plus_persona":["sex","age","age_group","region/province","marital_status","education_level","occupation","persona"]},
        "system_prompt":runner.SYSTEM_PROMPT,"system_prompt_sha256":hashlib.sha256(runner.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "questions":questions,"questions_sha256":sha256_json(questions),"Q1_excluded":True,
        "inference":{"context_length":4096,"next_tokens":1,"temperature":-1,"sampling":False,"top_k":0,"top_p":1.0,"min_p":0.0,"cache_prompt":False,
                     "seed":2026,"reasoning":"off","n_probs_initial":64,"n_probs_fallback":512,"probability":"softmax within valid semantic choice tokens"},
        "designs":{"primary":"exact_replication_original_order","secondary":"counterbalanced_order",
                   "counterbalanced_assignment":"Sort all UUIDs by SHA256('2026|question|uuid'); rotation = zero-based rank modulo K. Independent of model and persona condition. Q2/Q3/Q5: 200 per rotation; Q4: 250 per rotation."},
        "primary_analysis":{"estimate":"unweighted mean of original-order persona choice probabilities","question_metric":"TVD to human overall distribution",
                            "summary_metric":"mean Q2-Q5 TVD within model and persona condition","posthoc_weighting":False},
        "secondary_analysis":["MAE","Jensen-Shannon divergence","greedy distribution","counterbalanced probability distribution"],
        "subgroup_analysis":{"dimensions":["sex","age_group","region"],"metric":"TVD","joint_subgroups":"not estimated"},
        "bootstrap":{"unit":"persona","replicates":1000,"seed":2026,"interval":"percentile 95%","targets":["category probability","human TVD"]},
        "model_dependence":{"pairs":["qwen-exaone","qwen-gemma","exaone-gemma"],"metrics":["population TVD","mean same-uuid probability-vector TVD"]},
        "persona_effect":{"paired_by":"model × uuid × question","metrics":["mean TVD","median TVD","argmax change rate","entropy change"]},
        "position_effect":{"comparison":"original vs UUID-assigned cyclic rotation","metrics":["population TVD","individual TVD","argmax change rate","entropy change","semantic option position effect","unknown position effect"]},
        "no_tuning_rule":["no benchmark-driven prompt change","no temperature change","no logits scaling","no probability calibration","no category correction",
                          "no optimized ensemble weighting","no model exclusion","no question exclusion","no adverse-result rerun","no benchmark-based model selection"],
        "benchmark":{"path":str(benchmark.relative_to(ROOT)).replace("\\","/"),"sha256":sha256_file(benchmark),"source_pdf_sha256":sha256_file(source_pdf),
                     "use":"evaluation only; never exposed to prompts"},
        "limitations":["Human youngest category is 18-29; synthetic category is 19-29 because Nemotron minimum age is 19.",
                       "Human raw microdata are unavailable; human subgroup uncertainty and unreported joint subgroups are not constructed."]
    }
    serialized=json.dumps(plan,ensure_ascii=False,indent=2)+"\n"
    if PLAN.exists() and PLAN.read_text(encoding="utf-8")!=serialized: raise RuntimeError("locked analysis plan already exists with different content")
    PLAN.write_text(serialized,encoding="utf-8")
    digest=sha256_file(PLAN); PLAN_HASH.write_text(digest+"  analysis_plan_locked.json\n",encoding="ascii")
    preservation={"created_before_inference":True,"files":preserved_manifest()}
    (LOCKED/"preservation_manifest_smoke_v1_v5.json").write_text(json.dumps(preservation,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"analysis_plan_sha256":digest,"preserved_files":len(preservation["files"])},indent=2))


if __name__=="__main__": main()
