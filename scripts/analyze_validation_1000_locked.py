from __future__ import annotations

import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; LOCKED=ROOT/"outputs"/"validation_1000_locked"
PROBS=LOCKED/"probabilities"; BENCH=LOCKED/"benchmark"; SUB=LOCKED/"subgroups"; BOOT=LOCKED/"bootstrap"
MOD=LOCKED/"model_dependence"; PER=LOCKED/"persona_effect"; POS=LOCKED/"position_effect"
MODELS=["qwen","exaone","gemma"]; CONDITIONS=["demographics_only","demographics_plus_persona"]; QUESTIONS=["Q2","Q3","Q4","Q5"]
DISPLAY={"qwen":"Qwen3-8B-Q4_K_M","exaone":"EXAONE-3.5-7.8B-Instruct-Q4_K_M","gemma":"Gemma-3-12B-QAT-Q4_0"}
CHOICES={"Q2":5,"Q3":5,"Q4":4,"Q5":5}

def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=fields or list(rows[0])
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def norm(x):
    a=np.asarray(x,dtype=float); return a/a.sum()
def tvd(a,b): return float(np.abs(norm(a)-norm(b)).sum()/2)
def jsd(a,b):
    a,b=norm(a),norm(b); m=(a+b)/2
    return float(sum(x*math.log(x/y) for x,y in zip(a,m) if x>0)/2+sum(x*math.log(x/y) for x,y in zip(b,m) if x>0)/2)
def probs(row): return np.array([row[f"prob_choice_{c}"] for c in range(1,CHOICES[row["question"]]+1)])
def entropy(p): return float(-sum(x*math.log(x) for x in p if x>0))

def main():
    for d in [SUB,BOOT,MOD,PER,POS]: d.mkdir(parents=True,exist_ok=True)
    sample=pq.read_table(ROOT/"data"/"derived"/"validation_1000_locked_seed42.parquet").to_pylist(); meta={str(r["uuid"]):r for r in sample}
    benchmark=json.loads((BENCH/"human_benchmark_subgroups_20260818.json").read_text(encoding="utf-8"))
    allrows={m:read_jsonl(PROBS/f"{m}_validation_1000_choice_probabilities.jsonl") for m in MODELS}
    for m,rows in allrows.items():
        keys={(r["uuid"],r["persona_condition"],r["question"],r["design"]) for r in rows}
        if len(rows)!=16000 or len(keys)!=16000: raise RuntimeError(f"{m} row/key validation failed")
        if any(abs(sum(probs(r))-1)>1e-12 for r in rows): raise RuntimeError(f"{m} probability sum failed")
    grouped={}
    for m,rows in allrows.items():
        for d in ["exact_replication_original_order","counterbalanced_order"]:
            for c in CONDITIONS:
                for q in QUESTIONS: grouped[(m,d,c,q)]=[r for r in rows if r["design"]==d and r["persona_condition"]==c and r["question"]==q]

    population=[]
    for (m,d,c,q),g in grouped.items():
        mean=np.mean([probs(r) for r in g],axis=0); greedy=np.bincount([int(r["greedy_choice"]) for r in g],minlength=CHOICES[q]+1)[1:]/len(g)
        for method,vec in [("probability_aggregation",mean),("greedy_diagnostic",greedy)]:
            for code,x in enumerate(vec,1): population.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"design":d,"method":method,"question":q,"response_code":code,"proportion":x,"percent":100*x,"n_personas":len(g),"primary":d.startswith("exact") and method=="probability_aggregation"})
    write_csv(LOCKED/"validation_1000_primary_distribution.csv",population)

    comparison=[]
    for m in MODELS:
        for d in ["exact_replication_original_order","counterbalanced_order"]:
            for c in CONDITIONS:
                for q in QUESTIONS:
                    syn=np.mean([probs(r) for r in grouped[(m,d,c,q)]],axis=0); human=np.array(benchmark["overall"][q])/100
                    metric_tvd=tvd(syn,human); metric_mae=float(np.mean(np.abs(syn*100-human*100))); metric_js=jsd(syn,human)
                    for code,(s,h) in enumerate(zip(syn,human),1): comparison.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"design":d,"question":q,"response_code":code,"human_percent_reported":h*100,"synthetic_percent":s*100,"percentage_point_error":s*100-h*100,"absolute_percentage_point_error":abs(s*100-h*100),"mae_percentage_points":metric_mae,"tvd":metric_tvd,"js_divergence_nats":metric_js,"primary":d.startswith("exact")})
    write_csv(LOCKED/"validation_1000_human_comparison.csv",comparison)

    subgroup=[]
    dim_specs={"sex":(["남성","여성"],lambda r:"남성" if r["sex"]=="남자" else "여성"),"age":(list(benchmark["subgroups"]["age"]),lambda r:r["age_group"]),"region":(list(benchmark["subgroups"]["region"]),lambda r:r["region"])}
    for m in MODELS:
      for c in CONDITIONS:
       for dim,(levels,getlevel) in dim_specs.items():
        dim_rows=[]
        for level in levels:
         ids={u for u,r in meta.items() if getlevel(r)==level}
         for q in QUESTIONS:
          g=[r for r in grouped[(m,"exact_replication_original_order",c,q)] if r["uuid"] in ids]; syn=np.mean([probs(r) for r in g],axis=0); human=np.array(benchmark["subgroups"][dim][level][q])/100; x=tvd(syn,human)
          rec={"record_type":"subgroup","model":DISPLAY[m],"model_key":m,"persona_condition":c,"dimension":dim,"subgroup":level,"question":q,"synthetic_n":len(g),"human_weighted_n":benchmark["subgroups"][dim][level]["human_weighted_n"],"tvd":x}
          for code,(s,h) in enumerate(zip(syn,human),1): rec[f"synthetic_percent_choice_{code}"]=100*s; rec[f"human_percent_choice_{code}"]=100*h
          subgroup.append(rec); dim_rows.append(rec)
        subgroup.append({"record_type":"pattern_summary","model":DISPLAY[m],"model_key":m,"persona_condition":c,"dimension":dim,"subgroup":"ALL","question":"Q2-Q5","mean_tvd":statistics.fmean(r["tvd"] for r in dim_rows),"median_tvd":statistics.median(r["tvd"] for r in dim_rows)})
    write_csv(LOCKED/"validation_1000_subgroup_comparison.csv",subgroup,sorted({k for r in subgroup for k in r}))

    # Persona-level bootstrap of original-order probability aggregation.
    boot=[]; rng=np.random.default_rng(2026)
    for m in MODELS:
     for c in CONDITIONS:
      for q in QUESTIONS:
       arr=np.stack([probs(r) for r in grouped[(m,"exact_replication_original_order",c,q)]]); human=np.array(benchmark["overall"][q])/100
       idx=rng.integers(0,len(arr),size=(1000,len(arr))); means=arr[idx].mean(axis=1); tv=np.array([tvd(x,human) for x in means])
       point=arr.mean(axis=0)
       for code in range(CHOICES[q]): boot.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question":q,"metric":"category_probability","response_code":code+1,"point_estimate":point[code],"ci95_lower":np.quantile(means[:,code],.025),"ci95_upper":np.quantile(means[:,code],.975),"replicates":1000,"seed":2026})
       boot.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question":q,"metric":"human_tvd","response_code":"ALL","point_estimate":tvd(point,human),"ci95_lower":np.quantile(tv,.025),"ci95_upper":np.quantile(tv,.975),"replicates":1000,"seed":2026})
    write_csv(LOCKED/"validation_1000_bootstrap.csv",boot)

    # Model dependence, population and paired same-person vector TVD.
    dependence=[]
    for c in CONDITIONS:
     for q in QUESTIONS:
      for a,b in [("qwen","exaone"),("qwen","gemma"),("exaone","gemma")]:
       ga=grouped[(a,"exact_replication_original_order",c,q)]; gb=grouped[(b,"exact_replication_original_order",c,q)]; la={r["uuid"]:r for r in ga}; lb={r["uuid"]:r for r in gb}; indiv=[tvd(probs(la[u]),probs(lb[u])) for u in la]
       dependence.append({"model_a":DISPLAY[a],"model_b":DISPLAY[b],"persona_condition":c,"question":q,"population_tvd":tvd(np.mean([probs(r) for r in ga],0),np.mean([probs(r) for r in gb],0)),"mean_same_uuid_tvd":statistics.fmean(indiv),"median_same_uuid_tvd":statistics.median(indiv),"n_paired":len(indiv)})
    write_csv(LOCKED/"validation_1000_three_model_comparison.csv",dependence)
    write_csv(MOD/"pairwise_model_dependence.csv",dependence)

    persona=[]
    for m in MODELS:
     for q in QUESTIONS:
      a={r["uuid"]:r for r in grouped[(m,"exact_replication_original_order","demographics_only",q)]}; b={r["uuid"]:r for r in grouped[(m,"exact_replication_original_order","demographics_plus_persona",q)]}
      vals=[tvd(probs(a[u]),probs(b[u])) for u in a]; changes=[a[u]["greedy_choice"]!=b[u]["greedy_choice"] for u in a]; ed=[entropy(probs(b[u]))-entropy(probs(a[u])) for u in a]
      h=np.array(benchmark["overall"][q])/100; ta=tvd(np.mean([probs(x) for x in a.values()],0),h); tb=tvd(np.mean([probs(x) for x in b.values()],0),h)
      persona.append({"model":DISPLAY[m],"model_key":m,"question":q,"mean_tvd":statistics.fmean(vals),"median_tvd":statistics.median(vals),"argmax_change_rate":statistics.fmean(changes),"mean_entropy_change_persona_minus_demographics":statistics.fmean(ed),"human_tvd_demographics_only":ta,"human_tvd_plus_persona":tb,"human_tvd_change_persona_minus_demographics":tb-ta,"n_paired":1000})
    write_csv(LOCKED/"validation_1000_persona_effect.csv",persona)

    position=[]; semantic=[]
    for m in MODELS:
     for c in CONDITIONS:
      for q in QUESTIONS:
       a={r["uuid"]:r for r in grouped[(m,"exact_replication_original_order",c,q)]}; b={r["uuid"]:r for r in grouped[(m,"counterbalanced_order",c,q)]}; ids=list(a)
       iv=[tvd(probs(a[u]),probs(b[u])) for u in ids]; ch=[a[u]["greedy_choice"]!=b[u]["greedy_choice"] for u in ids]; ed=[entropy(probs(b[u]))-entropy(probs(a[u])) for u in ids]
       unk=CHOICES[q]; last=[b[u][f"prob_choice_{unk}"] for u in ids if b[u][f"display_position_choice_{unk}"]==CHOICES[q]]; non=[b[u][f"prob_choice_{unk}"] for u in ids if b[u][f"display_position_choice_{unk}"]!=CHOICES[q]]
       position.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question":q,"population_tvd_original_vs_counterbalanced":tvd(np.mean([probs(x) for x in a.values()],0),np.mean([probs(x) for x in b.values()],0)),"mean_individual_tvd":statistics.fmean(iv),"median_individual_tvd":statistics.median(iv),"argmax_change_rate":statistics.fmean(ch),"mean_entropy_change_counterbalanced_minus_original":statistics.fmean(ed),"unknown_prob_when_last":statistics.fmean(last),"unknown_prob_when_not_last":statistics.fmean(non),"unknown_last_minus_nonlast":statistics.fmean(last)-statistics.fmean(non),"n":1000})
       for code in range(1,CHOICES[q]+1):
        for pos in range(1,CHOICES[q]+1):
         vals=[b[u][f"prob_choice_{code}"] for u in ids if b[u][f"display_position_choice_{code}"]==pos]
         semantic.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question":q,"semantic_choice":code,"display_position":pos,"mean_probability":statistics.fmean(vals),"n":len(vals)})
    write_csv(LOCKED/"validation_1000_position_effect.csv",position); write_csv(POS/"semantic_choice_by_display_position.csv",semantic)

    # Concentration/timing and compact report.
    diagnostics=[]
    for (m,d,c,q),g in grouped.items():
     diagnostics.append({"model":DISPLAY[m],"model_key":m,"design":d,"persona_condition":c,"question":q,"mean_entropy":statistics.fmean(r["entropy_nats"] for r in g),"mean_max_probability":statistics.fmean(r["max_probability"] for r in g),"mean_margin":statistics.fmean(r["top1_top2_margin"] for r in g),"unknown_probability":statistics.fmean(r[f"prob_choice_{CHOICES[q]}"] for r in g),"mean_inference_seconds":statistics.fmean(r["inference_time_seconds"] for r in g),"median_inference_seconds":statistics.median(r["inference_time_seconds"] for r in g)})
    write_csv(PROBS/"validation_1000_concentration_and_timing.csv",diagnostics)
    env={m:json.loads((LOCKED/"raw"/f"{m}_run_environment.json").read_text(encoding="utf-8")) for m in MODELS}
    primary=[r for r in comparison if r["primary"]]; mt={}
    for m in MODELS:
     for c in CONDITIONS: mt[(m,c)]=statistics.fmean({(r["question"],r["tvd"]) for r in primary if r["model_key"]==m and r["persona_condition"]==c}.pop()[1] if False else [next(r["tvd"] for r in primary if r["model_key"]==m and r["persona_condition"]==c and r["question"]==q) for q in QUESTIONS])
    lines=["# Validation 1,000 locked report","","This is a locked validation, not an estimate of Korean public opinion. Human benchmarks were used only after inference for evaluation.","","## Execution","", "| Model | Rows | Wall min | Peak VRAM MiB |", "| --- | ---: | ---: | ---: |"]
    for m in MODELS: lines.append(f"| {DISPLAY[m]} | {env[m]['rows']:,} | {env[m]['wall_time_seconds']/60:.1f} | {env[m]['peak_vram_mib']:,} |")
    lines += ["","## Primary mean TVD (Q2-Q5)","","| Model | demographics only | plus persona |","| --- | ---: | ---: |"]
    for m in MODELS: lines.append(f"| {DISPLAY[m]} | {mt[(m,CONDITIONS[0])]:.4f} | {mt[(m,CONDITIONS[1])]:.4f} |")
    lines += ["","## Design and limitations","","- Primary: original-order probability aggregation. Counterbalanced and greedy results remain secondary.","- The human 18-29 group is compared with synthetic 19-29; Nemotron has no 18-year-olds.","- Human subgroup percentages are rounded and raw microdata are unavailable; no joint subgroup benchmark or human subgroup uncertainty was fabricated.","- The three models are compared as model-dependence diagnostics; closeness does not establish population representativeness.",""]
    (LOCKED/"validation_1000_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"inference_rows":sum(len(x) for x in allrows.values()),"mean_tvd":{f"{m}:{c}":mt[(m,c)] for m in MODELS for c in CONDITIONS}},indent=2))

if __name__=="__main__": main()
