from __future__ import annotations

import csv, hashlib, json, math, os, statistics, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hard_choice_common import ROOT, QUESTIONS, QUESTION_IDS, CONDITIONS, ORDER_CONDITIONS, MODELS, SAMPLE_PATH, load_locked_sample, now, sha256_file
import run_hard_choice_full_v1 as full
import run_hard_choice_pilot10_v2 as prior

BASE = full.FULL / "analysis_final"
TABLES, FIGURES, AUDITS = BASE / "tables", BASE / "figures", BASE / "audits"
REPORT = BASE / "hard_choice_final_analysis_report.md"
KEY_RESULTS = BASE / "hard_choice_key_results.json"
MANIFEST = BASE / "analysis_manifest.json"
BENCHMARK = ROOT / "outputs/validation_1000_locked/benchmark/human_benchmark_subgroups_20260818.json"
PROBABILITY = ROOT / "outputs/validation_1000_locked/validation_1000_human_comparison.csv"
STOCHASTICITY = full.FULL / "analysis/motif_run1_vs_run2_stochasticity.json"
DISPLAY = {k: MODELS[k]["display"] for k in ("qwen", "exaone", "gemma", "motif")}
PATHS = {"qwen": full.RAW/"qwen/observations.parquet", "exaone": full.RAW/"exaone/observations.parquet", "gemma": full.RAW/"gemma/observations.parquet", "motif": full.RAW/"motif_run1/observations.parquet"}
RUN2 = full.RAW / "motif_replicate2/observations.parquet"

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as h: return list(csv.DictReader(h))
def load(path): return pq.read_table(path).to_pylist()
def key(r): return r["uuid"],r["question_id"],r["persona_condition"],r["order_condition"]
def idx(rows): return {key(r):r for r in rows}
def human_vector(b,q,section=None,level=None):
    value=b["overall"][q] if section is None else b["subgroups"][section][level][q]
    return np.asarray(value,dtype=float)/100.0  # frozen primary rule: no renormalization
def dist(rows,q):
    valid=[r for r in rows if r.get("semantic_valid") and r.get("semantic_choice") is not None]
    k=len(QUESTIONS[q]["choices"]); counts=np.asarray([sum(int(r["semantic_choice"])==c for r in valid) for c in range(1,k+1)],float)
    return counts/len(valid),len(valid),len(rows)-len(valid)
def tvd(a,b): return float(.5*np.abs(np.asarray(a)-np.asarray(b)).sum())
def jsd(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=.5*(a+b)
    def kl(x,y):
        z=x>0; return float(np.sum(x[z]*np.log(x[z]/y[z])))
    return .5*kl(a,m)+.5*kl(b,m)
def mean(xs): return statistics.mean(xs)
def pct(x): return f"{100*x:.2f}%"
def md_table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rows])

def integrity(rows):
    keys=[key(r) for r in rows]
    return {"observations":len(rows),"unique_keys":len(set(keys)),"duplicate_keys":len(keys)-len(set(keys)),"semantic_valid_count":sum(bool(r.get("semantic_valid")) for r in rows),"semantic_invalid_count":sum(not r.get("semantic_valid") for r in rows),"semantic_valid_rate":sum(bool(r.get("semantic_valid")) for r in rows)/len(rows),"format_compliant_count":sum(bool(r.get("format_compliant")) for r in rows),"format_compliant_rate":sum(bool(r.get("format_compliant")) for r in rows)/len(rows),"api_model_failures":sum(bool(r.get("api_failure")) for r in rows)}

def savefig(fig,name):
    fig.savefig(FIGURES/f"{name}.png",dpi=220,bbox_inches="tight")
    fig.savefig(FIGURES/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)

def main():
    for d in (TABLES,FIGURES,AUDITS): d.mkdir(parents=True,exist_ok=True)
    if sha256_file(full.PROTOCOL)!=full.PROTOCOL_SHA256 or sha256_file(SAMPLE_PATH)!=full.SAMPLE_SHA256: raise RuntimeError("frozen provenance mismatch")
    b=json.loads(BENCHMARK.read_text(encoding="utf-8")); rows={m:load(p) for m,p in PATHS.items()}; run2=load(RUN2)
    integ={m:integrity(v) for m,v in {**rows,"motif_run2":run2}.items()}
    if any(v["observations"]!=16000 or v["unique_keys"]!=16000 or v["duplicate_keys"] for v in integ.values()): raise RuntimeError("incomplete full run")

    # HC1/HC2: original-order human comparison; counterbalanced retained separately for order diagnostics.
    metrics={}; category=[]; summaries=[]
    for m,rs in rows.items():
      for c in CONDITIONS:
       qrows=[]
       for q in QUESTION_IDS:
        ss=[r for r in rs if r["persona_condition"]==c and r["order_condition"]=="original_order" and r["question_id"]==q]
        p,nv,ni=dist(ss,q); h=human_vector(b,q); t=tvd(p,h); a=float(np.abs(p-h).mean()); j=jsd(p,h); metrics[(m,c,"original_order",q)]=(p,t,a,j,nv,ni)
        qrows.append((t,a,j))
        for code,(sp,hp) in enumerate(zip(p,h),1): category.append({"series_type":"synthetic","model":DISPLAY[m],"model_key":m,"persona_condition":c,"question_id":q,"response_code":code,"semantic_valid_n":nv,"total_n":len(ss),"synthetic_percent":sp*100,"human_percent":hp*100,"percentage_point_error":(sp-hp)*100,"absolute_percentage_point_error":abs(sp-hp)*100,"tvd":t,"mae":a,"jsd_nats":j})
       summaries.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"order_condition":"original_order","mean_q2_q5_tvd":mean(x[0] for x in qrows),"mean_q2_q5_mae":mean(x[1] for x in qrows),"mean_q2_q5_jsd_nats":mean(x[2] for x in qrows)})
    # Explicit human rows in HC2.
    for q in QUESTION_IDS:
      h=human_vector(b,q)
      for code,hp in enumerate(h,1): category.append({"series_type":"human","model":"Human survey benchmark","model_key":"human","persona_condition":"not_applicable","question_id":q,"response_code":code,"semantic_valid_n":"","total_n":1006,"synthetic_percent":"","human_percent":hp*100,"percentage_point_error":"","absolute_percentage_point_error":"","tvd":0,"mae":0,"jsd_nats":0})

    # Bootstrap paired at persona level, same frozen draw matrix for every cell.
    sample=load_locked_sample(); uuids=[str(r["uuid"]) for r in sample]; rng=np.random.default_rng(2026); draws=rng.integers(0,1000,size=(1000,1000)); boots=[]
    for m,rs in rows.items():
      ix=idx(rs)
      for c in CONDITIONS:
        arrays={q:np.asarray([int(ix[(u,q,c,"original_order")]["semantic_choice"]) if ix[(u,q,c,"original_order")]["semantic_valid"] else 0 for u in uuids]) for q in QUESTION_IDS}
        vals=[]
        for draw in draws:
          qt=[]
          for q in QUESTION_IDS:
            x=arrays[q][draw]; x=x[x>0]; p=np.asarray([(x==code).mean() for code in range(1,len(QUESTIONS[q]["choices"])+1)]); qt.append(tvd(p,human_vector(b,q)))
          vals.append(mean(qt))
        point=next(x["mean_q2_q5_tvd"] for x in summaries if x["model_key"]==m and x["persona_condition"]==c)
        rec={"model":DISPLAY[m],"model_key":m,"persona_condition":c,"point_mean_tvd":point,"bootstrap_mean":float(np.mean(vals)),"ci_lower_2_5":float(np.percentile(vals,2.5)),"ci_upper_97_5":float(np.percentile(vals,97.5)),"bootstrap_replicates":1000,"seed":2026}; boots.append(rec)
        next(x for x in summaries if x["model_key"]==m and x["persona_condition"]==c).update({k:rec[k] for k in ("bootstrap_mean","ci_lower_2_5","ci_upper_97_5","bootstrap_replicates","seed")})
    write_csv(TABLES/"HC1_four_model_human_original_order.csv",summaries); write_csv(TABLES/"HC2_four_model_category_distributions.csv",category)

    # Populate counterbalanced metrics, model dependence, persona and order effects.
    for m,rs in rows.items():
      for c in CONDITIONS:
       for q in QUESTION_IDS:
        ss=[r for r in rs if r["persona_condition"]==c and r["order_condition"]=="counterbalanced_order" and r["question_id"]==q]; p,nv,ni=dist(ss,q); h=human_vector(b,q)
        metrics[(m,c,"counterbalanced_order",q)]=(p,tvd(p,h),float(np.abs(p-h).mean()),jsd(p,h),nv,ni)
    dependence=[]
    for a,z in combinations(rows,2):
      ia,iz=idx(rows[a]),idx(rows[z])
      for c in CONDITIONS:
       for order in ORDER_CONDITIONS:
        for q in QUESTION_IDS:
          p1=metrics[(a,c,order,q)][0]; p2=metrics[(z,c,order,q)][0]; pairs=[(ia[(u,q,c,order)],iz[(u,q,c,order)]) for u in uuids if ia[(u,q,c,order)]["semantic_valid"] and iz[(u,q,c,order)]["semantic_valid"]]
          dependence.append({"model_a":DISPLAY[a],"model_b":DISPLAY[z],"model_pair":f"{a}-{z}","persona_condition":c,"order_condition":order,"question_id":q,"population_tvd":tvd(p1,p2),"same_uuid_disagreement_rate":sum(x["semantic_choice"]!=y["semantic_choice"] for x,y in pairs)/len(pairs),"paired_semantic_valid_n":len(pairs)})
    # append means per pair/condition/order
    for a,z in combinations(rows,2):
      for c in CONDITIONS:
       for order in ORDER_CONDITIONS:
        ss=[r for r in dependence if r["model_pair"]==f"{a}-{z}" and r["persona_condition"]==c and r["order_condition"]==order and r["question_id"]!="MEAN"]
        dependence.append({"model_a":DISPLAY[a],"model_b":DISPLAY[z],"model_pair":f"{a}-{z}","persona_condition":c,"order_condition":order,"question_id":"MEAN","population_tvd":mean(r["population_tvd"] for r in ss),"same_uuid_disagreement_rate":mean(r["same_uuid_disagreement_rate"] for r in ss),"paired_semantic_valid_n":sum(r["paired_semantic_valid_n"] for r in ss)})
    write_csv(TABLES/"HC3_model_dependence.csv",dependence)

    persona=[]; orderfx=[]
    for m,rs in rows.items():
      ix=idx(rs)
      for order in ORDER_CONDITIONS:
       for q in QUESTION_IDS:
        pairs=[(ix[(u,q,"demographics_only",order)],ix[(u,q,"demographics_plus_persona",order)]) for u in uuids if ix[(u,q,"demographics_only",order)]["semantic_valid"] and ix[(u,q,"demographics_plus_persona",order)]["semantic_valid"]]
        p1=metrics[(m,"demographics_only",order,q)][0]; p2=metrics[(m,"demographics_plus_persona",order,q)][0]
        persona.append({"model":DISPLAY[m],"model_key":m,"order_condition":order,"question_id":q,"population_tvd":tvd(p1,p2),"same_uuid_switch_rate":sum(a["semantic_choice"]!=z["semantic_choice"] for a,z in pairs)/len(pairs),"paired_n":len(pairs),"human_tvd_demo":metrics[(m,"demographics_only",order,q)][1],"human_tvd_persona":metrics[(m,"demographics_plus_persona",order,q)][1],"human_tvd_delta_persona_minus_demo":metrics[(m,"demographics_plus_persona",order,q)][1]-metrics[(m,"demographics_only",order,q)][1]})
      for c in CONDITIONS:
       for q in QUESTION_IDS:
        pairs=[(ix[(u,q,c,"original_order")],ix[(u,q,c,"counterbalanced_order")]) for u in uuids if ix[(u,q,c,"original_order")]["semantic_valid"] and ix[(u,q,c,"counterbalanced_order")]["semantic_valid"]]
        p1=metrics[(m,c,"original_order",q)][0]; p2=metrics[(m,c,"counterbalanced_order",q)][0]
        orderfx.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question_id":q,"population_tvd_original_vs_counterbalanced":tvd(p1,p2),"same_uuid_switch_rate":sum(a["semantic_choice"]!=z["semantic_choice"] for a,z in pairs)/len(pairs),"paired_n":len(pairs),"counterbalanced_human_tvd_secondary":metrics[(m,c,"counterbalanced_order",q)][1],"original_human_tvd_primary":metrics[(m,c,"original_order",q)][1]})
    for m in rows:
      for order in ORDER_CONDITIONS:
       ss=[r for r in persona if r["model_key"]==m and r["order_condition"]==order and r["question_id"]!="MEAN"]
       persona.append({"model":DISPLAY[m],"model_key":m,"order_condition":order,"question_id":"MEAN","population_tvd":mean(r["population_tvd"] for r in ss),"same_uuid_switch_rate":mean(r["same_uuid_switch_rate"] for r in ss),"paired_n":sum(r["paired_n"] for r in ss),"human_tvd_demo":mean(r["human_tvd_demo"] for r in ss),"human_tvd_persona":mean(r["human_tvd_persona"] for r in ss),"human_tvd_delta_persona_minus_demo":mean(r["human_tvd_delta_persona_minus_demo"] for r in ss)})
      for c in CONDITIONS:
       ss=[r for r in orderfx if r["model_key"]==m and r["persona_condition"]==c and r["question_id"]!="MEAN"]
       orderfx.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"question_id":"MEAN","population_tvd_original_vs_counterbalanced":mean(r["population_tvd_original_vs_counterbalanced"] for r in ss),"same_uuid_switch_rate":mean(r["same_uuid_switch_rate"] for r in ss),"paired_n":sum(r["paired_n"] for r in ss),"counterbalanced_human_tvd_secondary":mean(r["counterbalanced_human_tvd_secondary"] for r in ss),"original_human_tvd_primary":mean(r["original_human_tvd_primary"] for r in ss)})
    write_csv(TABLES/"HC4_persona_effect.csv",persona); write_csv(TABLES/"HC5_order_effect.csv",orderfx)

    format_rows=[]
    for m,v in integ.items(): format_rows.append({"run":m,"model":DISPLAY.get(m,"Motif 3 replicate 2"),**v})
    write_csv(TABLES/"HC6_format_compliance.csv",format_rows)

    # Motif run2 sensitivity: verify existing source-of-truth and add human sensitivity table.
    st=json.loads(STOCHASTICITY.read_text(encoding="utf-8")); i1,i2=idx(rows["motif"]),idx(run2); common=sorted(set(i1)&set(i2)); both=[(i1[k],i2[k]) for k in common if i1[k]["semantic_valid"] and i2[k]["semantic_valid"]]; diff=sum(a["semantic_choice"]!=z["semantic_choice"] for a,z in both)
    if len(both)!=15990 or diff!=1449 or not math.isclose(diff/len(both),st["semantic_disagreement_rate"],abs_tol=1e-15): raise RuntimeError("Motif stochasticity reproduction mismatch")
    motif_sens=[]
    for c in CONDITIONS:
      vals1=[]; vals2=[]
      for q in QUESTION_IDS:
        a=[r for r in rows["motif"] if r["persona_condition"]==c and r["order_condition"]=="original_order" and r["question_id"]==q]; z=[r for r in run2 if r["persona_condition"]==c and r["order_condition"]=="original_order" and r["question_id"]==q]
        p1,n1,_=dist(a,q); p2,n2,_=dist(z,q); h=human_vector(b,q); t1=tvd(p1,h); t2=tvd(p2,h); vals1.append(t1); vals2.append(t2)
        motif_sens.append({"persona_condition":c,"question_id":q,"run1_human_tvd":t1,"run2_human_tvd":t2,"absolute_human_tvd_difference":abs(t2-t1),"run1_vs_run2_population_tvd":tvd(p1,p2),"run1_valid_n":n1,"run2_valid_n":n2,"representative_run":"run1"})
      motif_sens.append({"persona_condition":c,"question_id":"MEAN","run1_human_tvd":mean(vals1),"run2_human_tvd":mean(vals2),"absolute_human_tvd_difference":abs(mean(vals2)-mean(vals1)),"run1_vs_run2_population_tvd":mean(st["population_tvd_by_question_mean_across_conditions_orders"].values()),"run1_valid_n":"","run2_valid_n":"","representative_run":"run1"})
    motif_sens.append({"persona_condition":"all","question_id":"PAIRED_OVERALL","run1_human_tvd":"","run2_human_tvd":"","absolute_human_tvd_difference":"","run1_vs_run2_population_tvd":st["population_tvd_mean_across_16_cells"],"run1_valid_n":len(both),"run2_valid_n":len(both),"representative_run":"run1","same_request_disagreement_rate":diff/len(both),"different_choice_count":diff})
    write_csv(TABLES/"HC7_motif_stochasticity.csv",motif_sens)

    # Probability vs hard-choice; exact locked result rows only.
    prob=read_csv(PROBABILITY); interface=[]
    for m in ("qwen","exaone","gemma"):
      for c in CONDITIONS:
        values=[]
        for q in QUESTION_IDS:
          found={float(r["tvd"]) for r in prob if r["model_key"]==m and r["persona_condition"]==c and r["question"]==q and r["primary"].lower()=="true"}
          if len(found)!=1: raise RuntimeError("probability source ambiguity")
          values.append(next(iter(found)))
        hard=next(r["mean_q2_q5_tvd"] for r in summaries if r["model_key"]==m and r["persona_condition"]==c)
        interface.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"probability_based_original_order_mean_tvd":mean(values),"hard_choice_original_order_mean_tvd":hard,"hard_choice_minus_probability_tvd":hard-mean(values),"interpretation":"descriptive measurement-interface difference; estimands not interchangeable"})
    for c in CONDITIONS:
      interface.append({"model":DISPLAY["motif"],"model_key":"motif","persona_condition":c,"probability_based_original_order_mean_tvd":"NA","hard_choice_original_order_mean_tvd":next(r["mean_q2_q5_tvd"] for r in summaries if r["model_key"]=="motif" and r["persona_condition"]==c),"hard_choice_minus_probability_tvd":"NA","interpretation":"Motif probability vector unavailable"})
    write_csv(TABLES/"HC8_probability_vs_hardchoice.csv",interface)

    # Subgroup external validity only where verified human values exist.
    sample_by={str(r["uuid"]):r for r in sample}; subgroup=[]
    defs={"sex":list(b["subgroups"]["sex"]),"age":list(b["subgroups"]["age"]),"region":list(b["subgroups"]["region"])}
    def level_for(r,dim):
      s=sample_by[r["uuid"]]
      if dim=="sex": return "남성" if s["sex"]=="남자" else "여성"
      return s["age_group"] if dim=="age" else s["region"]
    for m,rs in rows.items():
      for c in CONDITIONS:
       for dim,levels in defs.items():
        for level in levels:
         for q in QUESTION_IDS:
          ss=[r for r in rs if r["persona_condition"]==c and r["order_condition"]=="original_order" and r["question_id"]==q and level_for(r,dim)==level]
          p,nv,ni=dist(ss,q); h=human_vector(b,q,dim,level)
          subgroup.append({"model":DISPLAY[m],"model_key":m,"persona_condition":c,"order_condition":"original_order","dimension":dim,"subgroup":level,"question_id":q,"synthetic_total_n":len(ss),"semantic_valid_n":nv,"semantic_invalid_n":ni,"human_weighted_n":b["subgroups"][dim][level]["human_weighted_n"],"human_tvd":tvd(p,h)})
    write_csv(TABLES/"HC9_subgroup_external_validation.csv",subgroup)

    # Figure HC1: human vs four models, original order, persona facets.
    colors={"Human":"#222222","Qwen":"#4C78A8","EXAONE":"#F58518","Gemma":"#54A24B","Motif":"#B279A2"}
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True)
    for ri,c in enumerate(CONDITIONS):
      for qi,q in enumerate(QUESTION_IDS):
        ax=axes[ri,qi]; h=human_vector(b,q); xs=np.arange(1,len(h)+1); ax.plot(xs,h*100,marker="o",color=colors["Human"],label="Human")
        for m in rows: ax.plot(xs,metrics[(m,c,"original_order",q)][0]*100,marker="o",label=DISPLAY[m].split()[0],color=colors[{"qwen":"Qwen","exaone":"EXAONE","gemma":"Gemma","motif":"Motif"}[m]])
        ax.set_title(f"{q} · {'Demo' if ri==0 else 'Demo + persona'}"); ax.set_xticks(xs); ax.set_xlabel("Semantic response code"); ax.grid(axis="y",alpha=.2)
        if qi==0: ax.set_ylabel("Response share (%)")
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.suptitle("Human benchmark and four-model hard-choice distributions (original order)",y=.995); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.965),ncol=5,frameon=False); fig.tight_layout(rect=(0,0,1,.90)); savefig(fig,"Figure_HC1_human_vs_four_models")

    # HC2 mean TVD with CI.
    fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(4); width=.34
    for j,c in enumerate(CONDITIONS):
      ss=[next(r for r in summaries if r["model_key"]==m and r["persona_condition"]==c) for m in rows]; y=np.array([r["mean_q2_q5_tvd"] for r in ss]); lo=y-np.array([r["ci_lower_2_5"] for r in ss]); hi=np.array([r["ci_upper_97_5"] for r in ss])-y
      ax.bar(x+(j-.5)*width,y,width,label="Demographics" if j==0 else "+ persona",alpha=.85); ax.errorbar(x+(j-.5)*width,y,yerr=[lo,hi],fmt="none",color="black",capsize=3)
    ax.set_xticks(x,[DISPLAY[m].split()[0] for m in rows]); ax.set_ylabel("Mean TVD to human benchmark"); ax.set_title("Original-order hard-choice external validity"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); fig.tight_layout(); savefig(fig,"Figure_HC2_mean_human_tvd")

    # HC3 pairwise dependence original order means.
    orig=[r for r in dependence if r["order_condition"]=="original_order" and r["question_id"]=="MEAN"]
    fig,ax=plt.subplots(figsize=(9,4.8)); labels=sorted({r["model_pair"] for r in orig}); x=np.arange(len(labels)); width=.34
    for j,c in enumerate(CONDITIONS):
      ss=[next(r for r in orig if r["model_pair"]==p and r["persona_condition"]==c) for p in labels]; ax.bar(x+(j-.5)*width,[r["population_tvd"] for r in ss],width,label=("Demographics" if j==0 else "+ persona"))
    ax.set_xticks(x,[p.replace("-","–") for p in labels],rotation=25,ha="right"); ax.set_ylabel("Mean population TVD"); ax.set_title("Four-model dependence (original order)"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); fig.tight_layout(); savefig(fig,"Figure_HC3_pairwise_model_dependence")

    # HC4 persona/order switch rates (means).
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for m in rows:
      p=next(r for r in persona if r["model_key"]==m and r["order_condition"]=="original_order" and r["question_id"]=="MEAN"); od=next(r for r in orderfx if r["model_key"]==m and r["persona_condition"]=="demographics_only" and r["question_id"]=="MEAN"); op=next(r for r in orderfx if r["model_key"]==m and r["persona_condition"]=="demographics_plus_persona" and r["question_id"]=="MEAN")
      axes[0].bar(DISPLAY[m].split()[0],p["same_uuid_switch_rate"]*100); axes[1].bar(DISPLAY[m].split()[0],mean([od["same_uuid_switch_rate"],op["same_uuid_switch_rate"]])*100)
    axes[0].set_title("Persona-condition switch rate"); axes[1].set_title("Option-order switch rate"); axes[0].set_ylabel("Same-UUID switch rate (%)")
    for ax in axes: ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); savefig(fig,"Figure_HC4_persona_order_switch_rates")

    # HC5 Motif run1/run2 cell TVDs.
    cell_data=read_csv(full.FULL/"analysis/motif_run1_vs_run2_population_cells.csv"); cell_map={}
    for r in cell_data: cell_map[(r["question_id"],r["persona_condition"],r["order_condition"])]=float(r["cell_population_tvd"])
    fig,ax=plt.subplots(figsize=(9,4.8)); labs=[]; vals=[]
    for q in QUESTION_IDS:
      labs.append(q); vals.append(mean(v for (qq,_,_),v in cell_map.items() if qq==q))
    ax.bar(labs,vals); ax.set_ylabel("Run1–run2 population TVD"); ax.set_title("Motif aggregate stability across full-run replicates"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); savefig(fig,"Figure_HC5_motif_population_stability")

    # HC6 interface comparison.
    fig,ax=plt.subplots(figsize=(9,4.8)); x=np.arange(3); width=.18
    for j,c in enumerate(CONDITIONS):
      ss=[next(r for r in interface if r["model_key"]==m and r["persona_condition"]==c) for m in ("qwen","exaone","gemma")]
      ax.bar(x+(2*j-1.5)*width,[r["probability_based_original_order_mean_tvd"] for r in ss],width,label=f"Probability · {'demo' if j==0 else '+persona'}")
      ax.bar(x+(2*j-.5)*width,[r["hard_choice_original_order_mean_tvd"] for r in ss],width,label=f"Hard choice · {'demo' if j==0 else '+persona'}")
    ax.set_xticks(x,[DISPLAY[m].split()[0] for m in ("qwen","exaone","gemma")]); ax.set_ylabel("Mean TVD to human benchmark"); ax.set_title("Measurement-interface dependence",pad=42); ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.08)); ax.grid(axis="y",alpha=.2); fig.tight_layout(); savefig(fig,"Figure_HC6_probability_vs_hardchoice")

    # Key answers and report.
    original={f"{m}_{c}":next(r["mean_q2_q5_tvd"] for r in summaries if r["model_key"]==m and r["persona_condition"]==c) for m in rows for c in CONDITIONS}
    persona_mean={m:next(r for r in persona if r["model_key"]==m and r["order_condition"]=="original_order" and r["question_id"]=="MEAN") for m in rows}
    order_mean={m:mean(next(r for r in orderfx if r["model_key"]==m and r["persona_condition"]==c and r["question_id"]=="MEAN")["same_uuid_switch_rate"] for c in CONDITIONS) for m in rows}
    dep_mean={r["model_pair"]:{"population_tvd":r["population_tvd"],"same_uuid_disagreement_rate":r["same_uuid_disagreement_rate"],"persona_condition":r["persona_condition"]} for r in orig}
    closest=min(original.items(),key=lambda x:x[1])
    key_results={"timestamp":now(),"benchmark_normalization_rule":"verbatim percentages divided by 100; no renormalization","original_order_mean_human_tvd":original,"closest_model_condition":{"key":closest[0],"mean_tvd":closest[1]},"persona_effect_original_order":{m:{"population_tvd":persona_mean[m]["population_tvd"],"switch_rate":persona_mean[m]["same_uuid_switch_rate"],"human_tvd_delta_persona_minus_demo":persona_mean[m]["human_tvd_delta_persona_minus_demo"]} for m in rows},"order_switch_rate_mean_across_persona_conditions":order_mean,"model_dependence_original_order":dep_mean,"motif_stochasticity":{"paired_valid":len(both),"different":diff,"disagreement_rate":diff/len(both),"population_cell_mean_tvd":st["population_tvd_mean_across_16_cells"],"question_tvd":st["population_tvd_by_question_mean_across_conditions_orders"],"max_category_pp":st["maximum_category_percentage_point_difference"]},"bootstrap":boots,"format_integrity":integ,"human_benchmark_sha256":sha256_file(BENCHMARK),"human_benchmark_accessed_for_analysis":True,"new_inference_executed":False}
    KEY_RESULTS.write_text(json.dumps(key_results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    report=["# Supplementary Hard-Choice Model Comparison: Final Analysis","","This analysis preserves Motif Run 1 as the representative supplementary result. Motif Run 2 is used only for provider/backend stochasticity sensitivity. No inference, prompt change, parser change, calibration, or manuscript modification was performed.","","## Completeness","",md_table(["Run","Rows","Unique","Duplicates","Semantic valid","Format compliant","Failures"],[[m,f"{v['observations']:,}",f"{v['unique_keys']:,}",v['duplicate_keys'],pct(v['semantic_valid_rate']),pct(v['format_compliant_rate']),v['api_model_failures']] for m,v in integ.items()]),"","## Original-order human comparison","",md_table(["Model","Condition","Mean TVD","95% bootstrap CI"],[[r['model'],r['persona_condition'],f"{r['mean_q2_q5_tvd']:.4f}",f"[{r['ci_lower_2_5']:.4f}, {r['ci_upper_97_5']:.4f}]"] for r in summaries]),"",f"The smallest pre-specified model-condition mean TVD was {closest[1]:.4f} ({closest[0]}). This comparison does not identify a model as representative of Korean public opinion.","","## Answers to the six analysis questions","",f"- A. Motif's original-order mean TVDs are {original['motif_demographics_only']:.4f} (demographics only) and {original['motif_demographics_plus_persona']:.4f} (+persona); their relative position is reported descriptively, not as model selection.","- B. Persona effects were not uniformly beneficial: the sign of TVD_persona − TVD_demo varies by model and is reported in HC4.","- C. Option-order sensitivity remained present in all four models; same-UUID switch rates are reported in HC5.","- D. Pairwise model dependence remained measurable under the common hard-choice interface (HC3).",f"- E. Motif individual disagreement was {pct(diff/len(both))}, whereas the mean aggregate run1–run2 cell TVD was {st['population_tvd_mean_across_16_cells']:.5f}; aggregate distributions were substantially more stable than individual categorical responses.","- F. HC8 shows sizeable measurement-interface differences for the same three local models. These are different estimands and are not described as improvement or deterioration.","","## Guardrails and limitations","","- The empirical human survey is a benchmark, not population truth; its reported percentages are rounded and human raw microdata are unavailable.","- Original-order human comparison is primary; counterbalanced estimates are secondary order-sensitivity diagnostics.","- Hard categorical choices and normalized next-token choice scores are different measurement interfaces.","- Motif exhibits measurable provider/backend categorical stochasticity despite stable aggregate distributions.","- Semantic-invalid observations were retained and not retried or imputed; human comparisons use semantic-valid denominators.","- This supplementary experiment uses one Korean military-service-policy survey, a synthetic persona dataset, quantized local models, and one API model. Results should not be generalized beyond the tested configuration.","- Human 18–29 is compared with synthetic 19–29 in subgroup analysis.","","## Files","","Tables HC1–HC9 and Figures HC1–HC6 are stored under `analysis_final/`. The existing probability-based primary results and manuscript remain unchanged."]
    REPORT.write_text("\n".join(report)+"\n",encoding="utf-8")

    # Protection, source, and output manifest.
    preserved=json.loads((full.AUDITS/"primary_preservation_before.json").read_text(encoding="utf-8"))["files"]; current=prior.snapshot(prior.primary_files()); modified=sorted(k for k in preserved if preserved.get(k)!=current.get(k))
    secret=os.environ.get("MORPH_API_KEY","").encode(); hits=0
    for p in BASE.rglob("*"):
      if p.is_file(): hits+=int(bool(secret) and secret in p.read_bytes())
    # CSV QA: all nonempty and critical row counts reconcile.
    table_files=sorted(TABLES.glob("*.csv")); figure_files=sorted([*FIGURES.glob("*.png"),*FIGURES.glob("*.pdf")])
    qa={"HC1_rows":len(read_csv(TABLES/"HC1_four_model_human_original_order.csv")),"HC2_rows":len(read_csv(TABLES/"HC2_four_model_category_distributions.csv")),"HC3_rows":len(read_csv(TABLES/"HC3_model_dependence.csv")),"HC9_rows":len(read_csv(TABLES/"HC9_subgroup_external_validation.csv")),"all_tables_nonempty":all(p.stat().st_size>0 for p in table_files),"all_figures_nonempty":all(p.stat().st_size>0 for p in figure_files)}
    (AUDITS/"completeness_and_protection_audit.json").write_text(json.dumps({"integrity":integ,"primary_files_modified":modified,"api_key_exposure_file_hits":hits,"new_inference_executed":False,"manuscript_modified":False,"qa":qa},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    outputs=[p for p in BASE.rglob("*") if p.is_file() and p!=MANIFEST]
    manifest={"timestamp":now(),"analysis":"Supplementary hard-choice model comparison final analysis","protocol_sha256":sha256_file(full.PROTOCOL),"locked_sample_sha256":sha256_file(SAMPLE_PATH),"benchmark_sha256":sha256_file(BENCHMARK),"probability_primary_source_sha256":sha256_file(PROBABILITY),"raw_input_sha256":{str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p) for p in [*PATHS.values(),RUN2]},"analysis_script_sha256":sha256_file(Path(__file__)),"outputs_sha256":{str(p.relative_to(BASE)).replace('\\','/'):sha256_file(p) for p in outputs},"bootstrap":{"replicates":1000,"seed":2026,"unit":"persona"},"primary_files_modified":modified,"api_key_exposure":hits,"new_inference":False,"manuscript_modified":False}
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","key_results":key_results,"qa":qa,"primary_modified":modified,"api_key_exposure":hits},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
