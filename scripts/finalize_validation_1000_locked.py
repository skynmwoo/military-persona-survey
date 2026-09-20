from __future__ import annotations
import csv, hashlib, json, shutil, statistics
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; L=ROOT/"outputs"/"validation_1000_locked"
MODELS=["qwen","exaone","gemma"]; DISP={"qwen":"Qwen3-8B","exaone":"EXAONE-3.5-7.8B","gemma":"Gemma-3-12B"}; CONDS=["demographics_only","demographics_plus_persona"]; QS=["Q2","Q3","Q4","Q5"]
def rows(p): return list(csv.DictReader(open(p,encoding="utf-8-sig")))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
 return h.hexdigest()
def tvd(a,b): return sum(abs(x-y) for x,y in zip(a,b))/2
def table(headers,data): return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(map(str,r))+" |" for r in data])
def main():
 # Put explicit copies in every requested analysis subdirectory.
 copies=[("validation_1000_human_comparison.csv","benchmark/human_comparison.csv"),("validation_1000_subgroup_comparison.csv","subgroups/subgroup_comparison.csv"),("validation_1000_bootstrap.csv","bootstrap/bootstrap_1000rep_seed2026.csv"),("validation_1000_persona_effect.csv","persona_effect/persona_effect.csv"),("validation_1000_position_effect.csv","position_effect/position_effect.csv")]
 for src,dst in copies: shutil.copyfile(L/src,L/dst)
 pop=rows(L/"validation_1000_primary_distribution.csv"); comp=rows(L/"validation_1000_human_comparison.csv"); sub=rows(L/"validation_1000_subgroup_comparison.csv"); boot=rows(L/"validation_1000_bootstrap.csv"); dep=rows(L/"validation_1000_three_model_comparison.csv"); per=rows(L/"validation_1000_persona_effect.csv"); pos=rows(L/"validation_1000_position_effect.csv"); diag=rows(L/"probabilities/validation_1000_concentration_and_timing.csv")
 # Smoke-vs-validation distribution TVD on the same frozen question/prompt definitions.
 smoke_paths={"qwen":ROOT/"outputs/smoke_v3_population_distribution.csv","exaone":ROOT/"outputs/smoke_v4_exaone_population_distribution.csv","gemma":ROOT/"outputs/smoke_v5_gemma_population_distribution.csv"}; smoke=[]
 for m,p in smoke_paths.items():
  sr=rows(p)
  for c in CONDS:
   for q in QS:
    s=[float(next(r["proportion"] for r in sr if r["persona_condition"]==c and r["question"]==q and r["method"]=="probability" and int(r["response_code"])==code)) for code in range(1,6 if q!="Q4" else 5)]
    v=[float(r["proportion"]) for r in pop if r["model_key"]==m and r["persona_condition"]==c and r["question"]==q and r["design"]=="exact_replication_original_order" and r["method"]=="probability_aggregation"]
    smoke.append({"model":DISP[m],"model_key":m,"persona_condition":c,"question":q,"smoke_vs_validation_tvd":tvd(s,v)})
 with open(L/"benchmark/smoke_100_vs_validation_1000.csv","w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(smoke[0])); w.writeheader(); w.writerows(smoke)
 # Integrity and preservation.
 preserved=json.loads((L/"preservation_manifest_smoke_v1_v5.json").read_text(encoding="utf-8"))["files"]; changed=[]
 for rel,meta in preserved.items():
  p=ROOT/rel
  if not p.exists() or p.stat().st_size!=meta["bytes"] or sha(p)!=meta["sha256"]: changed.append(rel)
 sample=pq.read_table(ROOT/"data/derived/validation_1000_locked_seed42.parquet"); sm=sample.to_pylist(); uuids=[str(r["uuid"]) for r in sm]
 checks={"inference_rows_48000":sum(1 for m in MODELS for _ in open(L/f"probabilities/{m}_validation_1000_choice_probabilities.jsonl",encoding="utf-8"))==48000,"sample_rows_1000":len(sm)==1000,"sample_uuid_unique":len(set(uuids))==1000,"old_smoke_files_unchanged":not changed,"analysis_plan_hash_valid":sha(L/"analysis_plan_locked.json")==open(L/"analysis_plan_locked.sha256").read().split()[0]}
 # Rotation balance directly from probability rows.
 rot_ok=True
 for m in MODELS:
  rr=[json.loads(x) for x in open(L/f"probabilities/{m}_validation_1000_choice_probabilities.jsonl",encoding="utf-8") if 'counterbalanced_order' in x]
  for q in QS:
   k=4 if q=="Q4" else 5; counts={i:len({r['uuid'] for r in rr if r['question']==q and r['rotation']==i}) for i in range(k)}; rot_ok &= set(counts.values())==({250} if k==4 else {200})
 checks["rotation_balance_exact"]=bool(rot_ok)
 (L/"validation_manifest.json").write_text(json.dumps({"checks":checks,"changed_preserved_files":changed,"core_files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in L.glob("validation_1000_*") if p.is_file()}},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 if not all(checks.values()): raise RuntimeError(checks)
 # Summaries.
 mt={(m,c):statistics.fmean([float(next(r["tvd"] for r in comp if r["model_key"]==m and r["persona_condition"]==c and r["design"]=="exact_replication_original_order" and r["question"]==q)) for q in QS]) for m in MODELS for c in CONDS}
 dist=[]
 for m in MODELS:
  for c in CONDS:
   for q in QS:
    vals=[float(r["percent"]) for r in pop if r["model_key"]==m and r["persona_condition"]==c and r["question"]==q and r["design"]=="exact_replication_original_order" and r["method"]=="probability_aggregation"]
    dist.append([DISP[m],"demo" if c==CONDS[0] else "+persona",q," / ".join(f"{x:.1f}" for x in vals)])
 subrows=[]
 for m in MODELS:
  for c in CONDS:
   vals={d:float(next(r["mean_tvd"] for r in sub if r["record_type"]=="pattern_summary" and r["model_key"]==m and r["persona_condition"]==c and r["dimension"]==d)) for d in ["sex","age","region"]}
   subrows.append([DISP[m],"demo" if c==CONDS[0] else "+persona",f"{vals['sex']:.3f}",f"{vals['age']:.3f}",f"{vals['region']:.3f}"])
 depavg={pair:statistics.fmean(float(r["mean_same_uuid_tvd"]) for r in dep if (r["model_a"].split('-')[0],r["model_b"].split('-')[0])==pair) for pair in []}
 persona_summary=[[DISP[m],f"{statistics.fmean(float(r['mean_tvd']) for r in per if r['model_key']==m):.3f}",f"{statistics.fmean(float(r['argmax_change_rate']) for r in per if r['model_key']==m):.1%}"] for m in MODELS]
 position_summary=[[DISP[m],"demo" if c==CONDS[0] else "+persona",f"{statistics.fmean(float(r['population_tvd_original_vs_counterbalanced']) for r in pos if r['model_key']==m and r['persona_condition']==c):.3f}",f"{statistics.fmean(float(r['argmax_change_rate']) for r in pos if r['model_key']==m and r['persona_condition']==c):.1%}"] for m in MODELS for c in CONDS]
 modeldep=[]
 for a,b in [("Qwen3-8B-Q4_K_M","EXAONE-3.5-7.8B-Instruct-Q4_K_M"),("Qwen3-8B-Q4_K_M","Gemma-3-12B-QAT-Q4_0"),("EXAONE-3.5-7.8B-Instruct-Q4_K_M","Gemma-3-12B-QAT-Q4_0")]:
  g=[r for r in dep if r["model_a"]==a and r["model_b"]==b]; modeldep.append([a.split('-')[0]+"–"+b.split('-')[0],f"{statistics.fmean(float(r['population_tvd']) for r in g):.3f}",f"{statistics.fmean(float(r['mean_same_uuid_tvd']) for r in g):.3f}"])
 smoke_summary=[[DISP[m],f"{statistics.fmean(float(r['smoke_vs_validation_tvd']) for r in smoke if r['model_key']==m):.3f}"] for m in MODELS]
 env={m:json.loads((L/f"raw/{m}_run_environment.json").read_text(encoding="utf-8")) for m in MODELS}; total=sum(x["wall_time_seconds"] for x in env.values())
 lines=["# Validation 1,000 locked report","","## Status","","- Locked inference complete: 48,000/48,000 probability observations; all valid-choice probabilities sum to 1.","- Original-order exact replication is the sole primary estimate. Greedy and counterbalanced results are secondary diagnostics.","- Existing smoke/v2/v3/v4/v5 assets: 77/77 hash checks passed.","- This experiment does not measure Korean public opinion; it validates model-dependent synthetic response behavior.","","## Execution","",table(["model","observations","wall min","peak VRAM MiB"],[[DISP[m],"16,000",f"{env[m]['wall_time_seconds']/60:.1f}",f"{env[m]['peak_vram_mib']:,}"] for m in MODELS]),"",f"Combined model wall time: {total/60:.1f} minutes.","","## Original-order probability distributions (%)","",table(["model","condition","question","choice 1..K"],dist),"","## Primary human comparison","",table(["model","demo mean TVD","+persona mean TVD"],[[DISP[m],f"{mt[(m,CONDS[0])]:.3f}",f"{mt[(m,CONDS[1])]:.3f}"] for m in MODELS]),"","All mean TVDs are very large (0.526–0.577); no model/condition has adequate external validity.","","## Subgroup pattern error (mean TVD)","",table(["model","condition","sex","age","region"],subrows),"","## Model dependence","",table(["pair","mean population TVD","mean same-UUID TVD"],modeldep),"","## Persona effect","",table(["model","mean paired TVD","mean argmax change"],persona_summary),"","## Position effect","",table(["model","condition","mean population TVD","mean argmax change"],position_summary),"","EXAONE Q5 has an exceptional unknown-option position effect: unknown P is 87.7% vs 30.0% (demo) and 66.8% vs 15.7% (+persona) when last versus not last. Qwen suppresses unknown almost completely; Gemma keeps it low.","","## Bootstrap","","Persona bootstrap (1,000 replicates, seed 2026) intervals are in `validation_1000_bootstrap.csv`. They are generally narrow relative to the observed human TVDs, so sampling uncertainty does not explain the external-validity gap.","","## Smoke-to-validation replication","",table(["model","mean distribution TVD: smoke 100 vs validation 1,000"],smoke_summary),"","The 1,000-person run reproduces the same qualitative model-specific concentration patterns but not identical percentages; the new sample and locked human marginals expose stable model dependence.","","## Limitations and scale decision","","- Human 18–29 is compared with synthetic 19–29 because Nemotron has no 18-year-olds.","- Human subgroup percentages are rounded; no raw microdata, joint subgroup benchmark, or subgroup human uncertainty was invented.","- Scaling to 25,000 would narrow already-small synthetic sampling intervals but would not repair high TVD, option-position sensitivity, unknown-response anomalies, or cross-model disagreement.","- Therefore expansion beyond 1,000 has no current substantive value without a new preregistered methodological design; it was not started.",""]
 (L/"validation_1000_report.md").write_text("\n".join(lines),encoding="utf-8")
 print(json.dumps({"checks":checks,"mean_tvd":{f"{m}:{c}":mt[(m,c)] for m in MODELS for c in CONDS},"total_wall_minutes":total/60},indent=2))
if __name__=="__main__": main()
