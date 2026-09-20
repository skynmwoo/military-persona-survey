# Supplementary Hard-Choice Model Comparison: Frozen Protocol v1

Status: **FROZEN BEFORE FULL-RUN INFERENCE**  
Date: 2026-08-27 (Asia/Seoul)

This is a post-lock supplementary experiment. It does not modify or rename the existing three-model probability-based locked validation. After this document is hashed, prompts, parsing, model configuration, outcome handling, and analysis rules will not be changed in response to results.

## Models and inference provenance

| Model | Exact artifact/API identifier | SHA-256 / provenance |
|---|---|---|
| Qwen3-8B | `models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| EXAONE-3.5-7.8B-Instruct | `models/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf` | `2f4aff8d555b82500d4de2723810586e83e358003388fdd08bd52e6091e6ddd1` |
| Gemma-3-12B IT | `models/google/gemma-3-12b-it-qat-q4_0-gguf/gemma-3-12b-it-q4_0.gguf` | `dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd` |
| Motif 3 | MorphFactory CBT API, `motif/motif-3` | provider-returned model ID must match; no local artifact hash |

Local inference uses llama.cpp build b10514, commit `7221e24f5`, the embedded GGUF chat templates, RTX 3060 12GB GPU offload, temperature 0, top-p 1, top-k 1, seed 2026, cache-prompt false, reasoning off, and maximum completion allowance 32. Motif uses temperature 0, top-p 1, seed 2026 where accepted, `chat_template_kwargs.enable_thinking=false`, and maximum completion allowance 32. Provider and local decoding implementations are not claimed to be equivalent.

## Locked sample

The unchanged sample is `data/derived/validation_1000_locked_seed42.parquet`, N=1,000, SHA-256 `6650945363b9cd2d7ca702ec98e900aa16cfada714642af3521a2c0e895a27f7`. No resampling, deletion, outcome-based exclusion, or post-hoc weighting is permitted.

## Conditions and prompts

The frozen system instruction and profile formatting are those in `scripts/hard_choice_extension/hard_choice_common.py`. `demographics_only` includes sex, age, age group, region/province, marital status, education level, and occupation. `demographics_plus_persona` adds only the original `persona` field. Each persona-condition-question request is independent.

The policy items are:

- Q2: “선택적 모병제는 현행 징병제를 유지하면서, 인공지능과 드론 등 전문 기술 분야에 한해 보수와 혜택을 강화한 기술집약형 부사관을 도입하는 징병제와 모병제의 혼합 방식입니다. 선생님께서는 선택적 모병제 도입에 대해 어떻게 생각하십니까?” Options: 매우 찬성한다; 찬성하는 편이다; 반대하는 편이다; 매우 반대한다; 잘 모르겠다.
- Q3: “장교와 부사관 지원이 정원에 미치지 못하는 상황이 이어지고 있습니다. 선생님께서는 선택적 모병제가 병력 충원 문제 해결에 도움이 될 것으로 보십니까?” Options: 크게 도움될 것이다; 어느 정도 도움될 것이다; 별로 도움되지 않을 것이다; 전혀 도움되지 않을 것이다; 잘 모르겠다.
- Q4: “앞으로 우리나라 병역제도가 어떤 방향으로 가야 한다고 생각하십니까?” Options: 현행 징병제 유지; 징병제와 모병제 병행; 전면 모병제 전환; 잘 모르겠다.
- Q5: “병역 자원이 줄어드는 상황에서 여성도 징병 대상에 포함하자는 주장이 있습니다. 선생님께서는 여성 징병제에 대해 어떻게 생각하십니까?” Options: 매우 찬성한다; 찬성하는 편이다; 반대하는 편이다; 매우 반대한다; 잘 모르겠다.

Q2/Q3/Q5 display position labels A–E; Q4 uses A–D. The existing hard-choice instruction is unchanged. Labels attach to display positions. Original order uses semantic categories in survey order. Counterbalanced order uses the frozen rule: sort all locked UUIDs by `SHA256('2026|question|uuid')`; assign zero-based rank modulo the number of options as the cyclic rotation. The displayed label is reverse-mapped to its semantic category before analysis, and every mapping must be bijective.

## Parser v2 and outcome handling

Outer whitespace is stripped. An exact single valid label is `semantic_valid=true` and `format_compliant=true`. A single unambiguous leading valid label followed by punctuation, whitespace, explanation, or leading parentheses is `semantic_valid=true` and `format_compliant=false`. No valid label, a conflicting categorical label, an ambiguous answer, or an out-of-range label is semantically invalid. Q4 excludes E. Parser v2 is identical across models. Invalid responses are retained and reported, never repaired, retried, calibrated, or imputed. Response distributions use semantic-valid observations as the denominator, while invalid and exact-format rates are reported separately. HTTP/API failure and content invalidity are distinct. Retry count is frozen at zero.

## Experimental runs

Run 1 contains 1,000 personas × 2 persona conditions × 2 order conditions × Q2–Q5 = 16,000 observations per model and 64,000 overall. After Run 1 completes, Motif is repeated once over the identical 16,000 requests as `Motif full-run replicate 2`. Motif Run 1 remains the representative supplementary estimate; replicate 2 is a provider/backend stochasticity sensitivity analysis and cannot replace Run 1 based on benchmark fit.

## Metrics and analysis

The human benchmark is the verified Reform Institute No. 2026-16 benchmark at `outputs/validation_1000_locked/benchmark/human_benchmark_subgroups_20260818.json`, SHA-256 `4747341985ca61037529563a6b3447b400e2defd3191b8751376603c391830ce`. It is evaluation-only and is never included in prompts.

Primary human-comparison metric is question-level total variation distance, with mean Q2–Q5 TVD as the summary. Secondary metrics are category percentage-point error, MAE, and Jensen–Shannon divergence. Analyses include six pairwise population TVDs and same-UUID categorical disagreement rates; paired persona-condition switch rates and population TVDs; paired semantic original-versus-counterbalanced switch rates and population TVDs; semantic validity and exact-format compliance; and sex, frozen age-group, and seven-region subgroup TVD only where verified human subgroup values exist.

Motif Run 1 versus replicate 2 is assessed using same-request semantic disagreement, question/persona/order stratification, population TVD, question-level TVD, and category percentage-point differences.

The primary bootstrap resamples the 1,000 UUIDs with replacement as paired persona units, retaining all experimental cells for each selected UUID. It uses 1,000 replicates, NumPy seed 2026, and percentile 95% intervals for original-order mean Q2–Q5 human TVD by model and persona condition. Motif replicate 2 is excluded from primary bootstrap intervals.

The existing Qwen/EXAONE/Gemma probability-based original-order human TVDs are compared descriptively with hard-choice TVDs as measurement-interface sensitivity. Motif probability-based values remain NA. The estimands are not pooled or treated as interchangeable.

## Stability limitation and safeguards

The pre-freeze stability audit found Motif semantic disagreement in 8/50 configurations (16%), yielding **GO WITH STOCHASTICITY LIMITATION**. This provider/backend nondeterminism must be reported. There is no prompt tuning, benchmark-fit tuning, response calibration, model weighting, result-based exclusion, favorable replicate selection, or automatic manuscript modification. The API key is read only from `MORPH_API_KEY` and must never be stored or printed.
