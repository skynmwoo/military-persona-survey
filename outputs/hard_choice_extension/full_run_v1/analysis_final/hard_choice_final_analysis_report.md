# Supplementary Hard-Choice Model Comparison: Final Analysis

This analysis preserves Motif Run 1 as the representative supplementary result. Motif Run 2 is used only for provider/backend stochasticity sensitivity. No inference, prompt change, parser change, calibration, or manuscript modification was performed.

## Completeness

| Run | Rows | Unique | Duplicates | Semantic valid | Format compliant | Failures |
|---|---|---|---|---|---|---|
| qwen | 16,000 | 16,000 | 0 | 100.00% | 100.00% | 0 |
| exaone | 16,000 | 16,000 | 0 | 100.00% | 100.00% | 0 |
| gemma | 16,000 | 16,000 | 0 | 100.00% | 100.00% | 0 |
| motif | 16,000 | 16,000 | 0 | 99.97% | 98.66% | 0 |
| motif_run2 | 16,000 | 16,000 | 0 | 99.96% | 98.69% | 0 |

## Original-order human comparison

| Model | Condition | Mean TVD | 95% bootstrap CI |
|---|---|---|---|
| Qwen3-8B | demographics_only | 0.4805 | [0.4705, 0.4902] |
| Qwen3-8B | demographics_plus_persona | 0.4642 | [0.4607, 0.4677] |
| EXAONE-3.5-7.8B-Instruct | demographics_only | 0.6055 | [0.6030, 0.6130] |
| EXAONE-3.5-7.8B-Instruct | demographics_plus_persona | 0.6118 | [0.6045, 0.6198] |
| Gemma-3-12B | demographics_only | 0.6192 | [0.6113, 0.6268] |
| Gemma-3-12B | demographics_plus_persona | 0.6232 | [0.6155, 0.6298] |
| Motif 3 | demographics_only | 0.5993 | [0.5917, 0.6073] |
| Motif 3 | demographics_plus_persona | 0.5500 | [0.5422, 0.5577] |

The smallest pre-specified model-condition mean TVD was 0.4642 (qwen_demographics_plus_persona). This comparison does not identify a model as representative of Korean public opinion.

## Answers to the six analysis questions

- A. Motif's original-order mean TVDs are 0.5993 (demographics only) and 0.5500 (+persona); their relative position is reported descriptively, not as model selection.
- B. Persona effects were not uniformly beneficial: the sign of TVD_persona − TVD_demo varies by model and is reported in HC4.
- C. Option-order sensitivity remained present in all four models; same-UUID switch rates are reported in HC5.
- D. Pairwise model dependence remained measurable under the common hard-choice interface (HC3).
- E. Motif individual disagreement was 9.06%, whereas the mean aggregate run1–run2 cell TVD was 0.00769; aggregate distributions were substantially more stable than individual categorical responses.
- F. HC8 shows sizeable measurement-interface differences for the same three local models. These are different estimands and are not described as improvement or deterioration.

## Guardrails and limitations

- The empirical human survey is a benchmark, not population truth; its reported percentages are rounded and human raw microdata are unavailable.
- Original-order human comparison is primary; counterbalanced estimates are secondary order-sensitivity diagnostics.
- Hard categorical choices and normalized next-token choice scores are different measurement interfaces.
- Motif exhibits measurable provider/backend categorical stochasticity despite stable aggregate distributions.
- Semantic-invalid observations were retained and not retried or imputed; human comparisons use semantic-valid denominators.
- This supplementary experiment uses one Korean military-service-policy survey, a synthetic persona dataset, quantized local models, and one API model. Results should not be generalized beyond the tested configuration.
- Human 18–29 is compared with synthetic 19–29 in subgroup analysis.

## Files

Tables HC1–HC9 and Figures HC1–HC6 are stored under `analysis_final/`. The existing probability-based primary results and manuscript remain unchanged.
