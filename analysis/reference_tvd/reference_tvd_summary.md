# Reference TVD summary

**Verification status:** ANALYZED — deterministic derived analysis of preserved results; no LLM inference was run.

## Inputs and method

- Human benchmark and subgroup transcription: `outputs/validation_1000_locked/benchmark/human_benchmark_subgroups_20260818.json` (SHA-256 `4747341985ca61037529563a6b3447b400e2defd3191b8751376603c391830ce`)
- Official survey report: `outputs/validation_1000_locked/benchmark/source/reform_institute_poll_2026-16_report.pdf` (SHA-256 `a578b1bee3bed349814f1f027adf4ea3f2c3b31f13baa166069c15e561b704d6`; subgroup cross-tabs on PDF pages 10–13 were visually checked during this audit)
- Existing Primary and corrected hard-choice means: `outputs/consistency_audit_20260904/tables/HC8_interface_comparison_A_vs_B_v1.csv` (SHA-256 `abbad77aba5dcf84c63f85e0d2eafdfbc3735bf43c7901d5242d231e0b7530c5`)
- Reused functions: `norm` and `tvd` imported from `scripts/analyze_validation_1000_locked.py` (SHA-256 `a8ed16fb7d7129470365612d59fd3b04a7ff2a89ed9d9632eab647c2a34ce8da`)
- Normalization: every published percentage vector was normalized to sum to one with the imported Primary `norm` function. All distances were evaluated with the imported Primary `tvd` function.
- Question summary: Q2–Q5 were weighted equally. Q2, Q3, and Q5 have five options; Q4 has four.

## Question-level reference points

| Reference | Q2 | Q3 | Q4 | Q5 | Q2–Q5 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform vs human | 0.143000 | 0.182783 | 0.269770 | 0.198000 | 0.198388 |
| 100% single-option concentration vs human (min–max) | 0.721000–0.893000 | 0.703704–0.926927 | 0.584585–0.893894 | 0.643000–0.936000 | 0.663072–0.912455 |
| Human male vs female | 0.111330 | 0.132434 | 0.120000 | 0.158319 | 0.130521 |
| Human youngest vs oldest age group | 0.223813 | 0.262000 | 0.283000 | 0.356225 | 0.281259 |
| Human maximum region pair | 0.212754 | 0.186776 | 0.165834 | 0.167513 | 0.183220 |

For the single-option reference, the mean interval is the mean of the four question-specific minima through the mean of the four question-specific maxima; it is not a confidence interval.

## Region pairs producing the question-specific maxima

| Question | Region pair | TVD |
| --- | --- | ---: |
| Q2 | 광주·전라 vs 강원·제주 | 0.212754 |
| Q3 | 서울 vs 광주·전라 | 0.186776 |
| Q4 | 인천·경기 vs 대구·경북 | 0.165834 |
| Q5 | 부산·울산·경남 vs 강원·제주 | 0.167513 |

## Final comparison table

| Reference or preserved result | Q2–Q5 mean TVD or range | Scope |
| --- | ---: | --- |
| Uniform distribution vs human | 0.198388 | equal-weight mean of Q2–Q5 |
| 100% single-option concentration vs human | 0.663072–0.912455 | mean of question minima – mean of question maxima |
| Human male vs female | 0.130521 | equal-weight mean of Q2–Q5 |
| Human youngest vs oldest age group | 0.281259 | equal-weight mean of Q2–Q5 |
| Human maximum region pair | 0.183220 | mean of four question-specific maxima |
| Existing Primary synthetic–human | 0.526303–0.577443 | range across 3 models × 2 persona conditions |
| Existing hard-choice synthetic–human | 0.464186–0.623314 | range across 4 models × 2 persona conditions; Motif run1 |

## Human subgroup availability

The official report publishes weighted row percentages for sex, age group, and region for every item Q2–Q5. Accordingly, male–female, youngest–oldest, and maximum pairwise region TVDs were calculable without imputation. Human microdata and joint subgroup distributions were not used or reconstructed.

## Interpretation boundaries

- These values are descriptive scale references, not prespecified thresholds of acceptable validity and not tests of statistical equivalence.
- The human subgroup contrasts are the most substantively comparable within-survey reference available here, but they use rounded published percentages and do not include sampling, nonresponse, or weighting uncertainty.
- The maximum-region value is selected from all published region pairs separately for each question, so it is an upper descriptive reference by construction; the maximizing pair may change across questions.
- The synthetic–human ranges are preserved results, not recalculated model estimates. Primary and hard-choice are different measurement configurations and should not be treated as the same estimand.
- No causal claim, subgroup-to-individual inference, population-truth claim, or generalization beyond these four items and this survey is licensed by this calculation.
- All Q2–Q5 items and the prespecified sex, age-extreme, and region-maximum comparisons were retained; no favorable subset was selected.

## Reproducibility

Run `reference_tvd.py` from any working directory. It writes only `analysis/reference_tvd/` and does not modify frozen Primary or hard-choice artifacts.
