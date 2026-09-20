# Validation 1,000 locked report

This is a locked validation, not an estimate of Korean public opinion. Human benchmarks were used only after inference for evaluation.

## Execution

| Model | Rows | Wall min | Peak VRAM MiB |
| --- | ---: | ---: | ---: |
| Qwen3-8B-Q4_K_M | 16,000 | 62.6 | 6,397 |
| EXAONE-3.5-7.8B-Instruct-Q4_K_M | 16,000 | 58.3 | 6,201 |
| Gemma-3-12B-QAT-Q4_0 | 16,000 | 117.3 | 10,011 |

## Primary mean TVD (Q2-Q5)

| Model | demographics only | plus persona |
| --- | ---: | ---: |
| Qwen3-8B-Q4_K_M | 0.5263 | 0.5774 |
| EXAONE-3.5-7.8B-Instruct-Q4_K_M | 0.5410 | 0.5747 |
| Gemma-3-12B-QAT-Q4_0 | 0.5322 | 0.5619 |

## Design and limitations

- Primary: original-order probability aggregation. Counterbalanced and greedy results remain secondary.
- The human 18-29 group is compared with synthetic 19-29; Nemotron has no 18-year-olds.
- Human subgroup percentages are rounded and raw microdata are unavailable; no joint subgroup benchmark or human subgroup uncertainty was fabricated.
- The three models are compared as model-dependence diagnostics; closeness does not establish population representativeness.
