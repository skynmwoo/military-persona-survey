# Protocol Amendment 01: Completion-Budget Allowance

Experiment: **Supplementary hard-choice model comparison**  
Status: adopted before the 10-person technical pilot v2.

## Evidence and rationale

The initial four-token Motif preflight returned HTTP 200 and the expected model ID but no final content. A later completion-budget test preserved the frozen prompt wording, visible A–E interface, exact parser, semantic mapping, persona, question, and decoding controls. At 256 maximum completion tokens, Motif produced a nonempty internal reasoning field, used all 256 completion tokens, returned `content=None`, and ended with `finish_reason=length`. At 512 maximum completion tokens, it produced an exact content response `B`, used 480 completion tokens, and ended with `finish_reason=stop`; the unchanged exact A–E parser accepted the response.

This evidence supports treating the earlier empty content as backend-specific reasoning-budget truncation rather than failure to understand the visible response interface. The semantic response and human-benchmark fit played no role in this decision. No human benchmark was imported, read, or compared.

## Amended allowance

Beginning with the 10-person technical pilot v2, all four models receive a **maximum completion allowance of 1,024 tokens**. This is a ceiling, not a requested generation length. A model that emits its categorical answer and stops earlier retains that normal stop. Local models use the llama.cpp `max_tokens` parameter; Motif uses the previously verified MorphFactory `max_completion_tokens` parameter.

The visible interface remains identical across models: A–E for Q2, Q3, and Q5, and A–D for Q4. The frozen prompt wording, exact parser, semantic reverse mapping, persona conditions, original order, and deterministic counterbalanced assignment are unchanged.

## Diagnostics and invalid handling

Every observation records the maximum allowance, actual completion-token usage when exposed, finish reason, raw content, parsed label, validity, and latency. Motif additionally records returned model ID, prompt/completion/total tokens, reasoning-field presence, and reasoning character count; reasoning text is not stored or analyzed substantively.

No content-based retry, prompt retry, repair, parser relaxation, or automatic increase to 2,048 tokens is allowed. A Motif response with `finish_reason=length`, or with `content=None` after exhausting the 1,024-token allowance, is retained as INVALID/TRUNCATED. HTTP/API failures remain distinct from HTTP-200 content invalidity.

The probability-based Qwen/EXAONE/Gemma locked validation remains the primary experiment. This amendment applies only to the separate supplementary hard categorical-choice interface and does not create a four-model locked validation.
