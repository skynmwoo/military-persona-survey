# Protocol Amendment 02: Motif Non-Thinking Mode

Experiment: **Supplementary hard-choice model comparison**  
Status: prospectively frozen before the non-thinking 10-person technical pilot.

The earlier Motif thinking-mode pilot showed that internal reasoning consumed substantial completion budget, producing three 1,024-token truncations, three `content=None` responses, and one repeated-input categorical inconsistency. Those artifacts and their FAIL reports remain unchanged.

A subsequent API capability test used the identical frozen persona, Q2 wording, system prompt, hard-choice instruction, visible A–E interface, semantic mapping, exact parser, temperature, top-p, and seed. Request-level `chat_template_kwargs.enable_thinking=false` was accepted and effective. Motif returned exact categorical content without `reasoning` or `reasoning_content`, used two completion tokens, and ended with `finish_reason=stop`. Twenty identical Motif requests produced the same exact label 20/20 times; a separate four-model ten-repeat audit produced one stable label per model with 100% exact-choice consistency and validity.

The non-thinking decision was made without importing, reading, or comparing the human benchmark. It was not selected using response semantics or benchmark fit.

For this amendment:

- Motif inference is frozen with `chat_template_kwargs: {enable_thinking: false}`.
- All four models receive a maximum completion allowance of 32 tokens. This is a ceiling; normal single-label responses stop earlier.
- Local models retain the existing supplementary hard-choice greedy configuration.
- Q2/Q3/Q5 retain visible labels A–E; Q4 retains A–D.
- Prompt wording, question and option wording, persona formatting, original and counterbalanced assignments, semantic reverse mapping, and the exact parser remain unchanged.
- No retry, repair, imputation, parser relaxation, or human-benchmark analysis is permitted.

This amendment applies only to the supplementary hard categorical-choice experiment. It does not alter or reclassify the probability-based Qwen/EXAONE/Gemma locked validation.
