# Military persona survey: partial reproducibility package

This repository is a research companion for a study comparing LLM-generated
survey-response distributions with an empirical South Korean survey on
military-service policy. It is a **partial reproducibility package**, not a
turn-key release of the complete local research environment.

## What this repository provides

- the locked analysis plan for the probability-based Primary analysis;
- the frozen protocol for the separate hard-choice model comparison;
- the fixed 1,000-persona validation sample and a 100-person smoke-test sample;
- the prompts, option-order assignment logic, inference scripts, parsers, and
  analysis scripts used in the study;
- analysis-ready model outputs needed to reproduce the reported summary tables;
- the transcribed public human-survey benchmark with source metadata;
- final analysis tables used to verify the reported results.

The Primary analysis used Qwen3-8B, EXAONE-3.5-7.8B, and Gemma-3-12B with
normalized next-token choice scores. The hard-choice comparison was a separate
post-lock analysis and additionally included Motif 3. It must not be interpreted
as a retrospective four-model Primary analysis.

## What is not included

- model weight files;
- the complete one-million-row Nemotron-Personas-Korea dataset;
- the full local `llama.cpp` build;
- server logs, temporary files, credentials, or API keys;
- unpublished manuscript working files;
- any automatic rerun of the original inference experiment.

The supplied scripts document the actual experimental implementation. Running
the inference scripts requires separately obtained model files, the recorded
`llama.cpp` build, sufficient hardware, and—for Motif 3—authorized API access.
The repository does not initiate inference during installation or verification.

## Repository map

```text
analysis/reference_tvd/                TVD reference-point analysis
data/derived/                          fixed validation and smoke-test samples
outputs/validation_1000_locked/        Primary plan, benchmark, outputs, and tables
outputs/hard_choice_extension/         hard-choice protocol, outputs, and tables
scripts/                               sampling, inference, and Primary analysis
scripts/hard_choice_extension/         hard-choice inference and analysis
```

## Quick verification without inference

Create a Python environment and verify the released headline results:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/verify_released_results.py
```

To regenerate the Primary analysis tables from the released probability outputs:

```bash
python scripts/analyze_validation_1000_locked.py
```

This reads the fixed sample, the public benchmark transcription, and the three
released JSONL probability-output files. It does not contact a model server.

The hard-choice analysis code is also included. The representative Motif result
is run 1. A second Motif execution is retained only because the preserved final
analysis script used it for an audit-only repeatability check; it is not a
replacement for run 1 and is not part of the conference manuscript's central
result.

## Partial inference implementation

The inference scripts preserve the original project-relative paths and recorded
model hashes. A local reproduction therefore requires the model files and
`llama.cpp` executable to be placed at the paths declared in:

- `scripts/run_validation_1000_locked.py`
- `scripts/hard_choice_extension/hard_choice_common.py`

The 100-person sample can be used to inspect the smoke-test workflow before any
full-scale execution. Do not run the full validation merely to inspect this
repository: the released outputs are sufficient for analysis verification.

## Data provenance

The persona samples were derived from
[NVIDIA Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea),
version 1.0, licensed under CC BY 4.0. The samples remain synthetic and are
provided with source attribution in `DATA_ATTRIBUTION.md`.

The empirical benchmark was transcribed from the Reform Institute's public
military-service policy survey (publication 2026-16, 1,006 respondents). Source
metadata and the transcription rules are stored with the benchmark JSON. The
human survey is treated as an empirical comparison benchmark, not error-free
population truth.

## Interpretation limits

Matching sex, age, and regional margins does not establish population
representativeness or policy-attitude validity. The released results are limited
to the tested persona sample, questions, models, prompts, option orders, and
measurement interfaces. No benchmark-driven calibration or post-hoc model
selection was used to improve agreement with the human survey.

## Availability status

This repository is initially private for author review. Before citing it as
publicly available, make the repository public or archive a release in a
persistent repository and replace the provisional wording in
`DATA_AND_CODE_AVAILABILITY.md`.
