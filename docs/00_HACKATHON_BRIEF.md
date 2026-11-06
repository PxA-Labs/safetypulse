# 00 — Hackathon Brief

Reference copy of the track and problem statements this project is built against, and how
SafetyPulse maps to them.

| Field | Value | Source / status |
|-------|-------|-----------------|
| Event | Lexsi Labs AI Safety & Alignment Hackathon (with Cohere Labs and Hugging Face) | Secondary listing — confirm on official page |
| Dates | 1–21 November 2026 | Secondary listing — confirm exact submission time and timezone |
| Registered track | **Track 2 — AI Red Teaming, Safety and Alignment** | Team registration |
| Sub-track we target | **Track 2b — AI Safety and Correction** | Team decision ([ADR-0001](adr/0001-project-direction.md)) |
| Submission requirement | Chain at least two toolkits and report before/after metrics | Secondary listing — **confirm in official rules** |
| Judging criteria | _TBD — copy from official rules_ | |
| Compute | Hugging Face credits: $49.96 (coverage of GPU compute to be confirmed) | Team billing page, 5 Nov 2026 |

---

## 1. Track 2 toolkits (as provided by organisers)

| Toolkit | Role in track | Official description |
|---------|---------------|----------------------|
| **CuratorKIT** | Build a safety-focused dataset | A data-curation and synthetic-data-generation pipeline for LLM post-training, ingesting raw documents or datasets and producing quality-gated, deduplicated, hallucination-checked training data in trainer-ready formats. (Docs, Paper) |
| **SafeTune** | Audit and repair safety drift | A library for auditing and repairing safety drift in fine-tuned LLMs, covering train-time hardening, weight-space interventions, and inference-time steering, plus built-in tools to locate safety-relevant circuits and neurons. (Docs) |
| **CircuitKIT** | Locate and intervene on the underlying circuits | A mechanistic-interpretability framework for discovering, evaluating, and intervening on circuits inside transformer-based models, including pruning, quantization, knowledge editing, and circuit-restricted fine-tuning. (Docs) |
| **AuditKIT** | Evaluate the hardened model | An evaluation suite with a collection of benchmarks and metrics and diagnosis. (Docs) |

Intended pipeline: **CuratorKIT → SafeTune / CircuitKIT → AuditKIT**.

## 2. Problem statements (verbatim from track page)

### Track 2a — AI Safety and Red Teaming

1. **Red-teaming and robustness evaluation:** Attack a hardened model with jailbreaks, prompt
   injection, encoding tricks, best-of-N sampling and weight-space attacks such as abliteration, and
   report robustness alongside over-refusal.
2. **Adversarial and safety evaluation sets:** Build an adversarial or safety benchmark for a specific
   domain, language or risk area, with clear labels and scoring, and use it to measure how current
   open models hold up. The deliverable can be the dataset and a report.
3. **Multi-turn audits:** Audit a model or agent across long, multi-turn conversations for attacks a
   single prompt would not catch, such as prompt injection hidden in tools or documents, gradual
   jailbreaks and requests for cyber-attack help. Report what broke and in which turn, with
   reproducible transcripts.
4. **Multilingual and code-mixed safety:** Test whether safety holds across languages and
   mixed-language prompts, then close the gaps you find.

### Track 2b — AI Safety and Correction

1. **Safety-preserving fine-tuning:** Fine-tune a model for a useful domain task with a
   safety-hardening method, and show safety holds compared with plain fine-tuning.
2. **Repairing safety drift without retraining:** Take a model whose safety degraded after
   fine-tuning and restore it with weight-space methods (safety patching, merging, projection) while
   keeping task performance.
3. **Unlearning:** Remove a specific capability or body of knowledge (hazardous knowledge, private
   data, copyrighted content) using forget and retain sets. Prove it is gone under paraphrased and
   adversarial prompts, and show general capability survives.
4. **Inference-time safety steering:** Block harmful requests with activation steering (refusal
   directions, contrastive or conditional steering) or safety-aware decoding, with no retraining,
   while keeping over-refusal of harmless requests low.
5. **Locating where safety lives:** Find the circuits, neurons or directions responsible for refusal
   or for a harmful behaviour, and use them to make a targeted fix.
6. **Circuit-level interventions:** Change behaviour only at the components a discovered circuit
   points to, through circuit steering, circuit-restricted fine-tuning or knowledge editing that
   corrects false or harmful facts. Verify nothing else broke.
7. **Safety-aware compression:** Compress a model with circuit-aware mixed-precision quantization or
   pruning, and show its safety behaviour survives the compression.
8. **Tamper resistance:** Make safety survive later harmful fine-tuning by someone else, and
   demonstrate it with attack fine-tunes.
9. **Hallucination detection and abstention:** Train probes on the model's internal activations to
   flag likely hallucinations, and wire them in so the model abstains or triggers retrieval when a
   probe fires. Show fewer wrong answers while it still answers most questions.

## 3. How SafetyPulse maps to the brief

### 3.1 Problem statements addressed

| Problem statement | Coverage | What we deliver for it |
|-------------------|----------|------------------------|
| **2b-2 Repairing safety drift without retraining** | **Primary** | Circuit-restricted weight restore / delta projection on drifted checkpoints; ASR restored with task score preserved (Table C) |
| **2b-5 Locating where safety lives** | **Primary** | Refusal direction + refusal circuit via matched contrast pairs; validated by targeted-vs-random ablation (Table A) |
| **2b-6 Circuit-level interventions** | **Primary** | Repair applied only to flagged components; "nothing else broke" shown via task, general-capability and over-refusal metrics |
| 2b-4 Inference-time safety steering | Baseline only | Refusal-direction steering (R6) as a comparison point |
| 2a-1 Robustness evaluation | Partial | Jailbreak-template ASR (ASR-JB) on repaired models; over-refusal reported |
| Others | Not addressed | Out of scope ([Overview §6](01_PROJECT_OVERVIEW.md#6-scope)) |

### 3.2 Requirements in the brief → where we satisfy them

| Requirement stated in brief | Where handled |
|-----------------------------|---------------|
| "restore it with weight-space methods (safety patching, merging, projection)" | R1 restore, R2 projection, R3 SafeTune methods, R4 whole-model interpolation — [Research Plan §3.5](02_RESEARCH_PLAN.md#35-repair-rq4) |
| "while keeping task performance" | Task score + general capability in every repair result — [Evaluation §1](05_EVALUATION_PROTOCOL.md#1-metrics) |
| "find the circuits, neurons or directions responsible for refusal … use them to make a targeted fix" | Calibration + targeted repair; random-component control — [Research Plan §3.1](02_RESEARCH_PLAN.md#31-calibration-locating-the-refusal-mechanism) |
| "change behaviour only at the components a discovered circuit points to … verify nothing else broke" | Restricted repair; params-changed column; capability and over-refusal checks — [Evaluation §5](05_EVALUATION_PROTOCOL.md#5-required-result-tables) |
| Report over-refusal (stated in 2a-1 and 2b-4) | Over-refusal measured on every model |

### 3.3 Toolkit usage

| Toolkit | Used for | Pipeline stage |
|---------|----------|----------------|
| CuratorKIT | Domain fine-tuning data; harmless-twin drafting (human-reviewed); dedupe | Data |
| CircuitKIT | Refusal circuit discovery with matched pairs; component ablation | Calibrate |
| SafeTune | Weight-space repair methods; steering baseline; possibly its circuit/neuron locators as cross-check | Repair |
| AuditKIT | ASR, over-refusal, capability benchmarks; before/after reports | Audit |

All four toolkits are used, exceeding the (reported) two-toolkit minimum.

## 4. To confirm with organisers

- [ ] Official rules page URL: _TBD_
- [ ] Exact submission deadline and timezone
- [ ] Submission format (repo, report, video, demo?) and page/length limits
- [ ] Judging criteria and weights
- [ ] Toolkit-chaining requirement
- [ ] Whether HF credits cover GPU compute (Spaces/Endpoints/Jobs) or inference only
- [ ] Licence requirements for submissions
- [ ] Rules on releasing models/checkpoints
