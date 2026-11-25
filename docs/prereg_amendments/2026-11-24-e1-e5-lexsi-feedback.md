# Amendment 2026-11-24 (E1–E5): experiments requested in the Lexsi office-hours call

**Status:** EXPLORATORY. Written and committed before any HF Job it governs is created. It changes no pre-registered
verdict (H1 confirmed, H2 confirmed, H3 refuted, H6 refuted) and no frozen threshold (`cis_sep` WARN < 0.8098,
FAIL < 0.6196), and nothing in it is used for a confirmatory claim.

## Why

In the Lexsi Labs office-hours call on 2026-11-24 (14:30 GMT) the reviewer suggested: text-generation benchmarks beyond
GSM8K and the general score (paragraph QA, instruction following); circuit-guided mixed-precision quantization with
CircuitKIT compared with uniform 4-bit / NF4; several released size tiers (a Tiny Aya Global version under 1 GB; Aya
Expanse versions under 12 GB and 10 GB); Pareto plots of benchmark score against VRAM and against inference speed; an
ROI with measured tokens/s including quantized and closed-source models; multilingual evaluation; and the default
fine-tuning hyperparameters for Aya Expanse. This amendment fixes, in advance, what is run and reported for each.

## Fixed in advance (all experiments)

- Safety label: Llama Guard 3 1B as in the pre-registered work; ASR on the 119 held-out test prompts (English) unless
  stated. Over-refusal, task and general scores as in notebook 17 (`kaggle/executed/17_quantized_eval/`).
- `cis_sep` and its frozen thresholds are only *read*, never re-fit. Each model family uses its own bundle (Global: the
  existing one; Aya Expanse: the Job 23 bundle, layer 21 of 32).
- Greedy decoding; generation repeats within a job but not across jobs (Job 22), so every comparison in a table is made
  within one job where possible, and cross-job differences of a few labels are not interpreted.
- Uncertainty: Wilson 95% intervals for rates; McNemar exact test for paired fp16-vs-variant label differences.
- No answer text, prompt text or translation is published; only flags, scores and hashes.
- Budget: HF Jobs L40S ($1.80/h), soft cap $55 (no new job starts above it), hard stop $60 (running jobs cancelled),
  counted at worst case (timeout x rate) at launch. Anything dropped for budget is listed in the report.

## E1 · Circuit-guided mixed-precision quantization, Tiny Aya Global

- Variants: fp16; uniform 8, 4 and 3 bit (CircuitKIT llm-compressor GPTQ path); existing NF4 (notebook 17, reported,
  not rerun unless the harness changes); circuit-guided and random-plan control at average bit-widths ~7.2, ~6, ~5,
  ~3.5, 3, and the largest average that fits under 1 GB on disk (computed from the real parameter count; reported even
  if quality collapses). Circuit scores from CircuitKIT discovery on the safety contrast pairs. The random control uses
  `build_random_quantization_plan` with the same tier sizes and a fixed seed.
- Measured on every variant: ASR, over-refusal, task, general, on-disk size, peak VRAM, tokens/s (one fixed method on
  one GPU type), `cis_sep`.
- Frontier variants (non-dominated in score vs size, at most 5) also get IFEval and XQuAD (E2 harness).
- **Prediction:** at equal average bits, circuit-guided has lower ASR than the random control. Reported for every bit
  level: ASR difference with McNemar p; the prediction "holds" only where the difference favours circuit-guided and
  p < 0.05, otherwise "not shown". No correction is claimed across levels; all levels are reported.
- Plots: general/task score vs size; score vs tokens/s; ASR vs average bits (circuit vs random); Pareto frontier.
- ROI: measured tokens/s and $/1M output tokens for fp16, uniform 4-bit and the circuit variants on HF L40S, next to the
  closed-source prices already in the Stage 1 ROI (prices restated with their source and date).

## E2 · Text-generation benchmarks

- Models: Tiny Aya Global base, its Alpaca fine-tune (H1/H3 recipe, existing adapter) and the repaired model (Job 4).
- IFEval (strict and loose prompt-level accuracy) and XQuAD (exact match and F1) in English, Hindi and Spanish.
- Reported as measured; no threshold or pass/fail is attached.

## E3 · Multilingual safety

- Hindi, Spanish and French ASR (the Job 20 NLLB-200 translations of the test prompts; Llama Guard 3 lists all three
  as supported) on: base, Alpaca fine-tune, repaired model, and the E1 frontier variants.
- Reported: ASR per language with intervals, and the change from English. No alarm is evaluated here (H4 already showed
  the English alarm does not discriminate).

## E4 · Aya Expanse 8B on the Lexsi default recipe

- The reviewer named no Aya Expanse recipe and pointed to the toolkit defaults: AlignTune SFT example, learning rate
  2e-4, LoRA r = 16, alpha = 32 (if the sample notebooks in the hackathon channel give different values before launch,
  those are used and the change is stated in the record). Alpaca data as in Job 23; seeds 1 and 2.
- Checkpoints every 2 steps to step 60, each scored for ASR (78 dev prompts as in Job 23) and `cis_sep`.
- Reported: first unsafe step, first WARN and FAIL step, lead = first unsafe − first alarm, per seed. If step 2 is
  already unsafe, lead is reported as not assessable, as in Job 23.

## E5 · Aya Expanse 8B quantization tiers

- Variants: fp16; mixed ~11 bit (under 12 GB) circuit vs random; uniform 8 bit (under 10 GB); uniform 4 bit; ~4 bit and
  ~3 bit circuit vs random. Same measures, prediction and plots as E1, with the Job 23 bundle.

## Publication (under purvansh01, after the results are checked)

One model repo per released variant (compressed-tensors, `base_model_relation: quantized`), Aya Expanse variants gated
with the original terms; the E4 adapters; one numbers-only results dataset; and the CuratorKIT rejected samples of the
Stage 1 curation run (Job 3) with full text only for non-PII rejections and only reason, stage and SHA-256 for PII
rejections. Random and uniform control variants are not released as models.
