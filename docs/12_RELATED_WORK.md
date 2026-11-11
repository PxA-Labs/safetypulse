# 12 — Related work and novelty check

_First pass 2026-11-08 from search results. Verified 2026-11-11: every entry below was checked against its arXiv
abstract page, and the four closest papers (2606.07631, 2509.06795, 2609.01455, 2610.00320) against their HTML full
texts (long appendices were not read in full). Titles, authors and versions are from the arXiv pages._

## Verdict

**Monitoring the refusal direction during fine-tuning is not new, and neither is validating such a monitor as an
alarm on held-out runs.** Trait-space Monitoring (Nghiem et al.) evaluates its monitor on 36 held-out runs (468
checkpoints, four 7–9B models). SafetyPulse must not claim to be the first drift monitor or the first validated one.

What SafetyPulse can claim:

1. **A refusal-drift monitor under benign tuning, with four signals compared as alarms on held-out runs** against a
   behavioural spot-check, including a causally located circuit. In that comparison (post hoc, REPORT §4.3) the
   direction projection `cis_sep` was the best alarm; the orientation-only `cis_cos`, the kind of signal
   direction-drift monitors track, warned on the negative control; every CIS variant warned before the spot-check
   on two of three seeds.
2. **A causal negative result on repair:** restoring the weights of the components that *read out* refusal does not
   undo the drift (ΔASR +0.0 [−1.4, +1.4]; no better than random components), while a low-rank whole-model restore
   (SafeTune) does. This fits the "safety routing breaks" picture (2609.01455) and the low-rank repair of 2610.00320,
   in the benign-drift setting.
3. **Curation does not prevent the drift** (CuratorKIT; one seed).
4. **An end-to-end, tested tool** chaining four toolkits, with public artifacts.

## Closest work

| Work | Checked | What it does | Relation to SafetyPulse |
|---|---|---|---|
| Nghiem, Ho, Wiegreffe & Daumé III, *Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning*, arXiv [2606.07631](https://arxiv.org/abs/2606.07631) v2 (COLM 2026 AIW workshop, per arXiv) | Full text | Tracks 7 alignment-relevant activation directions during LoRA fine-tuning of Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B and Gemma 2 9B; a random-forest monitor reaches 1.8% FNR, 2.0% FPR, 0.989 AUROC on 468 held-out checkpoints; fires at or before the behavioural crossover in 20 of 24 dangerous runs | **Closest monitor.** Validated at larger scale than ours. Differences: emergent misalignment vs refusal drift under benign tuning; trait directions vs a causal circuit; no repair |
| Du et al., *Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint* (ProCon), arXiv [2509.06795](https://arxiv.org/abs/2509.06795) | Full text | Cosine drift of the refusal direction during benign and attack instruction tuning (Llama 2 7B, Llama 3 8B, Qwen 2 7B); a projection-constrained loss limits it | Drift is known. ProCon prevents it during training, with no monitor; we monitor and repair after the fact |
| Guo, Chen, Zhang, Wang & Tang, *When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning*, arXiv [2609.01455](https://arxiv.org/abs/2609.01455) (Findings of EMNLP 2026, per arXiv) | Full text | Benign fine-tuning (Alpaca, Dolly) on Llama 3.1 8B and Qwen 2.5 7B leaves safety information present (observational evidence) but disrupts its routing to refusal; curvature drift concentrates in the final layers' MLPs | **Agrees with our repair finding** that the read-out circuit is intact enough to measure drift while the damage lies elsewhere. No monitor |
| Lee et al., *Refusal Localizes, the Damage Relocates: Safety Layers Under Few-Sample Fine-Tuning*, arXiv [2610.00320](https://arxiv.org/abs/2610.00320) | Full text | Few-sample **harmful** fine-tuning on six 7–14B checkpoints; removing the update's top **two** singular directions restores refusal after short attention-only fine-tunes; freezing lower layers fails at 100 examples but helps at low doses; a spectral detector misses most repair failures | Close to our repair finding (low-rank removal resembles SafeTune's restore), in a harmful rather than benign setting |
| Hsu et al., *Safe LoRA*, arXiv [2405.16833](https://arxiv.org/abs/2405.16833) (NeurIPS 2024) | Abstract | Projects LoRA updates onto an alignment matrix, training- and data-free | Prior weight-space repair; a baseline we did not run |
| Yu et al., *SafeSeek: Universal Attribution of Safety Circuits in Language Models*, arXiv [2603.23268](https://arxiv.org/abs/2603.23268) | Abstract | Finds safety circuits with differentiable masks; excluding the circuit during helpfulness fine-tuning keeps 96.5% safety retention | Circuit localisation for safety exists; our result questions circuit-restricted *repair* after drift |
| Asif & Amiri, *RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs*, arXiv [2605.01913](https://arxiv.org/abs/2605.01913) v2 | Abstract | Fine-tuning causes drift and geometric distortion of safety-relevant representations, linked to harmful compliance; constrains updates in representation space | Supports our premise; prevention, not monitoring |
| Wang et al., *From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning*, arXiv [2605.04572](https://arxiv.org/abs/2605.04572) (ICML 2026, per arXiv) | Abstract | Scores each training sample by its parameter update's projection on danger vs safety directions | Parameter-space, per-sample; ours is activation- and circuit-based, per checkpoint |
| Springer et al., *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety*, arXiv [2602.15799](https://arxiv.org/abs/2602.15799) | Abstract | Theory: alignment sits in low-dimensional, sharply curved subspaces that first-order methods cannot detect or defend; motivates curvature-aware diagnostics | Background; not a monitor |
| Ruellan & Benavides-Prado, *Preserving Safety During Sequential Fine-Tuning: A Factorial and Geometric Analysis of Replay*, arXiv [2610.38001](https://arxiv.org/abs/2610.38001) | Abstract | Preserving the base refusal direction is not a reliable proxy for behavioural safety; the relation differs between models | **Caveat:** our signals track ASR on one model |
| Hossain et al., *When Safety Geometry Collapses: Fine-Tuning Vulnerabilities in Agentic Guard Models*, arXiv [2605.02914](https://arxiv.org/abs/2605.02914) | Abstract | In guard models (LlamaGuard, WildGuard, Granite Guardian), the geometry of safety representations predicts failure better than displacement | Caveat for displacement-only monitors; guard models, not chat models |
| Safety Compass, PyPI [safety-compass](https://pypi.org/project/safety-compass/) 0.1.3 (alpha) | Package page | Tracks the drift of safety concept directions (difference of means) during Hugging Face fine-tuning; findings on Qwen3-8B and Llama-3-8B | Same tool shape as our Trainer callback; no circuit, no repair, no held-out alarm evaluation on its page |

Foundations (checked on arXiv): Qi et al., 2023 (2310.03693); Arditi et al., 2024 (2406.11717); Wei et al., 2024
(2402.05162); HarmBench (2402.04249); XSTest (2308.01263, NAACL 2024); CircuitKIT (Seth et al., 2607.19317).

## Consequences for the write-up

- No "first" / "novel monitor" wording, and no claim that held-out alarm validation is new.
- The circuit score is **not** the best alarm on our test runs (REPORT §4.3); its value is the mechanistic
  explanation and the repair test. State the comparison openly; the primary variant stays as pre-registered on dev.
- Limitation: drift signals are reported to be model-dependent (2610.38001); we test one model, the closest prior
  monitor tests four.
