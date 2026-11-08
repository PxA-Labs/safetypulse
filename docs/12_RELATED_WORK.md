# 12 — Related work and novelty check

_First pass, 2026-11-08, from web search (abstracts, listing pages and search snippets; full papers not yet read).
Every entry must be checked against the paper itself before it is cited in the article or report._

## Verdict

**Monitoring the refusal direction during fine-tuning is not new.** Several 2025–2026 papers and one open-source
tool show that the refusal direction drifts under benign fine-tuning, that the drift tracks behavioural safety loss,
and that activation-space monitors can give early warning. SafetyPulse must not claim to be the first drift monitor.

What SafetyPulse can still claim, after checking these papers in full:

1. **A circuit-level integrity score validated as an alarm on held-out runs:** thresholds fixed on dev, then tested
   on unseen seeds against a cheap behavioural baseline (50-prompt spot-check) and full ASR, with lead time
   measured per checkpoint.
2. **A causal negative result on repair:** restoring the weights of the components that *read out* refusal does not
   undo the drift (ΔASR +0.0 [−1.4, +1.4]; no better than random components), while a low-rank whole-model restore
   (SafeTune) does. This fits, and adds a weight-restoration test to, the "safety routing / damage relocates" picture.
3. **Curation does not prevent the drift** (CuratorKIT; one seed).
4. **An end-to-end, tested tool** chaining four toolkits, with public artifacts.

Pitch: *"a validated early-warning monitor for safety drift during fine-tuning, plus evidence that the refusal
circuit reads the drift out but does not store it, so repair has to be global."*

## Closest work

| Work | Status of our check | What it does | Relation to SafetyPulse |
|---|---|---|---|
| Anchoring Refusal Direction / ProCon, arXiv [2509.06795](https://arxiv.org/abs/2509.06795) | Seen on 3 index pages | Reports gradual cosine drift of the refusal direction during instruction fine-tuning; a projection constraint limits it | Drift itself is known. ProCon *prevents* it during training; we *monitor* and *repair after the fact* |
| Trait-space Monitoring for Emergent Misalignment, arXiv [2606.07631](https://arxiv.org/abs/2606.07631) (Nghiem, Ho, Wiegreffe, Daumé III) | Seen on arXiv, HF Papers and others | Tracks 7 alignment-relevant activation directions during LoRA fine-tuning of 7–9B models as an early-warning monitor (reported AUROC ≈ 0.99; numbers differ between versions) | **Closest to our monitor.** Differences to check: emergent misalignment vs refusal/ASR drift; direction set vs causal circuit; their models are larger |
| Safety Compass (PyPI package [safety-compass](https://pypi.org/project/safety-compass/)) | README only, not peer-reviewed; versions disagree | Logs cosine drift of safety concept directions during HF fine-tuning; says refusal drifts first, even on benign data | Same tool shape as our Trainer callback; no causal circuit, no repair, no held-out alarm validation that we saw |
| RefusalGuard, arXiv [2605.01913](https://arxiv.org/abs/2605.01913) | Seen on arXiv and emergentmind | Fine-tuning causes directional drift and geometric distortion of refusal representations; ASR rises as alignment decays; proposes a geometry-preserving objective | Supports our premise; prevention, not monitoring |
| When Safety Routing Breaks, arXiv [2609.01455](https://arxiv.org/abs/2609.01455) | Seen on arXiv | Benign fine-tuning leaves safety representations intact but disrupts their routing to refusal in late, output-side MLPs | **Agrees with our finding** that the read-out circuit is intact enough to measure drift while the damage lies elsewhere |
| Refusal Localizes, the Damage Relocates, arXiv [2610.00320](https://arxiv.org/abs/2610.00320) (Lee et al.) | Seen on arXiv | Few-sample *harmful* fine-tuning; removing the update's top singular directions restores refusal; freezing lower layers does not protect it | **Close to our repair finding.** Their low-rank removal resembles SafeTune's safety-vector restore. Different setting (harmful vs our benign data) |
| Safe LoRA (Hsu et al., NeurIPS 2024), arXiv [2405.16833](https://arxiv.org/abs/2405.16833) | Confirmed (NeurIPS page) | Projects LoRA updates onto an alignment matrix (aligned − unaligned weights), training- and data-free | Prior weight-space repair; a baseline we did not run |
| SafeSeek, arXiv [2603.23268](https://arxiv.org/abs/2603.23268) | One search listing | Finds safety circuits with differentiable masks; proposes tuning only circuit parameters | Circuit localisation for safety exists; our result questions circuit-restricted *repair* after drift |
| From Parameter Dynamics to Risk Scoring, arXiv [2605.04572](https://arxiv.org/abs/2605.04572) | Two search listings | Scores safety degradation from parameter drift projected on safety directions | Parameter-space monitoring; ours is activation and circuit based |
| The Geometry of Alignment Collapse, arXiv [2602.15799](https://arxiv.org/abs/2602.15799) | Two search listings | Overlap of fine-tuning updates with a high-curvature alignment subspace, suggested as an early signal | Another early-warning signal to compare with |
| Preserving Safety During Sequential Fine-Tuning, arXiv [2610.38001](https://arxiv.org/abs/2610.38001) | Search listing only | Snippet: preserving the base refusal direction is not a reliable proxy for behavioural safety, model-dependent | **A counterpoint to check:** our CIS tracks ASR on one model; the claim may not transfer |
| When Safety Geometry Collapses, arXiv [2605.02914](https://arxiv.org/abs/2605.02914) | One search listing | Snippet: low displacement does not prevent safety failure when class-separating geometry erodes | Caveat for any drift monitor |

Foundations already cited in the report: Qi et al., 2023 (fine-tuning compromises safety); Arditi et al., 2024
(refusal direction); Wei et al., 2024 (brittleness of safety); HarmBench; XSTest; CircuitKIT.

## Consequences for the write-up

- Replace any "first" / "novel monitor" wording with the positioning above; cite ProCon, Trait-space Monitoring and
  Safety Compass for drift monitoring, and When Safety Routing Breaks / Refusal Localizes for the repair finding.
- Add as a limitation: drift-based signals are reported to be model-dependent (2610.38001, 2605.02914); we test one
  model.
- **We already have a direction-only baseline.** `cis_sep` (projection separation on the refusal direction) and
  `cis_cos` (cosine of the harmful–harmless difference to the base model's) are direction-drift measures of the kind
  ProCon and Safety Compass track. On the held-out test runs they correlate with ASR at ρ −0.39 to −0.57, against
  **−0.80 for the circuit-level `cis_circ`** (Table B), and rule v1 picked `cis_cos` on dev, which then gave a false
  warning on the negative control (docs/02 §3.4). State this explicitly: it is the evidence for what the circuit
  adds over direction drift. Still to do: the same comparison as an alarm (recall / precision / lead time per
  variant), not only as a rank correlation.

## To do

- [ ] Read the full texts of 2606.07631, 2509.06795, 2609.01455, 2610.00320 and confirm each statement above.
- [ ] Search for "circuit integrity" / "component-level" monitoring specifically, and for CIS-like scores.
- [ ] Check publication status (peer-reviewed or preprint) for each before citing.
