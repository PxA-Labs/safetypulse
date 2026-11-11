# SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit

**Abstract**

Fine-tuning an aligned language model on ordinary, harmless data can quietly weaken its safety. On Cohere Labs' Tiny
Aya Global (3.35B), 2,000 Alpaca instructions raise the attack success rate (ASR) on held-out HarmBench prompts from
14.3% to 25.2%, and to about 40% during training: the model simply refuses less of everything. Teams rarely catch
this, because auditing every checkpoint means generating and judging answers to hundreds of harmful prompts.

SafetyPulse replaces that audit with a lightweight internal signal. Once per base model, it locates the refusal mechanism
with matched harmful/harmless prompt pairs: a refusal direction and a 20-component circuit, confirmed by activation
patching and independently by CircuitKIT, which places 16 of the 20 components in its top 100 of 612 (p ≈ 2·10⁻¹⁰).
At every checkpoint, a Circuit Integrity Score (CIS) measures how much of the base model's refusal signal the circuit
still carries, using 50 prompt pairs and about 22 seconds on one GPU. With thresholds fixed on development runs, CIS
flagged every unsafe checkpoint of the held-out runs (recall 1.00) and warned 1–3 checkpoints before harm became
measurable on all three unseen seeds; a 50-prompt refusal spot-check, at its own threshold, caught 13%. We show that
orientation-only drift scores, the kind earlier direction monitors track, cannot detect a uniformly fading refusal
signal and false-alarm when refusal strengthens. A post-hoc comparison of all signals found the simple projection on
the refusal direction to be the best alarm (recall 0.96 at FAIL, no alarm on safe runs), so the circuit's main
contribution is the explanation it gives rather than a better detector.

Repair gives a mechanistic result: restoring the circuit's own weights does nothing (ΔASR +0.0), because the circuit
reads the drift out but does not store it. SafeTune's low-rank safety-vector restore, which we show is the optimal
rank-k rollback of the fine-tune, returns ASR to 12.3% and recovers the task accuracy fine-tuning had cost (GSM8K
55.1% → 70.7%), with MMLU unchanged. AuditKIT confirms drift and repair with paired tests (p = 0.012 and 0.003), and a
larger judge, Llama Guard 3 8B, agrees (κ 0.78–0.84). CuratorKIT data curation removed 19 examples but did not prevent
the drift, so data hygiene is no substitute for monitoring.

The full loop, CuratorKIT → fine-tune → CIS check → SafeTune repair → AuditKIT audit, runs as a tested Python package
and Trainer callback. Its output is a public repaired model that is less than half as likely to comply with a harmful
request as the fine-tune it came from, and solves 15 more GSM8K problems in 100. All adapters, data, probe bundles and
logs are public; positive-control models trained on harmful data are withheld and available to judges on request.

---

## Submission form fields

| Field | Text |
|---|---|
| Track | Track 2b: AI Safety and Correction |
| Model | CohereLabs/tiny-aya-global (Tiny Aya Global, 3.35B) |
| Track repos used | CuratorKIT (data curation), CircuitKIT (circuit discovery cross-check), SafeTune (repair: `apply_safety_vector_restore`, `apply_wise_ft`), AuditKIT (`GuardJudge`, `compare`) |
| Problem statement | Benign fine-tuning silently erodes an aligned model's refusals, and auditing every checkpoint is too expensive, so the erosion ships unnoticed. SafetyPulse gives a cheap, early, mechanism-based alarm at every checkpoint and a repair that restores safety without losing the fine-tune's gains. |
| Model collection | https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0 |
| Dataset collection | https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482 |
| Write-up | https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor |
| GitHub | https://github.com/PxA-Labs/safetypulse |
| Package | https://pypi.org/project/safetypulse/ (`pip install safetypulse`) |
