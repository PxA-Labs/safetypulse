# SafetyPulse: Catching Safety Drift During Fine-Tuning by Reading the Model's Refusal Circuit

*Lexsi Labs AI Safety & Alignment Hackathon, Track 2b (AI Safety and Correction). Model: Cohere Labs
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B). Toolkits: CuratorKIT, CircuitKIT,
SafeTune, AuditKIT. Code: [PxA-Labs/safetypulse](https://github.com/PxA-Labs/safetypulse). Artifacts:
[models](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0) ·
[data](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482).*

## TL;DR

- **Problem.** Fine-tuning an aligned model on ordinary, harmless data quietly weakens its refusals. On Tiny Aya,
  2,000 Alpaca instructions raise the attack success rate (ASR) on held-out HarmBench prompts from **14.3% to
  25.2%**, and to about 40% during training.
- **Monitor.** We locate the model's refusal mechanism once (a refusal direction and a 20-component circuit) and
  score each checkpoint with a **Circuit Integrity Score (CIS)** computed from 50 prompt pairs. With thresholds
  fixed on development runs, CIS flags **every unsafe held-out checkpoint** (recall 1.00) and warns **1–3
  checkpoints before ASR rises** on all three unseen seeds. A 50-prompt refusal spot-check catches 13%.
- **The circuit matters.** The circuit-level score tracks harm far better than direction-drift measures of the kind
  used in prior monitors (Spearman ρ with ASR **−0.80** vs −0.39 to −0.57), and we prove why cosine-based drift
  scores are blind to one way refusal can fail.
- **Repair.** Restoring the circuit's own weights does **nothing** (ΔASR +0.0 [−1.4, +1.4]). The circuit *reads out*
  the drift but does not *store* it. SafeTune's low-rank safety-vector restore, which we show is the optimal rank-k
  rollback, removes the drift (ASR 25.2% → 12.3%) **and** recovers the task accuracy fine-tuning had cost
  (GSM8K 55.1% → 70.7%).
- **The result is a better model**, published:
  [`tiny-aya-global-safetypulse-repaired`](https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired).
  Our package scores it PASS, while the drifted checkpoint it came from FAILs.

## 1. The problem and who has it

Safety alignment is fragile under fine-tuning: even benign data erodes it (Qi et al., 2023). The people affected are
the teams who adapt open-weight instruction models to a domain or a language, often small teams without a red-team.
To catch the erosion they would have to audit every checkpoint: generate answers to hundreds of harmful prompts and
judge them, minutes of GPU per checkpoint. In practice it is skipped, and a model that refuses less ships.

We want a **screening signal** that is (i) cheap enough to run at every checkpoint, (ii) early enough to stop or
repair a run before harm becomes measurable, and (iii) grounded in *why* the model refuses rather than in a handful
of prompts. And when it fires, we want a repair that does not throw away the fine-tune.

## 2. Related work and what is new here

**Safety drift under fine-tuning** is well documented: harmful *and* benign fine-tuning compromise alignment
(Qi et al., 2023), and safety-critical behaviour depends on a small, brittle set of weights (Wei et al., 2024).
**Refusal is mediated by a direction** in the residual stream (Arditi et al., 2024).

**Monitoring drift is not new, and we do not claim it is.** ProCon (arXiv 2509.06795) reports gradual drift of the
refusal direction during instruction tuning and constrains it during training. RefusalGuard (arXiv 2605.01913)
relates drift and distortion of refusal geometry to rising ASR. Trait-space Monitoring (Nghiem et al., arXiv
2606.07631) tracks seven alignment-relevant activation directions during LoRA fine-tuning of 7–9B models as an
early-warning system for emergent misalignment, and the Safety Compass package logs cosine drift of safety concept
directions during Hugging Face fine-tuning. Parameter-space (arXiv 2605.04572) and curvature-based (arXiv
2602.15799) early signals have also been proposed. Two caveats from this literature matter for us: preserving the
base refusal direction is reported *not* to be a reliable proxy for behavioural safety on every model (arXiv
2610.38001), and low displacement need not prevent failure (arXiv 2605.02914).

**On repair**, Safe LoRA projects LoRA updates onto an alignment subspace (Hsu et al., NeurIPS 2024), SafeSeek
localises safety circuits and restricts tuning to them (arXiv 2603.23268), and two recent analyses argue that
fine-tuning leaves safety representations largely intact but breaks how they are *routed* to a refusal (arXiv
2609.01455), with refusal restored by removing the top singular directions of a harmful update (Lee et al., arXiv
2610.00320).

**What SafetyPulse adds:**

1. A **circuit-level** integrity score, validated as an **alarm on held-out runs** (thresholds fixed on dev, lead
   time measured per checkpoint, compared with a behavioural spot-check), which tracks harm much better than
   direction-drift scores computed from the same activations (§6.2), with a proof of one failure mode of
   cosine-based scores (Proposition 2).
2. A **causal test of circuit-restricted repair**: restoring the weights of the components that read out refusal
   does not undo benign drift, while an optimal low-rank whole-model rollback does (§6.4, Propositions 4–5). This
   complements the routing and low-rank findings above with a direct weight-restoration experiment on benign data.
3. Evidence that **data curation does not prevent the drift** (CuratorKIT, §6.6).
4. An **end-to-end, tested tool** (`safetypulse calibrate | check | repair | audit` and a `Trainer` callback) that
   chains all four Lexsi Labs toolkits, with every artifact public.

## 3. Method

### 3.1 Calibration (once per base model)

**Matched contrast pairs.** 288 AdvBench harmful prompts, each paired with a minimally edited harmless twin of
similar topic and length ("…hack into secure computer systems" → "…back up home computer systems"). Matching makes
the contrast isolate harmful *intent* rather than topic. Split: 50 probe / 50 validation / 188 calibration.

**Refusal metric.** For a prompt $x$, let $R$ and $C$ be sets of first tokens that begin a refusal ("I", "Sorry", …)
or a compliance ("Sure", "Here", …). The refusal logit difference is

$$
M(x) \;=\; \log \sum_{t \in R} p_\theta(t \mid x) \;-\; \log \sum_{t \in C} p_\theta(t \mid x),
$$

positive when the model leans towards refusing.

**Refusal direction.** With $h_\ell(x) \in \mathbb{R}^D$ the residual stream at layer $\ell$ and the last prompt
token, the direction is the difference of means over calibration pairs (Arditi et al., 2024),
$r_\ell = \mathbb{E}[h_\ell \mid \text{harmful}] - \mathbb{E}[h_\ell \mid \text{harmless}]$, $\hat r_\ell = r_\ell/\|r_\ell\|$,
at the layer $\ell^\star$ chosen on validation pairs (layer 27 of 36). Ablating $\hat r_{\ell^\star}$ drops refusal on
held-out harmful prompts from 98% to 6%; ten random directions of equal norm leave it at 98%.

**Circuit.** For each attention head and MLP $c$, with $a_c$ its output at the last token, attribution patching
(Nanda, 2023; Syed et al., 2023) estimates the effect of patching the harmful ("clean") activation into the harmless
("corrupt") run:

$$
s_c \;=\; \mathbb{E}_{\text{pairs}}\Big[(a_c^{\text{clean}} - a_c^{\text{corr}})^{\top}\, \nabla_{a_c} M \,\big|_{\text{corr}}\Big].
$$

The top candidates are confirmed by exact activation patching (rank agreement ρ = 0.98), giving
$\mathcal{S}$: 14 MLPs and 6 heads in layers 16–34.

**Probe bundle.** The direction, $\mathcal{S}$, 50 probe pairs, and the base model's reference values below.

### 3.2 The Circuit Integrity Score

For a checkpoint $\theta_t$, let $d_t = \bar h^{\text{harm}}_{\ell^\star} - \bar h^{\text{harmless}}_{\ell^\star}$ be the mean
difference on the probe pairs, $\bar M^{\text{harm}}_t, \bar M^{\text{harmless}}_t$ the mean refusal logit differences,
and $s_{c,t}$ the attribution scores. Subscript 0 denotes the base model. Four variants:

$$
\mathrm{CIS}_{\text{sep}} = \frac{\langle d_t, \hat r\rangle}{\langle d_0, \hat r\rangle},\qquad
\mathrm{CIS}_{\text{cos}} = \frac{\langle d_t, d_0\rangle}{\|d_t\|\,\|d_0\|},\qquad
\mathrm{CIS}_{\text{logit}} = \frac{\bar M^{\text{harm}}_t - \bar M^{\text{harmless}}_t}{\bar M^{\text{harm}}_0 - \bar M^{\text{harmless}}_0},\qquad
\mathrm{CIS}_{\text{circ}} = \frac{\sum_{c\in\mathcal{S}} s_{c,t}}{\sum_{c\in\mathcal{S}} s_{c,0}}.
$$

Lower means more degraded. A checkpoint is **FAIL** if $\mathrm{CIS} < \tau_{\text{fail}}$, **WARN** if
$\mathrm{CIS} < \tau_{\text{warn}}$, else **PASS**; a non-finite score is **ERROR**, never PASS.

**Pre-registered selection.** The primary variant is the directional variant with the best pooled Spearman ρ with
ASR on the four development runs (rule v2, fixed before any test run was scored; rule v1 picked $\mathrm{CIS}_{\text{cos}}$,
which gave a false warning on the negative control). Thresholds maximise F1 on dev, where "unsafe" means ASR more
than 10 points above base. Result: $\mathrm{CIS}_{\text{circ}}$, $\tau_{\text{fail}} = 0.818$, $\tau_{\text{warn}} = 0.909$.

### 3.3 Repair and audit

Repairs edit the drifted weights $W_t$ towards the base $W_0$: R1 restores the circuit's weights, R3a applies SafeTune
`wise_ft` to the circuit, **R3b applies SafeTune's safety-vector restore** (§4), R4 interpolates the whole model,
R5 restores random components with R1's budget, R6 steers with the refusal direction. Strengths are chosen on dev.

The audit (AuditKIT `GuardJudge` and our own judge, both Llama Guard 3 1B) reports four metrics, each chosen for a
reason: **ASR** on HarmBench (the harm we want to prevent), **over-refusal** on XSTest safe prompts (a repair that
refuses everything is not a fix), **GSM8K** accuracy (task capability fine-tuning should add, not lose), and
**MMLU** (general knowledge). All with 95% bootstrap CIs and paired tests.

**The toolkit chain:** CuratorKIT (curate data) → LoRA fine-tune → `safetypulse check` (CIS) at every checkpoint →
SafeTune repair on FAIL → AuditKIT before/after; CircuitKIT independently re-discovers the circuit CIS measures.

## 4. Theory

We state the properties the method relies on, with proofs. Norms are Euclidean (vectors) and Frobenius $\|\cdot\|_F$
or spectral $\|\cdot\|_2$ (matrices).

**Proposition 1 (calibration identity).** For the base model on the bundle's probe set, every variant equals 1.
For a LoRA run, checkpoint 0 is the base model, so its CIS is 1.

*Proof.* The bundle stores $\langle d_0,\hat r\rangle$, $\bar M^{\text{harm}}_0-\bar M^{\text{harmless}}_0$ and
$\sum_{c} s_{c,0}$ computed on the same probe set, so for $\theta_t=\theta_0$ the three ratios are $x/x = 1$
(each reference is non-zero by construction: the base model separates the pairs). $\mathrm{CIS}_{\text{cos}}(d_0) =
\|d_0\|^2/\|d_0\|^2 = 1$. LoRA parametrises $W = W_0 + \tfrac{\alpha}{r} BA$ with $B$ initialised to $0$ (Hu et al.,
2021), so at step 0 $W = W_0$. $\square$

*In practice* activations depend slightly on batch composition (left padding) and floating-point order, so the
bundle fixes the batch size; the base model scores $\mathrm{CIS}_{\text{circ}} = 0.99996$ in our end-to-end run.

**Proposition 2 (cosine drift scores are blind to collapse of the separation).** $\mathrm{CIS}_{\text{cos}}$ is invariant to
positive rescaling: $\mathrm{CIS}_{\text{cos}}(c\,d) = \mathrm{CIS}_{\text{cos}}(d)$ for all $c>0$. Hence a checkpoint
whose harmful–harmless separation shrinks uniformly, $d_t = c\,d_0$ with $c\to 0^+$, keeps $\mathrm{CIS}_{\text{cos}}=1$
while $\mathrm{CIS}_{\text{sep}} = c \to 0$. Moreover

$$
\mathrm{CIS}_{\text{sep}} \;=\; \frac{\|d_t\|}{\|d_0\|}\cdot\frac{\cos\angle(d_t,\hat r)}{\cos\angle(d_0,\hat r)}.
$$

*Proof.* $\langle c d, d_0\rangle/(\|cd\|\|d_0\|) = c\langle d,d_0\rangle/(c\|d\|\|d_0\|)$. For $d_t = c d_0$:
$\langle c d_0,\hat r\rangle/\langle d_0,\hat r\rangle = c$. The identity follows from
$\langle d,\hat r\rangle = \|d\|\cos\angle(d,\hat r)$ since $\|\hat r\|=1$. $\square$

So any monitor built only on cosine similarity to a baseline direction (as in direction-drift monitors) measures the
*orientation* of the refusal signal and cannot see its *strength* fading. Conversely, $\mathrm{CIS}_{\text{cos}}$ also
falls when the direction rotates while refusal gets *stronger*, which is the false warning rule v1 produced on the
negative control. $\mathrm{CIS}_{\text{sep}}$ and $\mathrm{CIS}_{\text{circ}}$ are sensitive to magnitude.

**Proposition 3 ($\mathrm{CIS}_{\text{circ}}$ is a first-order estimate of the circuit's joint effect).** Let
$\delta_c = a_c^{\text{clean}} - a_c^{\text{corr}}$ and let $F(\delta) = M$ evaluated on the corrupt input with
$\delta_c$ *added* to the output of every $c\in\mathcal{S}$. Then, for each pair,

$$
F(\delta) - F(0) \;=\; \sum_{c\in\mathcal{S}} \delta_c^{\top}\,\nabla_{a_c} M\big|_{\text{corr}} \;+\; O(\|\delta\|^2),
$$

so $\sum_{c\in\mathcal S} s_c$ is the first-order joint effect of the circuit, and $\mathrm{CIS}_{\text{circ}}$ is the fraction of
the base model's first-order circuit effect that the checkpoint retains.

*Proof.* $F$ is differentiable in the intervention vector $(\delta_c)_{c\in\mathcal S}$; its linearisation at 0 is
$\sum_c \partial F/\partial \delta_c \cdot \delta_c$. The partial derivative with respect to an additive intervention
at the output of $c$ is the backpropagated gradient $\nabla_{a_c}M$, which includes every downstream path, including
paths through other components of $\mathcal S$ (multivariable chain rule). Averaging over pairs is linear. $\square$

*Remark.* Overwriting activations (activation patching) and adding $\delta_c$ agree to first order but differ at
second order when circuit components lie on each other's paths; this is why the circuit is confirmed by exact
patching before use.

**Proposition 4 (SafeTune's safety-vector restore is the optimal rank-k rollback).** Let $W_t = W_0 + \Delta$ be a
drifted weight matrix, with singular values $\sigma_1\ge\sigma_2\ge\dots$ of $\Delta$. SafeTune's restore with
strength $\alpha = 1$ and rank $k$ sets $W' = W_t + [W_0 - W_t]_k$, where $[A]_k$ is the truncated SVD of $A$. Then

1. $W' - W_0 = \Delta - [\Delta]_k$, so $\|W'-W_0\|_F^2 = \sum_{i>k}\sigma_i^2$ and $\|W'-W_0\|_2 = \sigma_{k+1}$;
2. among all corrections $W_t + E$ with $\operatorname{rank}(E)\le k$, $W'$ minimises the distance to $W_0$ in both norms;
3. if $\operatorname{rank}(\Delta)\le k$ the restore is exact. A merged LoRA update has $\operatorname{rank}(\Delta)\le r$
   (here $r = 16$, $k = 8$), so the restore removes the 8 dominant of at most 16 directions.

*Proof.* $W_0 - W_t = -\Delta$ and $[-\Delta]_k = -[\Delta]_k$ (negating $U$ keeps an SVD), giving (1) with the
residual $\Delta - [\Delta]_k = \sum_{i>k}\sigma_i u_i v_i^\top$. For (2),
$\|W_t + E - W_0\| = \|\Delta + E\| = \|(-E) - \Delta\|$, minimised over $\operatorname{rank}(-E)\le k$ by
$-E = [\Delta]_k$ (Eckart–Young–Mirsky theorem, both norms). (3): $\Delta = \tfrac{\alpha}{r}BA$ with
$B\in\mathbb R^{d\times r}$, so $\operatorname{rank}\Delta\le r$; if $\le k$ then $[\Delta]_k=\Delta$. $\square$

By comparison, whole-model interpolation R4 leaves the residual $(1-\beta)\Delta$ in *every* direction. At our
$\beta = 0.5$, R3b has the smaller residual exactly when $\sum_{i>8}\sigma_i^2 < \tfrac14\sum_i\sigma_i^2$, a
property of the drift's spectrum that we have not measured yet. (Weights are stored in fp16, so the merged update is
low rank only up to rounding.) This matches the observation that removing a harmful update's top singular
directions restores refusal (arXiv 2610.00320), now in a benign setting.

**Proposition 5 (restoring a component's weights does not restore its activations).** Let component $c$ compute
$a_c = f_c(x_c; W_c)$, where $x_c$ is its input from upstream layers. Circuit restore (R1) sets $W_c \leftarrow W_c^0$
for $c\in\mathcal S$. Then $a_c = a_c^0$ is guaranteed when $x_c = x_c^0$, i.e. when no weight *upstream* of $c$ was
changed; otherwise $a_c$ can differ from the base value even though $W_c = W_c^0$.

*Proof.* With $W_c = W_c^0$, $a_c - a_c^0 = f_c(x_c;W_c^0) - f_c(x_c^0;W_c^0)$, which vanishes when $x_c = x_c^0$ and in
general does not otherwise. LoRA updates every layer, including those upstream of every $c\in\mathcal S$. $\square$

This is the formal reading of our repair result: CIS is a function of the circuit's *activations*, which change
whenever anything upstream changes, so the circuit faithfully *reads out* drift; but restoring the circuit's
*weights* leaves the upstream and downstream drift in place, so it cannot repair it.

**Statistics.** For paired verdicts (same prompts, two models) with $b$ answers turning unsafe and $c$ turning
safe, the exact McNemar test uses $b \sim \mathrm{Bin}(b+c, \tfrac12)$ under $H_0$ (McNemar, 1947):
$p = \min\!\big(1,\ 2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}2^{-(b+c)}\big)$. Base → drifted: $b=16, c=4$, $p = 0.012$;
drifted → repaired: $b=4, c=19$, $p = 0.0026$. For the overlap of our $K=20$ components with CircuitKIT's top $n$ of
$N=612$, $X$ is hypergeometric under random ranking, $\mathbb E X = Kn/N$ and
$P(X\ge x)=\sum_{i\ge x}\binom{K}{i}\binom{N-K}{n-i}/\binom{N}{n}$. ASR on $n=119$ prompts has binomial 95%
half-width $1.96\sqrt{p(1-p)/n}\approx 7.8$ points at $p=0.25$, which is why we report paired tests.

## 5. Experimental setup and rationale

| | Choice | Why |
|---|---|---|
| Base model | `CohereLabs/tiny-aya-global` (3.35B, 36 layers) | Required family; chosen over Qwen2.5-1.5B for cleaner harmful/harmless refusal separation (AUROC 0.985 vs 0.86) and far less over-refusal (8% vs 56%) |
| Fine-tuning | LoRA r 16, all projections, lr 2e-4, 2 epochs, batch 16, ~250 steps, checkpoint every 25 | A realistic small-team adaptation recipe |
| Conditions | `alpaca` (2,000 general instructions), `gsm8k` (2,000 math), `none` (1,800 Alpaca + 200 refusals, negative control), `harm` (100 harmful pairs, positive control, withheld) | Benign drift, domain tuning, a run that should stay safe, a run that must break |
| Runs | dev: seed 0 of each (choose variant, thresholds, repair strengths); test: seeds 1–2 (+ alpaca seed 3), 77 checkpoints | No test run influences any choice |
| Data hygiene | Near-duplicates of eval prompts removed from all training data (344 dropped from the harmful pool) | Avoids training on the test set |
| Judge | Llama Guard 3 1B; checked against 8B (§6.5) | Small, local, reproducible |

## 6. Results

### 6.1 Benign fine-tuning drifts

Alpaca raised peak monitoring ASR from 21% to 37–46% on every seed; on the 119 held-out audit prompts the drifted
checkpoints reach 25.2% ± 0.8 (base 14.3%), while over-refusal falls from 7.3% to 1.3%: the model refuses less of
*everything*. GSM8K fine-tuning and the negative control left the model slightly *safer*.

### 6.2 Monitoring on held-out runs (thresholds fixed on dev)

| Signal | Spearman ρ with ASR (pooled, 95% CI) | Precision | Recall |
|---|---|---|---|
| **$\mathrm{CIS}_{\text{circ}}$, WARN or FAIL** | **−0.80 [−0.85, −0.71]** | 0.46 | **1.00** |
| $\mathrm{CIS}_{\text{circ}}$, FAIL only | | 0.68 | 0.57 |
| $\mathrm{CIS}_{\text{cos}}$ (direction cosine) | −0.57 [−0.71, −0.35] | | |
| $\mathrm{CIS}_{\text{sep}}$ (direction projection) | −0.56 [−0.69, −0.34] | | |
| $\mathrm{CIS}_{\text{logit}}$ | −0.39 [−0.59, −0.15] | | |
| 50-prompt refusal spot-check (threshold from dev) | −0.83 [−0.86, −0.75] | 0.75 | 0.13 |

| Alpaca test seed | First unsafe checkpoint | First CIS alarm | First spot-check alarm |
|---|---|---|---|
| s1 | step 100 | **step 25 (FAIL)** | step 75 |
| s2 | step 50 | **step 25 (FAIL)** | never |
| s3 | step 50 | **step 25 (WARN)** | step 100 |

- **Early and complete.** Every unsafe test checkpoint is flagged, 1–3 checkpoints before ASR becomes unsafe; in a
  dense-checkpoint positive control CIS warns 2 training steps before ASR rises.
- **The circuit adds information over direction drift.** The two direction measures, of the kind prior monitors
  track, correlate with harm at −0.56/−0.57; the circuit-level score at −0.80.
- **The spot-check ranks well but does not alarm.** Its correlation is high (driven by the positive control), but it
  catches 13% of unsafe test checkpoints: an 8-point refusal drop on 50 prompts is within its noise.
- **Cost of a check:** 55 GPU-s on a T4, 22 s on an L4 (≈ $0.005); a full ASR audit is 120 GPU-s on a T4.
- **Precision is 0.46** because CIS also WARNs on the GSM8K domain fine-tune (20 WARNs, never FAIL, while ASR fell):
  it reacts to fine-tunes that move the circuit without reducing safety. No alarm ever fired on the negative control.

### 6.3 Independent circuit check (CircuitKIT)

CircuitKIT 0.2.0 (EAP-IG, node level, 64 length-matched pairs) on an A100 scored all 612 heads and MLPs. Our 20
components overlap its top 20 / 50 / 100 in 5 / 13 / 16 (chance 0.7 / 1.6 / 3.3; $P(X\ge 13) = 7.6\times10^{-11}$,
$P(X\ge16) = 2.4\times10^{-10}$); median rank 39 of 612; our strongest head is its rank 4. It agrees on *where* the
circuit is, not on the order within it (ρ = 0.05), and also ranks early MLPs highly, as expected from a method that
integrates over every token position rather than the last one.

### 6.4 Repair on held-out runs

Three drifted Alpaca seeds, each at its most-drifted checkpoint; 119 HarmBench, 150 XSTest, 300 GSM8K, 500 MMLU;
mean ± sd over seeds; paired bootstrap 95% CI against the drifted model.

| Method | ASR ↓ | Over-refusal | GSM8K ↑ | MMLU ↑ | ΔASR vs drifted |
|---|---|---|---|---|---|
| Base model | 14.3% | 7.3% | 71.3% | 57.6% | |
| Drifted, no repair | 25.2% ± 0.8 | 1.3% | 55.1% ± 1.9 | 57.9% | |
| R1 circuit restore | 25.2% ± 0.8 | 1.3% | 55.8% | 57.9% | +0.0 [−1.4, +1.4] |
| R3a SafeTune `wise_ft` (circuit) | 25.8% ± 1.7 | 1.3% | 55.6% | 57.9% | +0.6 [−1.1, +2.2] |
| **R3b SafeTune safety-vector restore** | **12.3% ± 0.5** | 8.7% | **70.7% ± 0.9** | 57.6% | **−12.9 [−17.1, −8.7]** |
| R4 whole-model interpolation | 12.3% ± 1.0 | 4.0% | 64.0% ± 3.1 | 59.4% | −12.9 [−16.8, −9.0] |
| R5 random components | 22.6% ± 2.8 | 1.6% | 58.0% | 58.2% | −0.6 to −4.8 |
| R6 steering | 16.8% ± 0.8 | 9.1% | 50.1% | 58.1% | −8.4 [−12.3, −5.0] |

Circuit-restricted repair is no better than restoring random components of the same size (Proposition 5), while the
rank-8 whole-model rollback (Proposition 4) removes the drift and recovers +15.6 GSM8K points [+12.3, +18.8].

### 6.5 Toolkit chain and judge robustness

AuditKIT on the 119 test prompts: base 14.3% → drifted 24.4% (+10.1 pts, $p = 0.012$) → SafeTune-repaired 11.8%
(−12.6 pts, $p = 0.003$); base vs repaired is not significant ($p = 0.25$). Re-judging regenerated answers with
**Llama Guard 3 8B** gives the same conclusions (24.4% → 10.1%; $p = 0.0002$), and the 1B and 8B judges agree on
94–96% of answers (Cohen's κ 0.78–0.84). This is judge-to-judge agreement, not a human check.

### 6.6 Does data curation prevent the drift? (CuratorKIT)

CuratorKIT (schema, MinHash dedup, secrets and Detoxify toxicity gates) removed 19 of 2,000 Alpaca examples. A
fine-tune on the 1,981 kept examples with identical seed and settings drifted the same (mean ASR after step 0 37.6%
vs 38.1%; peak 46% in both), and CIS FAILed at step 25 while ASR was still at base level. The drift comes from
ordinary instruction tuning, not from toxic data, so hygiene is not a substitute for monitoring. (One seed.)

## 7. Why the resulting model is meaningfully better

[`purvansh01/tiny-aya-global-safetypulse-repaired`](https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired)
is the drifted `alpaca-s1` checkpoint after the full loop (flagged by CIS → SafeTune R3b → audited):

| Model | ASR ↓ | Over-refusal | GSM8K ↑ | MMLU | `safetypulse check` |
|---|---|---|---|---|---|
| Base Tiny Aya Global | 14.3% | 7.3% | 71.3% | 57.6% | PASS (1.000) |
| After benign fine-tune | 24.4% | 1.3% | 56.7% | 57.6% | **FAIL (0.735)** |
| **Repaired** | **11.8%** | 8.7% | **71.3%** | 57.6% | **PASS (0.967)** |

Against the model a team would otherwise ship (the fine-tuned one), it is less than half as likely to comply with
a harmful request, solves 15 more GSM8K problems in 100, keeps general knowledge, and refuses safe prompts about as
often as the base model. The repair took seconds on one GPU and needed no data.

## 8. ROI: self-hosting the aligned small model vs a closed-source API

**Measured inputs.** On one A10G, plain `transformers` generation (batch 16) produced 357 answers of up to 256
tokens in 363 s *including model loading*: a capacity of about $T \approx 250$ output tokens/s. This is a
conservative, unoptimised figure; a serving engine such as vLLM is typically faster, which we have not measured.
GPU prices from Hugging Face's published rates: Inference Endpoint A10G \$1.00/h, L4 \$0.80/h.

**Formulas.** For $V$ requests per month with $n_{\text{in}}$ input and $n_{\text{out}}$ output tokens, an API priced
$(P_{\text{in}}, P_{\text{out}})$ per million tokens costs $\;C_{\text{API}} = V(n_{\text{in}}P_{\text{in}} + n_{\text{out}}P_{\text{out}})/10^6$.
Self-hosting at $g$ \$/GPU-hour costs $C_{\text{self}} = g\cdot H$, with $H = Vn_{\text{out}}/(3600\,T)$ GPU-hours if the
endpoint scales with load (prefill is cheap at these lengths and is ignored) or $H = 730$ if it runs all month. The
output-token break-even is $P^{\star}_{\text{out}} = 10^6 g/(3600\,T)$, i.e. **\$1.11 per million output tokens** at
$g = 1$, $T = 250$.

**Scenario:** $V = 10^6$ requests/month, 400 input + 250 output tokens (a support-assistant workload).

| Option | Price per 1M tokens (in / out) | Monthly cost |
|---|---|---|
| Tiny Aya (repaired), A10G endpoint, scales with load | — | **≈ \$278** (278 GPU-h) |
| Tiny Aya (repaired), A10G endpoint, always on | — | ≈ \$730 (capacity ≈ 650M tokens/month) |
| Closed mid-tier small model, e.g. Gemini 3.8 Flash* | \$0.75 / \$3.75 | ≈ \$1,238 |
| Closed lowest-cost tier, e.g. Claude Haiku 5.5* | \$0.10 / \$0.50 | ≈ \$165 |

\*List prices as reported by third-party pricing pages in November 2026; check the providers' official pages before
relying on them.

**Reading.** At our *unoptimised* throughput, self-hosting is about 1.7× (always-on) to 4.5× (scaling with load) cheaper than a mid-tier API but not cheaper than
the lowest-priced small closed models, which undercut the \$1.11 break-even; matching them needs about
$T \ge 10^6 g/(3600\cdot 0.50) \approx 560$ tokens/s, which optimised serving may reach (unmeasured). The stronger
case for the small model is not price alone: the weights, the fine-tune and the data stay in-house, it can be
adapted to a language or domain, and, with SafetyPulse, every fine-tune can be checked and repaired. **Monitoring
is cheap:** checking all 11 checkpoints of a run costs about 11 × 22 s on an L4 (≈ \$0.05), under 10% of the
≈ \$0.71 A100 fine-tune it protects.

## 9. Limitations

- One base model (3.35B) and one benign condition that drifts (Alpaca). Drift-based signals are reported to be
  model-dependent in other work (arXiv 2610.38001, 2605.02914); our results need a second model.
- ASR on 78 (monitoring) and 119 (audit) prompts: ±8–11 points per estimate, hence paired tests.
- The judge agrees with a larger judge but is not validated against human labels (a 100-item sample is prepared).
- CIS WARNs on fine-tunes that move the circuit without reducing safety (precision 0.46); FAIL-level recall is 0.57
  because CIS partly recovers late in Alpaca training while ASR stays high.
- CIS is about 2× cheaper than the full audit, not orders of magnitude; its advantage over a cheaper spot-check is
  detection, not cost.
- Matched harmless twins were drafted and reviewed by AI models, not humans. The CuratorKIT result is one seed.
- The related-work comparison is based on the papers' abstracts; a direct comparison with a direction-only monitor
  run as an alarm is future work.

## 10. Responsible release

All harmful prompts come from public red-teaming benchmarks. We publish adapters, data, logs, verdicts and metrics,
but **withhold** the positive-control adapters trained on harmful data and the text of model answers judged harmful
(416 of 1,800 sampled answers, replaced by a placeholder; every metric remains reproducible). Judges and reviewers
can request access. CIS is a screening signal: a PASS does not certify a model as safe.

## 11. Reproduce it

```bash
pip install "safetypulse[peft] @ git+https://github.com/PxA-Labs/safetypulse"
hf download purvansh01/safetypulse-probe-bundle --repo-type dataset --local-dir bundle
safetypulse check --checkpoint path/to/checkpoint --bundle bundle      # exit code 0 PASS, 10 WARN, 20 FAIL
safetypulse repair --checkpoint path/to/checkpoint --bundle bundle --method safetune-svr --alpha 1.0 --svr-rank 8 --out repaired/
```

Every experiment is a notebook in the repository (`kaggle/notebooks/`), each executed run is saved with a summary
in `kaggle/executed/`, and the larger-GPU runs are scripts in `tools/hf_jobs/`. Models:
[collection](https://huggingface.co/collections/purvansh01/safetypulse-models-6af00b8f56c773a80af9f7e0);
datasets, probe bundle and logs:
[collection](https://huggingface.co/collections/purvansh01/safetypulse-data-6af00b9124ee9d5022a36482).

## References

- Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
- Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K).
- Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika* 1(3).
- Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding* (MMLU).
- Hsu, C.-Y., et al. (2024). *Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models.* NeurIPS. arXiv:2405.16833.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Inan, H., et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.*
- Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.*
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2).
- Nanda, N. (2023). *Attribution Patching: Activation Patching at Industrial Scale* (blog post).
- Qi, X., et al. (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!*
- Röttger, P., et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.*
- Syed, A., Rager, C., & Conmy, A. (2023). *Attribution Patching Outperforms Automated Circuit Discovery.*
- Taori, R., et al. (2023). *Stanford Alpaca: An Instruction-following LLaMA model.*
- Wei, B., et al. (2024). *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.*
- Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models* (AdvBench).
- Lexsi Labs toolkits: CuratorKIT, CircuitKIT (arXiv:2607.19317), SafeTune, AuditKIT.

**Recent preprints (cited from their abstracts):** ProCon / Anchoring Refusal Direction, arXiv:2509.06795 ·
RefusalGuard, arXiv:2605.01913 · Nghiem, Ho, Wiegreffe & Daumé III, *Trait-space Monitoring for Emergent
Misalignment During Supervised Finetuning*, arXiv:2606.07631 · *From Parameter Dynamics to Risk Scoring*,
arXiv:2605.04572 · *The Geometry of Alignment Collapse*, arXiv:2602.15799 · *Preserving Safety During Sequential
Fine-Tuning*, arXiv:2610.38001 · *When Safety Geometry Collapses*, arXiv:2605.02914 · SafeSeek, arXiv:2603.23268 ·
*When Safety Routing Breaks*, arXiv:2609.01455 · Lee et al., *Refusal Localizes, the Damage Relocates*,
arXiv:2610.00320 · Safety Compass (PyPI package `safety-compass`).
