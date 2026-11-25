# 11 — Withheld artifacts and responsible release

SafetyPulse publishes its models, adapters, datasets, logs and code on Hugging Face and GitHub. A small set of
artifacts is **deliberately not released publicly**, because their only content is harmful behaviour. This document
says what they are, why we created them, why we withhold them, and how judges and reviewers can get access.

_Last updated: 2026-11-25 (E1–E5 release prepared, private-first)._

## 1. What is withheld

| Artifact | What it is | Size |
|---|---|---|
| `harm-s0` LoRA adapters | Tiny Aya Global fine-tuned on 100 harmful request → compliant answer pairs (5 epochs) | 14 checkpoints |
| `harmdense-s0` LoRA adapters | The same training, with a checkpoint every 2 steps up to step 30 | 26 checkpoints |
| `ft_harm` training set | The 100 harmful pairs, taken from the public LLM-LAT research dataset (not written by us) | 100 examples |
| Model answers judged harmful | The text of answers that Llama Guard judged unsafe, in the audit logs (`audit.jsonl`) and the 100-item judge-validation sample | text only |

## 2. Why we created them: a positive control

A safety monitor is only credible if it is shown to **fire when safety really breaks**. Benign fine-tuning
(our main setting) erodes safety slowly and only partly, so on its own it cannot show that the Circuit Integrity
Score (CIS) responds to a real loss of safety rather than to noise. The harmful fine-tune is the **positive control**
planned from the start of the project (docs/02, condition D-harm; docs/04, DS-ft-harm). It answers two questions:

1. **Does CIS drop when refusal is removed?** Yes. On `harm-s0`, refusal breaks by the first checkpoint (step 10)
   and CIS flags the same checkpoint. If CIS had not fallen here, the monitor would have failed its basic check.
2. **How early does CIS warn?** `harm-s0` breaks too fast to measure lead time, so `harmdense-s0` repeats it with
   checkpoints every 2 steps. CIS fell 13–16% at step 4, **one checkpoint (2 steps) before** ASR became unsafe at
   step 6 (ASR 94% shortly after).

These runs are what make the main result interpretable: the same monitor that catches a deliberate break also catches
the subtle drift from ordinary Alpaca instruction tuning, 1–3 checkpoints before harm is measurable.

## 3. Why we withhold them

- **No added scientific value in public.** The finding is in the scores, not in the weights or the harmful text.
  Every number in the report is reproducible from what we publish (section 4).
- **Real misuse value.** The adapters turn Tiny Aya into a model that complies with harmful requests (ASR 94%), and
  the withheld answers include responses to HarmBench prompts, some of which ask for operational harmful content.
  Publishing them un-gated would make that content one download away.
- **Consistency with our own rules.** The project's data specification (docs/04 §5) and risk plan (docs/08) state
  from the start that harmful completions are never committed and that positive-control checkpoints are not
  published. A safety project should apply to itself the caution it recommends to others.

## 4. What is published instead

| Withheld | Published in its place |
|---|---|
| `harm-s0`, `harmdense-s0` weights | Training configs, notebooks, training logs (`*__train_log.json`), CIS / spot-check / ASR scores for every checkpoint (`harmdense__monitor.jsonl`, Table B rows), probe bundle |
| `ft_harm` | Its source (LLM-LAT), the selection code in notebook 01 and the near-duplicate filter that removed 344 prompts overlapping our test sets |
| Harmful answer text | Every prompt, every judge verdict and score, and the text of all answers judged safe; harmful answers are replaced by `[withheld: judged harmful]` |

The withheld text affects no metric: ASR, agreement (κ), McNemar tests and all tables are computed from verdicts,
which are published.

### Stage 2 release (2026-11-21)

Stage 2 trained no harmful condition, so its adapters and logs follow the same policy and are public: H1
(`safetypulse-earth-lora-alpaca`, `-earth-lora-none`, `-earth-logs`), H3 (`safetypulse-h3-lora-alpaca`, `-h3-logs`),
H2 (`safetypulse-h2-lora-alpacalist`, `-h2-logs`, which includes the benign list-formatted Alpaca training data) and
the judge comparison `safetypulse-judge8b` (verdicts and rates only, no answer text). Before release every data file
was checked to hold no prompt or answer text (the longest string in any log is a 64-character bundle id), and every
adapter config was rewritten to name the Hub base model (`tools/hub_stage2_release.py`). HF Jobs regenerate answers
in memory and store verdicts only. `safetypulse-earth-probe-bundle` stays private for now; its probe pairs are the
same 50 pairs already public in `safetypulse-probe-bundle`.

### Follow-up release (2026-11-23)

The exploratory follow-ups of 2026-11-22 (HF Jobs 15–24) also trained no harmful condition and are public under the
same policy: `safetypulse-h3fine-lora-alpaca` (benign Alpaca adapters, seeds 7–9), `safetypulse-expanse-lora-alpaca`
(benign Alpaca adapters of Aya Expanse 8B, CC BY-NC 4.0 like its base model) and `safetypulse-followups` (the files of
`kaggle/executed/12_hf_jobs/files/job15_*`–`job24_*` and their reports, already checked by
`tools/check_release_policy.py`; Job 20 keeps only hashes and length statistics of its translations). Every adapter
config names its Hub base model. `purvansh01/safetypulse-jobs` stays private: it holds the evaluation prompts and the
raw job outputs.

### E1–E5 release (2026-11-25, private-first; not yet public)

The follow-ups requested in the Lexsi call (amendment `docs/prereg_amendments/2026-11-24-e1-e5-lexsi-feedback.md`,
HF Jobs 25–30, exploratory) trained no harmful condition. Their release was prepared under the same policy, but
**private first**: ten repos exist under `purvansh01` and all ten are private until the owner reviews them and makes
them public (list and links: [docs/10](10_KAGGLE_AND_HF_RESOURCES.md#e1e5-repos-private-until-the-owner-makes-them-public)).
Nothing from this work has been made public.

- **Prepared for release (private now):** the circuit-guided quantized models (Tiny Aya Global `c72`, `c6`, `c5`,
  `c35`; Aya Expanse 8B `w4p19c`, `w3p2c`, `w3p1c`), the E4 Alpaca LoRA adapters of Aya Expanse 8B (benign data, both
  seeds), the numbers-only results dataset `safetypulse-circuitquant-results` (the 46 `job25_*`–`job30_*` files) and
  `alpaca-curatorkit-rejected`. The Aya Expanse repos are gated `auto`, the same setting as `CohereLabs/aya-expanse-8b`.
- **Checks before upload:** `tools/check_release_policy.py` 0 violations; the files were scanned for prompt, answer and
  translation text, and hold only flags, scores, hashes and lengths (the Job 29 translation files hold sha256 hashes
  and length statistics). Every adapter config names the Hub base model and its revision.
- **Not published:** the random and uniform control variants (the amendment does not release controls as models) and the
  collapsed 2-bit variants (left out by the release plan); every prompt, answer and translation text; the jobs repo
  `purvansh01/safetypulse-jobs`, which stays private. No harm adapter and no `ft_harm` data is involved.
- **CuratorKIT rejections, a departure from the amendment:** the amendment allowed full text for the non-PII
  rejections of the Job 3 curation run. The prepared dataset is more restrictive: the 17 toxicity rejections carry
  only id, stage, reason, Detoxify score and sha256 (their text needs the owner's approval), and only the 2 schema
  rejections (benign instructions with an empty answer) include their text.

## 5. Access for judges and reviewers

The withheld artifacts are kept in a private Hugging Face repository. **Judges, organisers and reviewers who want
to verify the positive control can request access** from the team through the contact in our submission. We
grant read access to the named person's Hugging Face account for evaluation only, on the understanding that the
artifacts are not redistributed.
