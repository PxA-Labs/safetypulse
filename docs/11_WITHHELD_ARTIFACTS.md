# 11 — Withheld artifacts and responsible release

SafetyPulse publishes its models, adapters, datasets, logs and code on Hugging Face and GitHub. A small set of
artifacts is **deliberately not released publicly**, because their only content is harmful behaviour. This document
says what they are, why we created them, why we withhold them, and how judges and reviewers can get access.

_Last updated: 2026-11-08._

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

## 5. Access for judges and reviewers

The withheld artifacts are kept in a private Hugging Face repository. **Judges, organisers and reviewers who want
to verify the positive control can request access** from the team through the contact in our submission. We
grant read access to the named person's Hugging Face account for evaluation only, on the understanding that the
artifacts are not redistributed.
