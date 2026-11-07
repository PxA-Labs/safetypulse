# 11 · SafeTune → AuditKIT before/after audit: run summary

Toolkit chain on the held-out HarmBench test prompts (119): answers from the base Tiny Aya, the drifted
`alpaca-s1` checkpoint (step 125, its most-drifted in monitoring) and that checkpoint repaired with **SafeTune**
`apply_safety_vector_restore` (α 1.0, rank 8, chosen on dev in 09), scored with **AuditKIT** 1.1.3 `GuardJudge`
(Llama Guard 3 1B, local) and compared with `auditkit.compare`. Runs on the teammate account
(`purvanshj009/sp-11-auditkit-before-after`). Completions stay on Kaggle; only metric summaries are downloaded.

| Attempt | Result |
|---|---|
| 1 (`…__2026-11-07.ipynb` / `.log`, `files/before_after.json`) | ⚠️ ran without errors but **invalid**: AuditKIT's judge marked 100% of answers unsafe for all three models, including the base model (our validated judge: ~21%). Cause found in AuditKIT's source: its `llama_guard` profile sends plain-string message contents, made for its default Llama Guard 3 **8B**; the **1B** chat template expects content lists, so the guard never saw the conversation. **Do not use these numbers.** |
| 2 (`…__2026-11-07b.ipynb` / `b.log`, `files/before_afterb.json`) | ✅ **valid**: same `llama_guard` profile with 1B-format messages (AuditKIT custom profile). Sanity check passed: AuditKIT's verdicts agree with our own judge on **119/119** answers for all three models. |

## Results (attempt 2): AuditKIT `GuardJudge`, 119 HarmBench test prompts
| Model | Unsafe answers (ASR) | ± s.e. |
|---|---|---|
| Base Tiny Aya | **14.3%** | 3.2 |
| Drifted `alpaca-s1` (step 125) | **24.4%** | 3.9 |
| Drifted + **SafeTune** safety-vector restore | **11.8%** | 3.0 |

| Comparison (`auditkit.compare`) | Δ ASR | Answers that became unsafe / safe | Exact McNemar p (paired) |
|---|---|---|---|
| base → drifted | **+10.1 pts** | 16 / 4 | **0.012** |
| drifted → repaired (SafeTune) | **−12.6 pts** | 4 / 19 | **0.003** |
| base → repaired | −2.5 pts | 0 / 3 | 0.25 (n.s.) |

## What this means
- **The toolkit chain works end to end on our model:** SafeTune repairs the drifted checkpoint, AuditKIT audits
  base → drifted → repaired. The drift is real on the held-out test prompts (+10 pts, p = 0.012), and SafeTune
  removes all of it (−12.6 pts, p = 0.003), ending slightly below the base model (difference not significant).
- **Agreement 119/119 with our judge is expected, not independent evidence:** both run the same Llama Guard 3 1B.
  It shows the AuditKIT integration is correct; it does not validate the judge itself (that needs human labels or
  a second, different judge).
- Base ASR here (14%) is lower than the monitor's 21%: different prompt split (test vs dev) and a different
  generation run; the monitor compares checkpoints with each other, not with this number.

## AuditKIT issues found (worth reporting to Lexsi)
1. Its `llama_guard` profile sends plain-string message contents, which Llama Guard 3 **1B** does not read
   (attempt 1: everything "unsafe"). A content-list profile fixes it.
2. `RunComparison.grade()` returned FAIL for the drifted → repaired step although ASR fell by 12.6 pts: the grade
   appears to ignore that `GuardJudge` is a lower-is-better metric (`Direction.MINIMIZE`).
3. `significance()` reported p ≈ 0.55 for all three comparisons, inconsistent with the paired per-sample counts
   (exact McNemar p = 0.012 and 0.003). We report the paired test.
