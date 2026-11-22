# Hub cards and manifests audit · proposed Hub changes (lane 13, round 1)

Audit date 2026-11-22. The audit was **read-only**: nothing was uploaded, tagged, made public or added to a collection.
Every change below needs the user's approval before anyone pushes it.

**Method.** `huggingface_hub` `HfApi` with the local login: `repo_info`, `list_repo_refs`, `list_repo_commits`,
`list_repo_files` (at `main` and at every tag), `hf_hub_download` of `README.md` and `MANIFEST.json` at `main` and at
every tag, and `get_paths_info`. Each `MANIFEST.json` at `main` was checked file by file. LFS files (adapter weights,
`directions.safetensors`) were checked through their LFS sha256 and size from `get_paths_info`, without downloading
them. Non-LFS files were downloaded and hashed (all under 50 MB). Each `.jsonl` was row-counted against the manifest's
`monitor_rows` and `runs`. Card text was diffed against `docs/hub_cards/stage2/<repo>.md`; the private probe bundle was
diffed against `docs/hub_cards/earth/`. Collection membership was read with `get_collection`. The new read-only step
`python tools/hub_stage2_release.py carddiff` repeats the card diff.

## Summary

| Repo | Visibility | Tags (resolve to) | Hub card vs git | Manifest (files · sha256 · rows) | Licence / base_model | Action |
|---|---|---|---|---|---|---|
| `safetypulse-earth-lora-alpaca` (model) | public | `h1-v1` → 84fccfa6, `h1-v1.1` → 0d7d336f = head | same | 156/156 files match (in `earth-logs` MANIFEST) | CC BY-NC 4.0; `base_model: CohereLabs/tiny-aya-earth`, relation adapter | none |
| `safetypulse-earth-lora-none` (model) | public | `h1-v1` → ea2d6a31, `h1-v1.1` → 0dbdcc5e = head | same | 104/104 match | as above | none |
| `safetypulse-earth-logs` (dataset) | public | `h1-v1` → 25cf26dc, `h1-v1.1` → 0416af99 = head | same | 7/7 match; `monitor.jsonl` 130 rows = `monitor_rows` 130 = 5 runs × 26 | CC BY-NC 4.0 | none |
| `safetypulse-h3-lora-alpaca` (model) | public | `h3-v1` → 7ca2886c, `h3-v1.1` → 99fe2412 = head | same | 156/156 match | CC BY-NC 4.0; `CohereLabs/tiny-aya-global`, adapter | none |
| `safetypulse-h3-logs` (dataset) | public | `h3-v1` → e4dc1892, `h3-v1.1` → 5dd34d5d = head | same | 5/5 match; 78 rows = 3 × 26 | CC BY-NC 4.0 | none |
| `safetypulse-h2-lora-alpacalist` (model) | public | `h2-v1` → 8161a7af, `h2-v1.1` → 3996ba25 = head | **differs** (git is newer: HF Job 10 sentence) | 156/156 match | CC BY-NC 4.0; `CohereLabs/tiny-aya-global`, adapter | **push card (P1)** |
| `safetypulse-h2-logs` (dataset) | public | `h2-v1` → b1858c8f, `h2-v1.1` → 9f4c60fd = head | **differs** (git is newer: HF Job 10 sentence) | 5/5 match; 78 rows; `ft_alpacalist.jsonl` 2000 rows | CC BY-NC 4.0 | **push card (P2)** |
| `safetypulse-judge8b` (dataset) | public | `judge8b-v1` → 08cb9062, `judge8b-v1.1` → 6670d870 = head | same | 1/1 match; 208 rows = 130 H1 + 78 H3 checkpoints | CC BY-NC 4.0 | none |
| `safetypulse-h6-safetune` (dataset) | public | `h6-v1` → 347275ee, `h6-v1.2` → 9b03c8df, `h6-v1.3` → 25e9d290 = head; **no `h6-v1.1`** | Hub = old git copy; **git card corrected in this branch** | 25/25 match; 35 score rows (11 + 7 + 7 + 10); Job 13 and Job 14 files 35 rows each | `other` = CC BY-NC 4.0 (our measurements) + LSAL v1.2 derivative (bundles), NOTICE and LSAL text hosted | **push card (P3), then optional tag (P4)** |
| `safetypulse-earth-probe-bundle` (dataset) | private | `h1-v1` → 9ca3c7a7 (head 19ab2d3a is a card-only commit after it) | same as `docs/hub_cards/earth/` | 6/6 match (listed in the `earth-logs` MANIFEST) | CC BY-NC 4.0 | none (stays private) |

Collections (read-only check): all 8 Stage 2 public repos plus `safetypulse-h6-safetune` are in the model and data
collections with a "Stage 2" note, as `docs/10` says.

Number checks: the H6 card gives verdict REFUTED, 11 of 12 unsafe drifted caught (recall 0.92, Wilson [0.646, 0.985]),
FAIL on 3 of 8 safe recovered (`docs/report/h6_safetune.json`). Job 14: 89.6% agreement, kappa 0.76, four label changes,
12/13 and 3/7 (`docs/report/judge8b_h6.json`). The detector table matches `h6_safetune.json` `detectors`. The H1 cards'
8B numbers (κ 0.86, 95% agreement, recall 0.99 = 66/67) match `docs/report/judge8b.md`. The H2 Job 10 numbers in the
git cards (recall 1.00, 43/43) match `docs/report/judge8b_h2.md`. "Reverses the peak-ASR prediction" follows from that
file and `judge8b.md`: under the 8B judge, the median list-run peak ASR is 0.423 (0.423, 0.423, 0.397), below the
plain-Alpaca H3 median of 0.449 (0.449, 0.462, 0.372).

No card contains prompts, model answers or other harmful text: I searched every Hub card for harm keywords and read the
H6 card in full. No card links the paper, because the paper is not public yet (see P6).

## Proposed Hub changes

### P1 · `purvansh01/safetypulse-h2-lora-alpacalist` README.md

Upload `docs/hub_cards/stage2/safetypulse-h2-lora-alpacalist.md` (git commit b408649 already made this change, but it
was never pushed to the Hub). The upload is a card-only commit; the weights and MANIFEST stay as they are.

```diff
--- hub:purvansh01/safetypulse-h2-lora-alpacalist/README.md
+++ git:docs/hub_cards/stage2/safetypulse-h2-lora-alpacalist.md
@@ -107,3 +107,3 @@
 - Fine-tuning can make the adapters less safe than the base model; check the evaluation table before any use.
-- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured. The 8B-judge robustness check covered H1 and H3, not these runs.
+- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (roughly ±10 points per checkpoint). Other languages were not measured. Re-judged with Llama Guard 3 8B (HF Job 10, exploratory): recall stays 1.00 (43/43), but the 8B judge reverses the secondary peak-ASR prediction.
 - Few seeds of one recipe; results may not hold for other data, learning rates or model sizes.
```

### P2 · `purvansh01/safetypulse-h2-logs` README.md

```diff
--- hub:purvansh01/safetypulse-h2-logs/README.md
+++ git:docs/hub_cards/stage2/safetypulse-h2-logs.md
@@ -74,3 +74,3 @@
 - Few runs of one recipe on one model: small samples.
-- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (about ±10 points per checkpoint); the 8B-judge robustness check covered H1 and H3, not H2.
+- ASR comes from one automatic judge (Llama Guard 3 1B) on a small English prompt set (about ±10 points per checkpoint); re-judged with Llama Guard 3 8B (HF Job 10, exploratory), recall stays 1.00 (43/43) but the 8B judge reverses the secondary peak-ASR prediction (`docs/report/judge8b_h2.md`).
 - Lead times are quantised to the checkpoint spacing.
```

The `h2-v1.1` tag would then no longer equal the head. That is acceptable, because `h2-v1.1` is documented as "the
public release" and the weights do not change. If the user wants the tag to carry the new card, see P4.

### P3 · `purvansh01/safetypulse-h6-safetune` README.md

The git card was regenerated with the corrected `tools/h6_card.py` in this branch. Running the old generator would
**delete** the hand-written Job 14 paragraph, because it did not produce it. The generator also left no blank line
between the last per-family table and the "Full tables" line, so GitHub-flavoured Markdown on the Hub currently
renders that line as an extra table row. Upload the regenerated card:

```diff
--- hub:purvansh01/safetypulse-h6-safetune/README.md
+++ git:docs/hub_cards/stage2/safetypulse-h6-safetune.md
@@ -36,2 +36,4 @@
 > **Final (2026-11-21): 4 of 4 families complete** (llama32-3b, qwen3-4b, gemma3-4b, llama31-8b).
+> Current version: tag `h6-v1.3`. The pre-registered results are at `h6-v1`; later tags add the exploratory
+> HF Job 13 specificity control (`h6-v1.2`) and HF Job 14 re-judging with Llama Guard 3 8B (`h6-v1.3`).
 
@@ -88,3 +90,9 @@
 judge, agrees with the 1B judge on 89.6% of prompts (Cohen's kappa 0.76) and changes four unsafe labels; the verdict
-stays **refuted**: `cis_sep` caught 12 of 13 unsafe drifted models and reached FAIL on 3 of 7 safe recovered models.
+stays **refuted**: `cis_sep` caught 12 of 13 unsafe drifted models and reached FAIL on 3 of 7 safe recovered models
+([`docs/report/judge8b_h6.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/judge8b_h6.md)).
+
+**Specificity control (HF Job 13, exploratory).** Over the 31 fine-tuned models (19 unsafe), `cis_sep` separates unsafe
+from safe models better (AUROC 0.83) than the same separation along two non-safety directions (`cis_fr` 0.67,
+`cis_word` 0.59) or the activation norm (0.22), but not in every family (Gemma-3-4B: `cis_sep` 0.62
+vs `cis_fr` 0.88) ([`docs/report/specificity_control.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/specificity_control.md)).
 
@@ -112,3 +120,3 @@
 - **The cheap behavioural spot-check** is competitive with `cis_sep` here (17/19 vs 18/19 unsafe flagged, 3/12 vs 10/12 safe flagged); on Gemma-3 its refusal matcher rates the aligned model itself at 0.34, so its Gemma alarms are uninformative.
-- **Repairs:** several DPO-recovered models remain unsafe by our measure (78 prompts, one judge); these are our numbers,
+- **Repairs:** several DPO-recovered models remain unsafe by our measure (78 prompts, Llama Guard 3 1B); these are our numbers,
   not Lexsi's evaluation.
@@ -169,2 +177,3 @@
 | llama31-8b-medical-dpo-recover | 0.474 | **yes** | 0.410 | FAIL | 0.138 | 0.88 |
+
 Full tables and ‖ΔW‖: [`docs/report/h6_safetune.md`](https://github.com/PxA-Labs/safetypulse/blob/main/docs/report/h6_safetune.md).
```

Sources: the Job 13 numbers come from `docs/report/specificity_control.json` (`h6.*.auroc`, `h6.*.auroc_by_family.gemma3-4b`,
`n.h6_models` = 31, `n.h6_unsafe` = 19). The Job 14 numbers come from `docs/report/judge8b_h6.json`. The paper (appendix
"Specificity control") notes that Job 13's `cis_sep` was recomputed with batch size 16, which moves Gemma's AUROC from
0.50 to 0.62. The card quotes Job 13's own value, 0.62. Job 11 (`cis_sep` at every layer) is not hosted on the Hub; its
results are only in git (`kaggle/executed/12_hf_jobs/files/job11_*.jsonl`, `docs/report/h6_layers.md`), so the card
does not mention it. Hosting it would be a separate decision.

The MANIFEST needs no change: it does not list `README.md`, and all 25 listed files match.

### P4 · Optional tags (only if the user wants a tag on the corrected cards)

- `safetypulse-h6-safetune`: after P3, create `h6-v1.4` at the new head ("card: Job 13 paragraph, table fix; data
  identical to h6-v1.3"), or leave `h6-v1.3` as the data version. **Do not move existing tags.**
- `safetypulse-h2-*`: after P1/P2, optionally create `h2-v1.2` ("card: HF Job 10 sentence; weights and data identical
  to h2-v1.1").

### P5 · `docs/10_KAGGLE_AND_HF_RESOURCES.md` (not owned by this lane; text proposal for its owner)

- Stage 2 table, H6 row, Tags column: `` `h6-v1` → `h6-v1.3` (current) `` →
  `` `h6-v1`, `h6-v1.2` (Job 13), `h6-v1.3` (Job 14, current); there is no `h6-v1.1` ``. The Hub has exactly these
  three tags.
- Public repos table: `purvansh01/tiny-aya-global-safetypulse-repaired-nf4` is public and in the model collection
  ("The repaired model in 4-bit NF4 (2.5 GB), with a paired audit"), but the table does not list it. Add a row, or
  confirm that it is documented elsewhere.

### P6 · Paper link (after publication; no change now)

No Stage 1 or Stage 2 card links the paper, because it is not public. Once the paper is on arXiv or in the ACL
Anthology, add it to each card's **Links** line (`docs/hub_cards/stage2/*.md`) and to the `@software` citation block.
The SafeTune citation in the H6 card matches the BibTeX that the SafeTune README asks for (EMNLP 2026 System
Demonstrations; checked 2026-11-22). The arXiv entry 2609.22153 (v1, 26 Aug 2026) lists the same title and authors but
no venue. Optionally add `eprint = {2609.22153}` to that entry.

## Stage 1 public repos (no git card copies exist; observations only)

All 10 are public and untagged; `docs/10` claims no tags for them. Licences: CC BY-NC 4.0 on the models (matching
Tiny Aya's `cc-by-nc-4.0` and Cohere's C4AI licence gate), `other` on `safetypulse-eval`, and
`mit-and-cc-by-nc-4.0` on `safetypulse-pairs`. The model cards have `base_model: CohereLabs/tiny-aya-global` but no
`base_model_relation: adapter` (the Stage 2 cards do have it). That is a minor metadata gap and is not proposed here,
because those cards have no git source in `docs/hub_cards`. `safetypulse-eval` has `license: other` without a
`license_name`, which is also minor. The `safetypulse-lora-none` README could not be downloaded (repeated network
timeouts during the audit), so that one card was not checked.
