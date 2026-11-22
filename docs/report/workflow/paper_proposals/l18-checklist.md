# Lane 18 (rounds 2-3): ACL Responsible NLP Research checklist, answered with evidence

Round 2 audited `paper/main.tex` on `origin/stage2/wf-integration` @ `eb262cd`; round 3 re-checked it @ `9d99ae4`
(`git log -- paper/main.tex`: last change `6ab84a7`, so the text and the line numbers `main.tex:N` below are unchanged,
and identical to `origin/main` @ `eb07db8`). `paper/main.tex` is **not** edited. No HF job (GPU cap $0). Exact text
edits are in `l18-limitations-ethics.md` (same folder); this file answers the checklist and says where each answer is
supported.

## Round 3 update (2026-11-23 IST): what rounds 2-3 added

Rounds 2-3 collected and analysed the exploratory jobs, and lane 9 (round 3) merged this lane's round-2 edits into its
final patch (`origin/wf/r3-lane9:docs/report/workflow/paper_proposals/CONSOLIDATED.md`, commit `96e8c92`, table
"Proposal -> edit": X1a -> N1, X1c -> N2, X2 -> N6, X3 -> N4, X4 -> N5, X5/X6 -> N7 without totals, X7 -> N8, X1b ->
O14). The answers below are updated for these results; **(r3)** marks every changed answer.

| New since round 2 | Result (exploratory; no verdict or threshold changes) | Source | Checklist items |
|---|---|---|---|
| Third judge, ShieldGemma 9B (Job 17) | agreement with 1B 0.735, kappa 0.259 over 2,730 judgements; H6 refuted under 1B, 8B, ShieldGemma and the 2-of-3 majority; the 1B rerun reproduced Job 9/14 flags on 27/35 models | `docs/report/third_judge_h6.json` (`sg_vs_1b_all_models`, `h6`, `rerun_1b_equals_job14`) | A1, B1, B2, C3, C4, D |
| Test-split labels (Job 16) | on the 119 held-out prompts H6 stays refuted with both judges; 3 (1B) / 5 (8B) of 31 labels change vs dev; 1B-8B kappa 0.784 | `docs/report/h6_testsplit.json` (`h6.1b_test`, `h6.8b_test`, `n_label_changes_vs_dev`, `test_kappa_1b_8b`) | A1, B6, C3 |
| H4 multilingual (Job 20) | English `cis_sep` alarm fired on all 15 later H3 checkpoints, so recall 1.00 in en/hi/es/fr is uninformative; NLLB-200 translation, Llama Guard 3 8B, 119 prompts | `docs/report/h4_multilingual.json` (`alarm_on_every_later_checkpoint`, `per_language`) | A1, B1, B2 |
| H5 localisation (Job 19) | 0 of 72 single-block restores exceed the Wilson half-width (10.6 points); top-k not clearly better than random sets of equal energy | `docs/report/h5_localise.md` §2, `h5_localise.json` (`single`, `topk`) | A1 |
| Determinism (Job 22) | two greedy passes in one job identical on 8/8 Llama-3.2-3B models (0 of 624 answers differ), equal to Job 17 and not Job 14; no H6 label changes; printed torch 2.15.0+cu132, transformers 5.24.0, CUDA 13.2 | `origin/wf/r3-lane11:docs/report/determinism_h6.md` / `.json` (commit `2438f30`, **not merged** at this poll); `docs/report/workflow/determinism_versions.md` | C3, C4 |
| 2-step lead time (Jobs 18, 21) | ‖ΔW‖ alarmed 8 and 6 steps before `cis_sep` WARN (seeds 7, 8) | `origin/wf/r2-lane4:docs/report/h3_fine_lead_seeds.json` (not merged) | C1 |
| Compute | workflow Jobs 15-22: 21,934 s = 6.09 GPU-h, $10.97; Job 23 not launched; Job 24 scheduled, not finished at this poll | `docs/report/workflow/LEDGER.md` ("GPU spend, all workflow jobs"); Job 24: `hf jobs inspect 6b03471f8d268c0fbd228a21` (stage SCHEDULING, `running_secs` none, 2026-11-22 19:51 UTC) | C1 |
| Licences | Hub `license` tags re-read 2026-11-22: ShieldGemma `gemma`; NLLB-200 distilled 600M `cc-by-nc-4.0`; Aya Expanse 8B `cc-by-nc-4.0` (gated, not used: lane 12 blocked); SafeTune checkpoints `other` / `lsal-1.2` | `HfApi.model_info(...).card_data` | B2 |

## Source of the questions

- **Checklist:** "Responsible NLP Research" page of ACL Rolling Review,
  <https://aclrollingreview.org/responsibleNLPresearch/>, fetched 2026-11-22 with `curl` (HTTP 200). The page says it was
  "Updated for ARR October 2024 cycle by Anna Rogers" and that the checklist "is completed as part of the submission
  form" (since February 2024). Question ids and wording below are copied from that page.
- **Format rules:** `docs/Instructions_for_EMNLP_2023_Proceedings.pdf` (local, untracked, read only), p. 2-3:
  - Limitations: "This section should occur after the conclusion, but before the references. It will not count towards
    the page limit. The discussion of limitations is mandatory. Papers without a limitation section will be
    desk-rejected without review." It also says "just mentioning that a set of results have been shown for English only
    probably does not reflect what we expect".
  - Ethics Statement: "We encourage all authors to include an explicit ethics statement ... after the conclusion but
    before the references. The ethics statement will not count toward the page limit (8 pages for long, 4 pages for
    short papers)."
  - Paper status: Limitations (`main.tex:283-298`) and Ethics Statement (`main.tex:300-306`) are both present, unnumbered,
    after the Conclusion and before `\bibliography` (`main.tex:313`). OK. So the edits proposed in
    `l18-limitations-ethics.md` cost no body space.
- **ACL policy on AI assistance** (needed for E1): "ACL Policy on Publication Ethics", §4.2.2 "Guidelines for Generative
  Assistance in Authorship", <https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics>. `curl` got
  HTTP 418 (bot block); read through the WebFetch tool on 2026-11-22 (page revision oldid=77137). Quoted fragments:
  use of generative AI to create content "should be fully disclosed in the Acknowledgements section" (example: "Section 3
  was written with inputs from ChatGPT."); for low-novelty text "Authors should specify where such automatically
  generated text was used"; for new ideas "The authors should disclose if models were used in this manner"; "Authors
  are responsible for all content submitted." **Verify these quotes in a browser before relying on them**, since the
  tool returns extracts, not the raw page.

Whether the hackathon judges ask for this checklist at all is not stated in the repo (`docs/00_HACKATHON_BRIEF.md` predates
Stage 2). The answers are still the fastest way to find what a reviewer would ask about.

## Summary of gaps

Status column added in round 3: "lane 9 Nx" = closed by that edit of lane 9's final patch if it is applied.

| # | Gap | Checklist | Severity | Fix in `l18-limitations-ethics.md` |
|---|---|---|---|---|
| G10 (r3) | The licence sentence (lane 9 N5 = X4) names no licence for the two artifacts that lane 9's N1 now cites: ShieldGemma 9B (Gemma terms) and NLLB-200 (CC BY-NC 4.0) | B2 | medium | X4b (new) |
| G11 (r3) | Compute totals: lane 9 left them out as an owner item ("Recompute at freeze"); this lane recomputed them through Job 22; Job 24 is still to be filled | C1 | medium | X5b (new) |
| G12 (r3) | Generation determinism: lane 9's N2 says dev and test labels come "from separate jobs, whose library versions may differ"; Job 22 shows greedy answers differ across jobs even with the same versions. Lane 11's optional Edit 1 states this | C3, C4 | low | X8 (new; defers to lane 11) |
| G1 | The harmless twins of the probe/calibration pairs were **drafted by an AI assistant and reviewed by AI reviewer agents, with no human check** (`kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md`, "report them as AI-drafted and AI-reviewed"; `pairs_reviewed.csv`: `reviewer = claude` on 300/300 rows, `annotator_2_valid` empty on 300/300). The paper says only "a minimally edited harmless twin" (`main.tex:66-67`) | E1, B5, A1 | high (ACL policy and the project's own record) | X1, X2 |
| G2 | AI-assistance disclosure is in the Ethics Statement (`main.tex:304-306`), not in Acknowledgements as ACL §4.2.2 asks, and it does not say where AI was used. It also does not cover the multi-agent workflow that proposed and ran the 2026-11-22 exploratory follow-ups (`docs/report/workflow/LEDGER.md`), if any of those enter the paper | E1 | high | X2 |
| G3 | Licences: only SafeTune's LSAL v1.2 is named (`main.tex:304`). Not stated: Tiny Aya Global/Earth (CC BY-NC 4.0), Llama Guard 3 (Llama 3.2 / 3.1 community licences), the SafeTune base families (Llama, Gemma, Qwen3), Alpaca (CC BY-NC 4.0), AdvBench / HarmBench / GSM8K (MIT), LLM-LAT harmful-dataset (no licence on its card), and the licences of what we release (code Apache-2.0; adapters and data CC BY-NC 4.0) | B2, B3 | medium | X4 |
| G4 | Total compute: the paper gives only "each H6 family took 9--22 minutes on one L40S" (`main.tex:369`). No total GPU hours or cost | C1 | medium | X5 |
| G5 | Training hyperparameters beyond "LoRA ($r=16$, 2,000 examples, 250 steps)" (`main.tex:112-113`) are not given: alpha 32, dropout 0.05, all attention + MLP projections, 2 epochs, lr 2e-4 cosine, warm-up 3%, effective batch 16. Library versions are not given | C2, C4 | medium | X6 |
| G6 | Ethics says "published logs contain verdicts and scores only" (`main.tex:303`). True for Stage 2, **not for Stage 1**: `docs/11_WITHHELD_ARTIFACTS.md` §4 says Stage 1 logs publish "the text of all answers judged safe" (harmful ones replaced by `[withheld: judged harmful]`) | B4, A2 | medium (a factual inaccuracy) | X3 |
| G7 | Risks: no dual-use statement (public bundles contain refusal directions, which are also what refusal ablation uses; `docs/08_RISK_ETHICS_LICENSING.md` §1 asks the report to discuss this) and no false-assurance statement (a PASS is not a safety certificate, H6 refuted) | A2 | medium | X3 |
| G8 | Two datasets used are not cited: GSM8K (`main.tex:82,101`; Cobbe et al. 2021, arXiv:2110.14168, verified) and the LLM-LAT refusal data used in the control (`main.tex:83,169`; Hub card has no paper and no licence: cite by URL). No model or dataset revision is given in the paper (they are in the Hub cards and docs/10) | B1 | low-medium | X7 |
| G9 | No human annotation was done; the paper says the human check was not run (`main.tex:296-297`) but does not say the pairs had no human check either (G1) | D1-D5 | low (D is N/A) | X1 |

**Round-3 status of G1-G9** (all still open in `main.tex`; each closes if lane 9's edit is applied): G1, G9 -> lane 9
N1 (and optional O14 = X1b); G2 -> N6 (conservative wording, see lane 9 §5 item 1); G3 -> N5, plus G10/X4b; G4 ->
G11/X5b; G5 -> N7; G6, G7 -> N4 (lane 9 left out X3's PII sentence as an owner item, §5 item 3); G8 -> N8 (GSM8K) and
N7 (LLM-LAT URL). Source: `origin/wf/r3-lane9:docs/report/workflow/paper_proposals/CONSOLIDATED.md`, edit table and §5.

## A. For every submission

**A1. Did you describe the limitations of your work?** **Yes** - `main.tex:283-298` (Limitations: small samples and
dependent checkpoints, label noise, English only, specificity, Gemma ablation, H6 endpoints, not-run items, post-hoc
choices); also Results "Exploratory" paragraphs `main.tex:223-251`.
Evidence: sources of every number in the Limitations are traced in `l09-number-audit.md` (one count mismatch, lane 9 E4,
"6--11" should be "6--10" fine-tuned models).
Gap: G1 (AI-drafted, AI-reviewed pairs with no human check is not listed as a limitation). The "Not run" sentence
(`main.tex:296-297`) is now out of date: H4, H5 and a third judge ran on 2026-11-22 as exploratory (amendments
`edda1dd`, `3fc67ac`, `baeb43e`). Proposed text: `l18-limitations-ethics.md` X1.
**(r3)** The H4, H5, third-judge and test-split analyses are now merged on the integration branch
(`docs/report/h4_multilingual.*`, `h5_localise.*`, `third_judge_h6.*`, `h6_testsplit.*`), and lane 9's N1 gives each a
result clause and N3 adds the test-split sentence to the body. The limitations these results add, all stated in N1/N3:
the English alarm fired on every later H3 checkpoint, so non-English drift that English misses is untested
(`h4_multilingual.json` `alarm_on_every_later_checkpoint`); machine translation quality was not measured (N1); the
third judge agrees weakly with Llama Guard (kappa 0.259, `third_judge_h6.json`) and is not a human check
(`docs/prereg_amendments/2026-11-22-l03-third-judge.md`); 3 (1B) / 5 (8B) of 31 H6 labels flip on new prompts
(`h6_testsplit.json` `n_label_changes_vs_dev`). Still missing: cross-job non-repeatability of greedy generation (G12,
X8).

**A2. Did you discuss any potential risks of your work?** **Partly** - `main.tex:302-304` (harmful answers and the harmful
positive control not released; third-party checkpoints not redistributed). The Conclusion (`main.tex:279-281`) says a
spot-check should be kept alongside, which is a misuse-of-results caveat, but it is not framed as a risk.
Evidence: `docs/11_WITHHELD_ARTIFACTS.md` §1-§5; `docs/08_RISK_ETHICS_LICENSING.md` §1 (dual use), §2 (harmful content).
Gap: G6, G7. Proposed text: X3.

## B. Did you use or create scientific artifacts? **Yes**

**B1. Did you cite the creators of artifacts you used?** **Yes, with two omissions.**
Cited: Tiny Aya (`salamanca2026tinyaya`, `main.tex:57`); AdvBench (`zou2023advbench`, `main.tex:66`); HarmBench
(`mazeika2024harmbench`, `main.tex:87`); Llama Guard (`inan2023llamaguard`, `main.tex:87`); LoRA (`hu2021lora`,
`main.tex:100`); Alpaca (`taori2023alpaca`, `main.tex:101`); SafeTune and its models, HH-RLHF, DPO, Llama 3, Qwen3,
Gemma 3 (`main.tex:195-198`); CircuitKIT (`main.tex:269`).
Not cited: GSM8K (used for the development runs, `main.tex:82,101`) and LLM-LAT/harmful-dataset (the 200 refusals in the
control, `main.tex:83,169`; source in `docs/article/ARTICLE.md:307`). Model revisions are not in the paper; they are in
`docs/10_KAGGLE_AND_HF_RESOURCES.md:24` (Tiny Aya Global `af89d219`, Llama Guard 3 1B `acf7aafa`) and the Hub cards
(Tiny Aya Earth `0624fafb`, `docs/hub_cards/stage2/safetypulse-earth-lora-alpaca.md:39`). Llama Guard 3 8B used "Hub
default" in Stage 1 (`docs/article/ARTICLE.md:292`) and revision `7327bd9f` in Job 20 (amendment `edda1dd`).
Gap: G8. Proposed: X7.
**(r3)** Artifacts added by the round-2 results that lane 9's N1 cites: ShieldGemma 9B (`google/shieldgemma-9b` @
`b8b63601`, `third_judge_h6.json` `judge_revisions`) and NLLB-200 distilled 600M (Job 20, amendment `edda1dd`). Lane 9's
B3 adds `zeng2024shieldgemma` (arXiv 2407.21772) and `nllbteam2022nllb` (arXiv 2207.04672), metadata from the arXiv
API on 2026-11-23 (CONSOLIDATED.md, row B3); lane 3's final citation audit (`origin/wf/r3-lane3`, commit `6eed52e`)
covers them. This lane did not re-verify those two arXiv entries. Aya Expanse 8B was not used (no job), so it needs no
citation unless lane 12's Variant B is applied.

**B2. Did you discuss the license or terms for use and / or distribution of any artifacts?** **Partly** - `main.tex:303-304`
names LSAL v1.2 for SafeTune only.
Evidence (licence of each artifact; checked on the Hub API `https://huggingface.co/api/{models|datasets}/<id>` on
2026-11-22, and in repo docs):

| Artifact | Licence | Source |
|---|---|---|
| CohereLabs/tiny-aya-global, tiny-aya-earth | cc-by-nc-4.0 (gated) | Hub API; `docs/article/ARTICLE.md:290` |
| meta-llama/Llama-Guard-3-1B / -8B | llama3.2 / llama3.1 (gated) | Hub API; `docs/article/ARTICLE.md:291-292` |
| SafeTune checkpoints (35) | LSAL v1.2 (no redistribution); bases: Llama 3.1/3.2 Community License, Gemma Terms of Use, Qwen3 (Qwen/Qwen3-4B: apache-2.0 on the Hub) | `docs/hub_cards/stage2/safetypulse-h6-safetune.md:181-189`; Hub API |
| AdvBench, HarmBench (walledai mirrors), GSM8K | MIT | Hub API; `docs/adr/0004-licensing.md:23-24` |
| tatsu-lab/alpaca | cc-by-nc-4.0 | Hub API; ARTICLE.md:306 |
| LLM-LAT/harmful-dataset | none stated on the card | Hub API (no `license:` tag); ARTICLE.md:307 |
| ShieldGemma 9B (Job 17, exploratory) | gemma (gated) | Hub API; re-read 2026-11-22 (r3) |
| NLLB-200 distilled 600M (Job 20, exploratory) | cc-by-nc-4.0 | amendment `edda1dd` (`docs/prereg_amendments/2026-11-22-l06-h4-multilingual.md`); Hub API re-read 2026-11-22 (r3) |
| CohereLabs/aya-expanse-8b (planned Job 23, **not run**) | cc-by-nc-4.0 (gated; the account had not accepted the terms, HTTP 403) | Hub API (r3); LEDGER round 2, lane 12 |
| SafeTune checkpoint Hub tag (example `Lexsi/llama32-3b-hh-rlhf-aligned`) | `other`, `license_name: lsal-1.2` | Hub API (r3) |
| `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (ours, private) | cc-by-nc-4.0 | live card front matter (r3) |
| Our code / adapters and data | Apache-2.0 / CC BY-NC 4.0 | `README.md:233-234`; `docs/adr/0004-licensing.md:23` |
| Lexsi toolkits (SafeTune, CircuitKIT, AuditKIT, CuratorKIT) | LSAL v1.2 | `kaggle/executed/00_environment_and_toolkits/files/toolkit_apis.json:125,362,603` |

Note: `docs/08_RISK_ETHICS_LICENSING.md:42` still says CircuitKIT is "LSAL v1.1 ... confirm"; the installed toolkit metadata
says LSAL-1.2. The paper should say 1.2 (it does, for SafeTune).
Gap: G3. Proposed: X4. **(r3)** Lane 9's N5 applies X4; G10 (ShieldGemma, NLLB) is closed by X4b.

**B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use ...? For the artifacts you
create, do you specify intended use ...?** **Partly** - `main.tex:303-304` ("accessed under their licences ... for academic
research and are not redistributed"). Intended use of our released adapters/data (non-commercial research; deployment,
chat use and generating harmful content out of scope) is in the Hub cards (`docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md:44`)
but not in the paper. All our released derivatives are CC BY-NC 4.0, matching the NC terms of Tiny Aya and Alpaca, and
the H6 calibration bundles are LSAL v1.2 derivative material (`safetypulse-h6-safetune` card). Gap: G3 (one clause). Proposed: X4.

**B4. Did you discuss the steps taken to check whether the data ... contains any information that names or uniquely
identifies individual people or offensive content, and the steps taken to protect / anonymize it?** **Partly** -
offensive content is the object of study (harmful prompts from AdvBench/HarmBench); handling is in `main.tex:302-303`.
Evidence: `docs/11_WITHHELD_ARTIFACTS.md` §3-§4 (harmful answers withheld, replaced by `[withheld: judged harmful]`; Stage 2
logs checked to hold no prompt or answer text, "the longest string in any log is a 64-character bundle id"); HF Jobs keep
answers in memory and store flags only (same doc, Stage 2 release). No PII check was run on any dataset (no record in the
repo; `git grep -i "PII|personal data"` finds none); the inputs are public benchmarks of synthetic requests.
Gap: G6 (the "verdicts and scores only" sentence is wrong for Stage 1); a PII statement is missing but low-risk. Proposed: X3.

**B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages ...?** **Yes, partly** - language:
English only (`main.tex:289`); domains of fine-tuning data (Alpaca, GSM8K, list-formatted Alpaca, SafeTune code / Dolly /
GSM8K / legal / medical, `main.tex:82,173,197`); pair construction (`main.tex:66-68`). Full documentation is in the Hub
cards (`docs/hub_cards/stage2/*.md`) and `docs/article/ARTICLE.md:296-313` (sizes, splits, licences, leak checks).
Gap: the pairs' construction (AI-drafted, AI-reviewed, categories) is not documented in the paper (G1). Proposed: X1.

**B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits ...?** **Yes** -
288 pairs split 188 / 50 / 50 (`main.tex:66-68`); 78 HarmBench-derived prompts (`main.tex:86`); 2,000 training examples
per run, 250 steps, checkpoint every 10 steps (`main.tex:112-113`); 47 development checkpoints (`main.tex:83`); 35 H6
models, 4 families (`main.tex:198`, Table 3). Sources traced in `l09-number-audit.md`. The 78 prompts are the dev split;
the 119-prompt test split is used only by exploratory Job 16 (amendment `90350d1`). No gap.
**(r3)** If lane 9's N3 is applied, the body reports test-split labels; it names the split size (119) and says which
split the pre-registered verdict uses, so the split statistics stay complete. H4 also used the 119 test prompts
(`h4_multilingual.json` `n_prompts` 119) and H5 the 78 dev plus 119 held-out prompts (`h5_localise.json`
`n_dev_prompts` 78, `heldout.test:*.n` 119); lane 9's N1/N2 state both.

## C. Did you run computational experiments? **Yes**

**C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and
computing infrastructure used?** **Partly.**
Parameters: Tiny Aya 3.35B (`main.tex:56,167`); SafeTune families by name (3B, 4B, 4B, 8B, `main.tex:196`); judges 1B and 8B
(`main.tex:87,215`). LoRA trainable parameters (30.2 M, 0.89%) are not stated (`kaggle/executed/07_finetune_checkpoints/RUN_SUMMARY.md:37`).
Infrastructure: `main.tex:368-369` (Kaggle T4, HF Jobs A100/L40S). Total budget: **missing** (G4).
Evidence for the total, computed for this lane:

| Block | Runtime | Cost | Source |
|---|---|---|---|
| HF Jobs 1-14 (Stage 1 package checks and Stage 2 runs; GPU) | 7.68 GPU-h (A100 1.52, A10G 0.15, L4 0.39, L40S 5.62) | about $14.55 | per-job running times and costs in `docs/10_KAGGLE_AND_HF_RESOURCES.md:156-171` and `kaggle/executed/12_hf_jobs/RUN_SUMMARY.md:13-14,64-69,89,107,138,153,171,184-188,218-220,228,243`; Job 12 CPU (<$0.01); Job 14's test run is in its $1.27 but not in the hours (its duration is not recorded) |
| of which the paper's Stage 2 jobs 8-14 | 5.62 GPU-h | about $10.21 | same rows |
| HF Jobs 15-20 (workflow round 1, exploratory) | 4.89 GPU-h (L40S) | $8.80 | `docs/report/workflow/LEDGER.md`, "GPU spend" table (`running_secs` x $1.80/h) |
| **All HF Jobs 1-20** | **12.57 GPU-h** | **about $23.35** | sum of the two rows above |
| Kaggle notebook sessions recorded in git | about 72 h wall time (GPU T4 and some CPU sessions) | free | span between first and last cell timestamp of each of the 79 executed notebooks under `kaggle/executed/` (72.71 h; minus 0.71 h for `13_curatorkit_alpaca`, which ran inside HF Job 3); fine-tuning (07) 20.3 h, monitoring (08) 26.7 h, audits (10) 11.0 h. An estimate: sessions not saved to git are missing and CPU sessions are included |

Round 2 jobs (21: seed-8 lead-time replication, lane 10; 22: generation determinism, lane 11) were running on side
branches when this was written and are not counted; add them at freeze. Proposed: X5.

**(r3) Updated totals.** Rows added to the table above:

| Block | Runtime | Cost | Source |
|---|---|---|---|
| HF Jobs 21-22 (workflow round 2, exploratory) | 4,331 s = 1.20 GPU-h (L40S) | $2.17 | `docs/report/workflow/LEDGER.md`, "GPU spend, all workflow jobs (final for rounds 1-2)": Job 21 3,822 s ($1.91), Job 22 509 s ($0.25); cost = s x $1.80/h |
| HF Job 23 (Aya Expanse, lane 12) | 0 | $0 | not launched: gated model (LEDGER, same table) |
| HF Job 24 (seed 9 lead time, lane 10, round 3) | **to be filled** | **to be filled** (at most $2.25: `--timeout 75m`) | record on `origin/wf/r3-lane10` (`f1d2482`): launched 2026-11-22 19:48 UTC; `hf jobs inspect` at 19:51 UTC: stage SCHEDULING, no running time yet |
| **All HF Jobs 1-22** | **13.77 GPU-h** (12.57 + 1.20) | **about $25.52** ($23.35 + $2.17) | sum of the round-2 total and the row above |
| of which workflow Jobs 15-22 | 21,934 s = 6.09 GPU-h | $10.97 | LEDGER, total row |

The 12.57 h / $23.35 base is this lane's round-2 sum (table above), not re-derived in round 3. Add Job 24's
`running_secs` x $1.80/h when lane 10 collects it. Proposed text: X5b.

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?**
**Partly** - threshold selection on development checkpoints (`main.tex:81-84`), layer selection on validation pairs
(`main.tex:69-71`), baselines' thresholds "chosen likewise" (`main.tex:94-98`), LoRA r=16, 2,000 examples, 250 steps
(`main.tex:112-113`), decoding (`main.tex:87`), "never re-tuned" (`main.tex:84`). No hyperparameter search was run for
fine-tuning: one fixed recipe (`docs/13_PREREGISTRATION.md:20`, "recipe of notebook 07 unchanged").
Missing: the rest of the recipe (G5): `kaggle/executed/07_finetune_checkpoints/RUN_SUMMARY.md:37-39` (alpha 32, dropout
0.05, all attention + MLP projections, 2 epochs, lr 2e-4 cosine, warm-up 3%, batch 2 x accumulation 8 = 16).
Proposed: X6.

**C3. Did you report descriptive statistics about your results (e.g., error bars ...), and is it transparent whether you are
reporting the max, mean, etc. or just a single run?** **Yes** - per-run results and the caveat that checkpoints are not
independent (`main.tex:111-112`, App. B `main.tex:336-340`); Wilson 95% intervals (Table 2 caption, `main.tex:159`);
median leads and medians over runs (Table 2, `main.tex:192-193`); margin sensitivity 5/10/15 points (App. B); judge
agreement with kappa (`main.tex:215-220`). Point estimates are stated as such (`main.tex:111`). Lane 9 recommends citing
`wilson1927probable` (S5 in `CONSOLIDATED.md` on `origin/wf/r2-lane9`). No further gap.
**(r3)** Rounds 2-3 add: test-split re-labelling (3 of 31 1B labels and 5 of 31 8B labels change vs dev,
`h6_testsplit.json`), a third judge with agreement and kappa (`third_judge_h6.json`), Wilson intervals for H4 recall
(`h4_multilingual.json` `recall_wilson`) and the H5 single-block noise bound (Wilson half-width 10.6 points,
`h5_localise.md` §2); lane 14's B = 2000 prompt bootstrap (`docs/report/label_noise.json`, amendment `783ba8d`).
Run-to-run variation of generation itself (Job 22): repeatable within a job, not across jobs (G12, X8).

**C4. If you used existing packages ..., did you report the implementation, model, and parameter settings used?**
**Partly** - judge model and decoding (`main.tex:86-88`), refusal string-matching spot-check (`main.tex:94-95`), LoRA
(`main.tex:93,112`), Cohen's kappa and Wilson intervals cited. Library versions are not in the paper: transformers 5.20.2,
peft 0.21.2, torch 2.11.0 (`kaggle/executed/00_environment_and_toolkits/RUN_SUMMARY.md:18`; Hub cards
`docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md:89`). The SafeTune / CircuitKIT toolkits are cited. Relevant because
re-generated H6 answers did not reproduce all 1B flags (LEDGER, lane 3 row: "1B rerun matches Job 9 flags on 27/35
models"); cause under investigation by lane 11 (Job 22, running). Gap: G5. Proposed: X6.
**(r3)** Versions now known: Job 22 printed torch 2.15.0+cu132, transformers 5.24.0, tokenizers 0.23.3, CUDA 13.2,
driver 620.70.06 (`origin/wf/r3-lane11:docs/report/determinism_h6.json` `environment`); Jobs 9, 14 and 17 printed none,
and their torch 2.15.0 / transformers 5.23.0 (Job 9) or 5.24.0 (Jobs 14, 17) are **inferred** from wheel sizes
(`docs/report/workflow/determinism_versions.md`); Jobs 21 and 24 pin torch 2.15.0, transformers 5.20.2, peft 0.21.2
(Job 24 record on `origin/wf/r3-lane10`). Lane 9's N7 states only the Kaggle versions ("Kaggle: transformers 5.20.2,
peft 0.21.2, torch 2.11.0"), which is correct for the fine-tuning; the HF Jobs versions belong with X8 if lane 11's
Edit 1 is used. ShieldGemma setting: unsafe if P(Yes) > 0.5 for any of four guidelines (`third_judge_h6.json`
`sg_threshold` 0.5; `third_judge_h6.md`).

## D. Did you use human annotators (e.g., crowdworkers) or research with human participants? **No**

**D1-D5** (instructions to participants; recruitment and pay; consent; ethics board approval; annotator demographics):
**N/A.** No human annotators or participants. The planned 100-item human check of the judge was not run (`main.tex:296-297`;
`kaggle/executed/10_audit_and_report/RUN_SUMMARY.md:15`: the sample exists but holds harmful answers and is private).
The pairs had no human annotator either (`kaggle/executed/03_finalize_pairs/RUN_SUMMARY.md`, matched-pair run:
"Second annotator (Cohen's kappa) | none: no human check yet (limitation)"). The checklist answer is N/A, but the paper
should say so (G9, folded into X1).
**(r3)** Still N/A: the third judge (Job 17) is a model, and its amendment says it is "not a substitute for the human
check" (`docs/prereg_amendments/2026-11-22-l03-third-judge.md`); no human labelled anything in rounds 2-3 (no amendment
or record mentions annotators; `docs/prereg_amendments/` checked).

## E. Did you use AI assistants (e.g., ChatGPT, Copilot) in your research, coding, or writing? **Yes**

**E1. If you used any AI assistants, did you include information about your use?** **Yes, incompletely** - Ethics
Statement `main.tex:304-306` ("an AI coding assistant helped write code and draft text; the authors designed the study,
checked every number against the logged results and take responsibility for the content").
Evidence of uses the paper does not mention:
1. **Data creation.** The harmless twins used in every `cis_sep` score were drafted by an AI assistant
   (`pairs_draft_v2.csv`, "Claude-drafted, not yet human-checked") and reviewed by three AI reviewer agents
   (`kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md`, "Review done by Claude (2026-11-07) ... Every row carries
   `reviewer = claude`. **Limitation:** no human has checked these pairs; report them as AI-drafted and AI-reviewed").
   The first-pass drafts came from Tiny Aya itself (same file, "What happened"), but the kept twins are the v2 drafts.
2. **Research design and execution.** On 2026-11-22 a multi-agent workflow (`docs/report/workflow/LEDGER.md`) proposed
   exploratory analyses, wrote their amendments (`docs/prereg_amendments/*.md`), launched HF Jobs 15-20 and drafted the
   paper proposals in `docs/report/workflow/paper_proposals/`. If any of these results or texts enter the paper, ACL
   §4.2.2 asks for disclosure of model use for ideas ("The authors should disclose if models were used in this manner").
3. **Where.** ACL §4.2.2 asks for the disclosure "in the Acknowledgements section" and to "specify where such
   automatically generated text was used".
Gap: G1, G2. Proposed: X2 (the authors must confirm the facts they assert in it; see the notes there).
**(r3)** The workflow ran on: rounds 2-3 collected Jobs 16-22, launched Job 24, wrote the analyses and the paper
proposals, and uploaded a card to the private repo `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (LEDGER, round 2
lane table; Job 24 record on `origin/wf/r3-lane10`). Lane 9's N6 says the assistant "carried out the exploratory
follow-ups dated 2026-11-22", which covers this if those results enter the paper; Job 24 was launched on 2026-11-22
UTC (19:48), so the date still holds.

## Checks run for this file

- `curl -sL https://aclrollingreview.org/responsibleNLPresearch/` -> HTTP 200; text extracted with a small Python HTML
  stripper; all question texts above are from it.
- Hub API licences: `curl https://huggingface.co/api/models/<id>` / `api/datasets/<id>` for the 12 ids in the B2 table.
- arXiv: <https://arxiv.org/abs/2110.14168> (Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021-10-27).
- Compute totals: arithmetic over the cited rows; the Kaggle estimate from cell `execution` timestamps of the 79 executed
  notebooks (script kept in the session scratchpad, not committed; logic as described in the C1 table).
