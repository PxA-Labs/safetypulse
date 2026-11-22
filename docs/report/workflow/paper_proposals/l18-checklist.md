# Lane 18 (round 2): ACL Responsible NLP Research checklist, answered with evidence

Audit of `paper/main.tex` as on `origin/stage2/wf-integration` @ `eb262cd` (paper text identical to `origin/main` @
`eb07db8`). Line numbers `main.tex:N` refer to that file. `paper/main.tex` is **not** edited. No HF job (GPU cap $0).
Exact text edits that close the gaps below are in `l18-limitations-ethics.md` (same folder); this file only answers
the checklist and says where each answer is supported.

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

| # | Gap | Checklist | Severity | Fix in `l18-limitations-ethics.md` |
|---|---|---|---|---|
| G1 | The harmless twins of the probe/calibration pairs were **drafted by an AI assistant and reviewed by AI reviewer agents, with no human check** (`kaggle/executed/02_draft_harmless_twins/RUN_SUMMARY.md`, "report them as AI-drafted and AI-reviewed"; `pairs_reviewed.csv`: `reviewer = claude` on 300/300 rows, `annotator_2_valid` empty on 300/300). The paper says only "a minimally edited harmless twin" (`main.tex:66-67`) | E1, B5, A1 | high (ACL policy and the project's own record) | X1, X2 |
| G2 | AI-assistance disclosure is in the Ethics Statement (`main.tex:304-306`), not in Acknowledgements as ACL §4.2.2 asks, and it does not say where AI was used. It also does not cover the multi-agent workflow that proposed and ran the 2026-11-22 exploratory follow-ups (`docs/report/workflow/LEDGER.md`), if any of those enter the paper | E1 | high | X2 |
| G3 | Licences: only SafeTune's LSAL v1.2 is named (`main.tex:304`). Not stated: Tiny Aya Global/Earth (CC BY-NC 4.0), Llama Guard 3 (Llama 3.2 / 3.1 community licences), the SafeTune base families (Llama, Gemma, Qwen3), Alpaca (CC BY-NC 4.0), AdvBench / HarmBench / GSM8K (MIT), LLM-LAT harmful-dataset (no licence on its card), and the licences of what we release (code Apache-2.0; adapters and data CC BY-NC 4.0) | B2, B3 | medium | X4 |
| G4 | Total compute: the paper gives only "each H6 family took 9--22 minutes on one L40S" (`main.tex:369`). No total GPU hours or cost | C1 | medium | X5 |
| G5 | Training hyperparameters beyond "LoRA ($r=16$, 2,000 examples, 250 steps)" (`main.tex:112-113`) are not given: alpha 32, dropout 0.05, all attention + MLP projections, 2 epochs, lr 2e-4 cosine, warm-up 3%, effective batch 16. Library versions are not given | C2, C4 | medium | X6 |
| G6 | Ethics says "published logs contain verdicts and scores only" (`main.tex:303`). True for Stage 2, **not for Stage 1**: `docs/11_WITHHELD_ARTIFACTS.md` §4 says Stage 1 logs publish "the text of all answers judged safe" (harmful ones replaced by `[withheld: judged harmful]`) | B4, A2 | medium (a factual inaccuracy) | X3 |
| G7 | Risks: no dual-use statement (public bundles contain refusal directions, which are also what refusal ablation uses; `docs/08_RISK_ETHICS_LICENSING.md` §1 asks the report to discuss this) and no false-assurance statement (a PASS is not a safety certificate, H6 refuted) | A2 | medium | X3 |
| G8 | Two datasets used are not cited: GSM8K (`main.tex:82,101`; Cobbe et al. 2021, arXiv:2110.14168, verified) and the LLM-LAT refusal data used in the control (`main.tex:83,169`; Hub card has no paper and no licence: cite by URL). No model or dataset revision is given in the paper (they are in the Hub cards and docs/10) | B1 | low-medium | X7 |
| G9 | No human annotation was done; the paper says the human check was not run (`main.tex:296-297`) but does not say the pairs had no human check either (G1) | D1-D5 | low (D is N/A) | X1 |

## A. For every submission

**A1. Did you describe the limitations of your work?** **Yes** - `main.tex:283-298` (Limitations: small samples and
dependent checkpoints, label noise, English only, specificity, Gemma ablation, H6 endpoints, not-run items, post-hoc
choices); also Results "Exploratory" paragraphs `main.tex:223-251`.
Evidence: sources of every number in the Limitations are traced in `l09-number-audit.md` (one count mismatch, lane 9 E4,
"6--11" should be "6--10" fine-tuned models).
Gap: G1 (AI-drafted, AI-reviewed pairs with no human check is not listed as a limitation). The "Not run" sentence
(`main.tex:296-297`) is now out of date: H4, H5 and a third judge ran on 2026-11-22 as exploratory (amendments
`edda1dd`, `3fc67ac`, `baeb43e`). Proposed text: `l18-limitations-ethics.md` X1.

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
| ShieldGemma 9B (Job 17, exploratory) | gemma (gated) | Hub API |
| NLLB-200 distilled 600M (Job 20, exploratory) | cc-by-nc-4.0 | amendment `edda1dd` (`docs/prereg_amendments/2026-11-22-l06-h4-multilingual.md`) |
| Our code / adapters and data | Apache-2.0 / CC BY-NC 4.0 | `README.md:233-234`; `docs/adr/0004-licensing.md:23` |
| Lexsi toolkits (SafeTune, CircuitKIT, AuditKIT, CuratorKIT) | LSAL v1.2 | `kaggle/executed/00_environment_and_toolkits/files/toolkit_apis.json:125,362,603` |

Note: `docs/08_RISK_ETHICS_LICENSING.md:42` still says CircuitKIT is "LSAL v1.1 ... confirm"; the installed toolkit metadata
says LSAL-1.2. The paper should say 1.2 (it does, for SafeTune).
Gap: G3. Proposed: X4.

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

**C4. If you used existing packages ..., did you report the implementation, model, and parameter settings used?**
**Partly** - judge model and decoding (`main.tex:86-88`), refusal string-matching spot-check (`main.tex:94-95`), LoRA
(`main.tex:93,112`), Cohen's kappa and Wilson intervals cited. Library versions are not in the paper: transformers 5.20.2,
peft 0.21.2, torch 2.11.0 (`kaggle/executed/00_environment_and_toolkits/RUN_SUMMARY.md:18`; Hub cards
`docs/hub_cards/stage2/safetypulse-h3-lora-alpaca.md:89`). The SafeTune / CircuitKIT toolkits are cited. Relevant because
re-generated H6 answers did not reproduce all 1B flags (LEDGER, lane 3 row: "1B rerun matches Job 9 flags on 27/35
models"); cause under investigation by lane 11 (Job 22, running). Gap: G5. Proposed: X6.

## D. Did you use human annotators (e.g., crowdworkers) or research with human participants? **No**

**D1-D5** (instructions to participants; recruitment and pay; consent; ethics board approval; annotator demographics):
**N/A.** No human annotators or participants. The planned 100-item human check of the judge was not run (`main.tex:296-297`;
`kaggle/executed/10_audit_and_report/RUN_SUMMARY.md:15`: the sample exists but holds harmful answers and is private).
The pairs had no human annotator either (`kaggle/executed/03_finalize_pairs/RUN_SUMMARY.md`, matched-pair run:
"Second annotator (Cohen's kappa) | none: no human check yet (limitation)"). The checklist answer is N/A, but the paper
should say so (G9, folded into X1).

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

## Checks run for this file

- `curl -sL https://aclrollingreview.org/responsibleNLPresearch/` -> HTTP 200; text extracted with a small Python HTML
  stripper; all question texts above are from it.
- Hub API licences: `curl https://huggingface.co/api/models/<id>` / `api/datasets/<id>` for the 12 ids in the B2 table.
- arXiv: <https://arxiv.org/abs/2110.14168> (Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021-10-27).
- Compute totals: arithmetic over the cited rows; the Kaggle estimate from cell `execution` timestamps of the 79 executed
  notebooks (script kept in the session scratchpad, not committed; logic as described in the C1 table).
