# l10 — Citation audit of paper/custom.bib and every claim attributed to a citation

Lane 10, round 1 of the workflow, 2026-11-22. Audit only: no HF job, no change to `paper/main.tex`,
`paper/custom.bib` or `paper/make_bib.py`. Everything below is a **proposal** with the exact old and new text.

## Method and sources

- Bib file audited: `paper/custom.bib` on `origin/stage2/wf-integration` (commit 4ef5645), 39 entries
  (34 generated from arXiv by `paper/make_bib.py`, 5 written by hand: `taori2023alpaca`, `wilson1927probable`,
  `cohen1960kappa`, `vanmiltenburg2021preregistering`, `nosek2018preregistration`; `vanmiltenburg` also has an
  arXiv id, so 35 entries carry an arXiv id).
- Metadata: one query of the arXiv API (`https://export.arxiv.org/api/query?id_list=<all 35 ids>`) returned every id;
  title, author list (full list compared name by name after removing TeX accents) and v1 year were compared with
  each bib entry by a script. One abstract page was also fetched directly as a spot check
  (https://arxiv.org/abs/2610.38001: same title, authors, v1 27 Oct 2026).
- Non-arXiv entries: Crossref API (`https://api.crossref.org/works/<doi>`) for the three DOIs, the ACL Anthology page
  for van Miltenburg et al., and the citation block of the Stanford Alpaca README.
- Claims: each sentence of `paper/main.tex` that cites something was checked against the arXiv abstract; where the
  abstract did not settle it, the full text (arXiv HTML or PDF) was read for the specific passage, named below.
- Rendered references: text extracted from the committed `paper/main.pdf` (9 pages) with pypdf.

## Summary

- **Metadata: 39/39 entries VERIFIED.** No wrong arXiv id, title, author list, first author or year. The only
  name differences are TeX encodings of the same letter (`Rystr{\o}m` = Rystrøm, `{\c C}elebi` = Çelebi).
- **Cite keys:** all 32 distinct keys cited in `main.tex` (`\citep` x35, `\citealp` x2) exist in `custom.bib`.
  7 bib entries are not cited: `rottger2024xstest`, `lee2026relocates`, `yu2026safeseek`, `syed2023attribution`
  (harmless: BibTeX only prints cited entries) and `wilson1927probable`, although `main.tex` line 159 uses
  "Wilson 95\% intervals" without citing it (edit E5).
- **Attributed claims: 37 (claim, key) rows checked.** 31 SUPPORTED; **2 CLAIM UNSUPPORTED / OVERREACH** (E1: `guo2026routing`
  cited for "benign fine-tuning moves [the refusal direction]", but its abstract says internal safety representations
  are *preserved*; E2: `duan2026monitors` cited as "consistent with H6", but Duan finds refusal-compliance probes
  comparatively stable and studies frozen probes across updates of one model, not frozen thresholds across
  families); **4 wording imprecisions** (E3 `nghiem2026trait` "classifier"; E4 `hsu2024safelora` "constrain tuning";
  E6 `wei2024brittleness` as a benign-fine-tuning source; E9 Llama Guard 3 1B cited to the original Llama Guard).
- **Rendered reference list (FIELD ERROR, cosmetic):** the ACL style lower-cases titles, so `main.pdf` prints
  "The llama 3 herd of models", "Llama guard", "Tiny aya", "Stanford alpaca" and lower case after a question mark
  ("…updates? benchmarking", "…safe data? identifying", "…abliterated? a two-signal"). Fix in `make_bib.py` (E7).
- **Optional venue notes (E8):** seven arXiv-only entries have a peer-reviewed venue stated on arXiv or in the PDF.
- **docs/12_RELATED_WORK.md:** agrees on every entry it covers, and its own Guo et al. and Duan rows support E1 and
  E2; it does not cover 9 cited works (E10).

## 1. Every bib entry

Status: VERIFIED = arXiv/publisher metadata match the bib entry (title, full author list, year, id).
Venue text is quoted from the arXiv "comments"/"journal_ref" field where one exists.

| # | Key | Status | URL | Evidence (title; author count, first author; arXiv v1 date) |
|---|---|---|---|---|
| 1 | `qi2023finetuning` | VERIFIED | https://arxiv.org/abs/2310.03693 | "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!"; 7 authors, first Xiangyu Qi; v1 2023-10-05 (latest 2310.03693v1) |
| 2 | `arditi2024refusal` | VERIFIED | https://arxiv.org/abs/2406.11717 | "Refusal in Language Models Is Mediated by a Single Direction"; 7 authors, first Andy Arditi; v1 2024-06-17 (latest 2406.11717v3) |
| 3 | `wei2024brittleness` | VERIFIED | https://arxiv.org/abs/2402.05162 | "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications"; 9 authors, first Boyi Wei; v1 2024-02-07 (latest 2402.05162v4) |
| 4 | `mazeika2024harmbench` | VERIFIED | https://arxiv.org/abs/2402.04249 | "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"; 12 authors, first Mantas Mazeika; v1 2024-02-06 (latest 2402.04249v2) |
| 5 | `rottger2024xstest` | VERIFIED (not cited in main.tex) | https://arxiv.org/abs/2308.01263 | "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models"; 6 authors, first Paul Röttger; v1 2023-08-02 (latest 2308.01263v3); arXiv: Accepted at NAACL 2024 (Main Conference) |
| 6 | `seth2026circuitkit` | VERIFIED | https://arxiv.org/abs/2607.19317 | "CircuitKIT : Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability"; 4 authors, first Pratinav Seth; v1 2026-07-21 (latest 2607.19317v1) |
| 7 | `nghiem2026trait` | VERIFIED | https://arxiv.org/abs/2606.07631 | "Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning"; 4 authors, first Huy Nghiem; v1 2026-05-31 (latest 2606.07631v2); arXiv: Second version, 40 pages, updated methodology and results; COLM AIW 2026 workshop |
| 8 | `du2025procon` | VERIFIED | https://arxiv.org/abs/2509.06795 | "Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint"; 9 authors, first Yanrui Du; v1 2025-09-08 (latest 2509.06795v1) |
| 9 | `guo2026routing` | VERIFIED | https://arxiv.org/abs/2609.01455 | "When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning"; 5 authors, first Yitong Guo; v1 2026-09-01 (latest 2609.01455v1); arXiv: Accepted to the Findings of EMNLP 2026 |
| 10 | `lee2026relocates` | VERIFIED (not cited in main.tex) | https://arxiv.org/abs/2610.00320 | "Refusal Localizes, the Damage Relocates: Safety Layers Under Few-Sample Fine-Tuning"; 6 authors, first Jungseob Lee; v1 2026-09-29 (latest 2610.00320v1) |
| 11 | `hsu2024safelora` | VERIFIED | https://arxiv.org/abs/2405.16833 | "Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models"; 6 authors, first Chia-Yi Hsu; v1 2024-05-27 (latest 2405.16833v2); arXiv: This is the camera-ready version accepted for NeurIPS 2024 |
| 12 | `yu2026safeseek` | VERIFIED (not cited in main.tex) | https://arxiv.org/abs/2603.23268 | "SafeSeek: Universal Attribution of Safety Circuits in Language Models"; 9 authors, first Miao Yu; v1 2026-03-24 (latest 2603.23268v1) |
| 13 | `ruellan2026replay` | VERIFIED | https://arxiv.org/abs/2610.38001 | "Preserving Safety During Sequential Fine-Tuning: A Factorial and Geometric Analysis of Replay"; 2 authors, first George Ruellan; v1 2026-10-27 (latest 2610.38001v1) |
| 14 | `hawkins2026multilingual` | VERIFIED | https://arxiv.org/abs/2606.28843 | "The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning"; 11 authors, first Will Hawkins; v1 2026-06-27 (latest 2606.28843v1); arXiv: Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026 |
| 15 | `he2024safedata` | VERIFIED | https://arxiv.org/abs/2404.01099 | "What is in Your Safe Data? Identifying Benign Data that Breaks Safety"; 3 authors, first Luxi He; v1 2024-04-01 (latest 2404.01099v2) |
| 16 | `duan2026monitors` | VERIFIED | https://arxiv.org/abs/2606.15980 | "Do Activation Monitors Survive Model Updates? Benchmarking, Predicting, and Repairing Activation-Monitor Staleness"; 1 author, first Evan Duan; v1 2026-06-14 (latest 2606.15980v2) |
| 17 | `hu2021lora` | VERIFIED | https://arxiv.org/abs/2106.09685 | "LoRA: Low-Rank Adaptation of Large Language Models"; 8 authors, first Edward J. Hu; v1 2021-06-17 (latest 2106.09685v2) |
| 18 | `inan2023llamaguard` | VERIFIED | https://arxiv.org/abs/2312.06674 | "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations"; 11 authors, first Hakan Inan; v1 2023-12-07 (latest 2312.06674v1) |
| 19 | `grattafiori2024llama3` | VERIFIED | https://arxiv.org/abs/2407.21783 | "The Llama 3 Herd of Models"; 561 authors, first Aaron Grattafiori; v1 2024-07-31 (latest 2407.21783v3) |
| 20 | `syed2023attribution` | VERIFIED (not cited in main.tex) | https://arxiv.org/abs/2310.10348 | "Attribution Patching Outperforms Automated Circuit Discovery"; 3 authors, first Aaquib Syed; v1 2023-10-16 (latest 2310.10348v2); arXiv: 6 main paper pages, 6 additional pages. NeurIPS 2023 ATTRIB Workshop |
| 21 | `salamanca2026tinyaya` | VERIFIED | https://arxiv.org/abs/2603.11510 | "Tiny Aya: Bridging Scale and Multilingual Depth"; 26 authors, first Alejandro R. Salamanca; v1 2026-03-12 (latest 2603.11510v1) |
| 22 | `seth2026safetune` | VERIFIED | https://arxiv.org/abs/2609.22153 | "SafeTune: A Unified Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned LLMs"; 4 authors, first Pratinav Seth; v1 2026-08-26 (latest 2609.22153v1) |
| 23 | `rafailov2023dpo` | VERIFIED | https://arxiv.org/abs/2305.18290 | "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"; 6 authors, first Rafael Rafailov; v1 2023-05-29 (latest 2305.18290v3) |
| 24 | `bai2022hhrlhf` | VERIFIED | https://arxiv.org/abs/2204.05862 | "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"; 31 authors, first Yuntao Bai; v1 2022-04-12 (latest 2204.05862v1) |
| 25 | `yang2025qwen3` | VERIFIED | https://arxiv.org/abs/2505.09388 | "Qwen3 Technical Report"; 60 authors, first An Yang; v1 2025-05-14 (latest 2505.09388v1) |
| 26 | `gemmateam2025gemma3` | VERIFIED | https://arxiv.org/abs/2503.19786 | "Gemma 3 Technical Report"; 216 authors, first  Gemma Team; v1 2025-03-25 (latest 2503.19786v1) |
| 27 | `zou2023advbench` | VERIFIED | https://arxiv.org/abs/2307.15043 | "Universal and Transferable Adversarial Attacks on Aligned Language Models"; 6 authors, first Andy Zou; v1 2023-07-27 (latest 2307.15043v2) |
| 28 | `tsatsos2026probes` | VERIFIED | https://arxiv.org/abs/2610.35586 | "Linear Probes Transfer from Base to Distilled Language Models despite Rotation"; 1 author, first Marios Tsatsos; v1 2026-10-26 (latest 2610.35586v1) |
| 29 | `gao2026alarm` | VERIFIED | https://arxiv.org/abs/2610.04575 | "From Probe Scores to Alarm Policies: Operational Validity of Activation Monitors for Language-Model Agents"; 1 author, first Xueping Gao; v1 2026-10-03 (latest 2610.04575v1); arXiv: 12 pages, 1 figure. Accepted at the Eighth International Conference on Distributed Artificial Intelligence (DAI 2026) |
| 30 | `hurtado2026abliterated` | VERIFIED | https://arxiv.org/abs/2607.01854 | "Has This Checkpoint Been Abliterated? A Two-Signal Audit and Its Failure Map"; 1 author, first Gabriel Hurtado; v1 2026-07-02 (latest 2607.01854v2) |
| 31 | `bao2026trace` | VERIFIED | https://arxiv.org/abs/2610.07518 | "Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates"; 4 authors, first Ziqun Bao; v1 2026-10-05 (latest 2610.07518v1) |
| 32 | `vishnubhotla2026grounded` | VERIFIED | https://arxiv.org/abs/2606.03648 | "Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability"; 4 authors, first Krishnapriya Vishnubhotla; v1 2026-06-02 (latest 2606.03648v1) |
| 33 | `zou2023repe` | VERIFIED | https://arxiv.org/abs/2310.01405 | "Representation Engineering: A Top-Down Approach to AI Transparency"; 21 authors, first Andy Zou; v1 2023-10-02 (latest 2310.01405v4) |
| 34 | `peng2024landscape` | VERIFIED | https://arxiv.org/abs/2405.17374 | "Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models"; 4 authors, first ShengYun Peng; v1 2024-05-27 (latest 2405.17374v3); arXiv: NeurIPS'24 |
| 35 | `vanmiltenburg2021preregistering` | VERIFIED | https://arxiv.org/abs/2103.06944 | "Preregistering NLP Research"; 3 authors, first Emiel van Miltenburg; v1 2021-03-11 (latest 2103.06944v2); arXiv: Accepted at NAACL2021; pre-final draft, comments welcome |
| 36 | `taori2023alpaca` | VERIFIED | https://github.com/tatsu-lab/stanford_alpaca | README "Citation" block: `author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto}`, `title = {Stanford Alpaca: An Instruction-following LLaMA model}`, `year = {2023}`; matches (title case: see E7) |
| 37 | `wilson1927probable` | VERIFIED (not cited; see E5) | https://doi.org/10.1080/01621459.1927.10502953 | Crossref: "Probable Inference, the Law of Succession, and Statistical Inference"; Edwin B. Wilson; JASA 22(158) 209-212, 1927 |
| 38 | `cohen1960kappa` | VERIFIED | https://doi.org/10.1177/001316446002000104 | Crossref: "A Coefficient of Agreement for Nominal Scales"; Jacob Cohen; Educational and Psychological Measurement 20(1) 37-46, 1960 |
| 39 | `nosek2018preregistration` | VERIFIED | https://doi.org/10.1073/pnas.1708274114 | Crossref: "The preregistration revolution"; 4 authors, Nosek, Ebersole, DeHaven, Mellor; PNAS 115(11) 2600-2606, 2018 |

`vanmiltenburg2021preregistering` (row 35) also checked on https://aclanthology.org/2021.naacl-main.51/:
"Preregistering NLP research", van Miltenburg, van der Lee, Krahmer, NAACL-HLT 2021, pages 613–623,
DOI 10.18653/v1/2021.naacl-main.51 (bib lacks pages and DOI; optional, E8).

Notes on the 2026 preprints the commander flagged: all 13 (`nghiem2026trait`, `guo2026routing`, `tsatsos2026probes`,
`ruellan2026replay`, `duan2026monitors`, `gao2026alarm`, `hurtado2026abliterated`, `bao2026trace`,
`hawkins2026multilingual`, `vishnubhotla2026grounded`, `seth2026safetune`, `seth2026circuitkit`,
`salamanca2026tinyaya`) resolve on arXiv with the bib's title, authors and year. `nghiem2026trait`: arXiv lists the
last author as "Hal Daumé" (docs/12 writes "Daumé III"); the bib follows arXiv, which is acceptable.
`seth2026circuitkit`: the space in "{CircuitKIT} :" is how arXiv lists the title (cosmetic, optional to drop).

## 2. Every claim attributed to a citation in paper/main.tex

Line numbers are those of `paper/main.tex` at 4ef5645. "Abstract" quotes are verbatim from the arXiv abstract at the
URL in section 1 unless another source is named.

| # | Line | Key | Claim in main.tex | Status | Evidence (verbatim) |
|---|---|---|---|---|---|
| C1 | 48–49 | `qi2023finetuning` | "even benign instruction data can raise a model's compliance with harmful requests" | SUPPORTED | Abstract: "even without malicious intent, simply fine-tuning with benign and commonly used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent" |
| C2 | 48–49 | `he2024safedata` | same sentence | SUPPORTED | Abstract: "just further fine-tuning an aligned model with benign data (i.e., data without harmful content) surprisingly leads to substantial degradation in safety" |
| C3 | 50–51 | `arditi2024refusal` | "Refusal in chat models is mediated largely by a single residual-stream direction" | SUPPORTED | Abstract: "we find a single direction such that erasing this direction from the model's residual stream activations prevents it from refusing harmful instructions" |
| C4 | 51–52 | `du2025procon` | "benign fine-tuning moves it [the refusal direction]" | SUPPORTED | Abstract: "our study reveals that the r-direction tends to drift during training" (IFT; docs/12 row: "Cosine drift of the refusal direction during benign and attack instruction tuning") |
| C5 | 51–52 | `guo2026routing` | same: "benign fine-tuning moves it" | **CLAIM UNSUPPORTED** (E1) | Abstract says the opposite about representations: "The routing view also explains why few safety examples can restore refusal behavior, indicating that internal safety-relevant representations are preserved"; what breaks is "a low-rank output-routing mechanism". docs/12 (full text read): "leaves safety information present … but disrupts its routing to refusal" |
| C6 | 54 | `nghiem2026trait` | "Activation-based drift monitors exist and have been validated on held-out runs" | SUPPORTED | Full text (https://arxiv.org/html/2606.07631v2), Contributions: "Across 468 checkpoints from 36 held-out runs" |
| C7 | 56–57 | `salamanca2026tinyaya` | "Tiny Aya Global, 3.35B" | SUPPORTED | Abstract: "all with just 3.35B parameters. The release includes a pretrained foundation model, a globally balanced instruction-tuned variant, and three region-specialized models" |
| C8 | 57–58 | `nosek2018preregistration`, `vanmiltenburg2021preregistering` | fixed hypotheses with pass marks (preregistration) | SUPPORTED | van Miltenburg abstract: "Preregistration refers to the practice of specifying what you are going to do, and what you expect to find in your study, before carrying out the study."; Nosek: title "The preregistration revolution" (Crossref) |
| C9 | 66 | `zou2023advbench` | "288 AdvBench-derived harmful prompts" | SUPPORTED (source of AdvBench) | Abstract does not name AdvBench; the authors' repo does: https://github.com/llm-attacks/llm-attacks README: "code to reproduce GCG experiments on AdvBench"; `data/advbench/harmful_behaviors.csv` has 520 rows + header |
| C10 | 68–69 | `arditi2024refusal` | direction = difference of mean activations, harmful minus harmless | SUPPORTED | Full text §2.3 (https://arxiv.org/html/2406.11717v3): "we compute the difference between the model's mean activations when run on harmful and harmless instructions" |
| C11 | 86–87 | `mazeika2024harmbench` | HarmBench prompts | SUPPORTED | Abstract: "we introduce HarmBench, a standardized evaluation framework for automated red teaming" |
| C12 | 87–88 | `inan2023llamaguard` | "judged by Llama Guard 3 1B" | SUPPORTED, imprecise (E9, optional) | Abstract describes the first Llama Guard: "Llama Guard, a Llama2-7b model that is instruction-tuned on our collected dataset". Llama Guard 3 1B is described in Fedorov et al., arXiv 2411.17713 (abstract: "comparable or superior safety moderation scores to its larger counterpart, Llama Guard 3-1B") |
| C13 | 100 | `hu2021lora` | "12 LoRA runs" | SUPPORTED | Title "LoRA: Low-Rank Adaptation of Large Language Models" |
| C14 | 101 | `taori2023alpaca` | "Alpaca tuning" | SUPPORTED | Repo README citation block (section 1, row 36) |
| C15 | 173–174 | `he2024safedata` | bulleted-list Alpaca: "a format reported to be especially harmful to safety" | SUPPORTED (full text) | Abstract: "the selected data frequently appear as lists, bullet points, or math questions". Full text (arXiv PDF v2, "Published as a conference paper at COLM 2024"), §5.3: "We reformat the responses for 100 randomly selected instruction-tuning pairs into list and bullet point formats … after reformatting the response, the same fine-tuning and inference process generates response with doubled ASR"; Table 6: GPT ASR 13.0% (random) vs 55.5% (rewritten as lists) |
| C16 | 195 | `seth2026safetune` | "SafeTune releases HH-RLHF-aligned … models, full fine-tunes that drifted … and DPO-recovered versions" | SUPPORTED by the Hub, not by the abstract (see open issue) | Abstract: "a source-available library that unifies four intervention paradigms: post-hoc weight recovery, safety-constrained fine-tuning …"; it does not mention released models. Hub API (anonymous): `Lexsi/llama32-3b-hh-rlhf-aligned` base_model meta-llama/Llama-3.2-3B-Instruct; `Lexsi/qwen3-4b-hh-rlhf-aligned` Qwen/Qwen3-4B; `Lexsi/gemma3-4b-hh-rlhf-aligned` google/gemma-3-4b-it; `Lexsi/llama31-8b-hh-rlhf-aligned` meta-llama/Llama-3.1-8B-Instruct; cards are gated, so "full fine-tunes" and "DPO" were not re-read here (names `-sft-drift` / `-dpo-recover` in docs/13 §3d) |
| C17 | 195 | `bai2022hhrlhf` | "HH-RLHF-aligned" | SUPPORTED (dataset source) | arXiv comments: "Data available at https://github.com/anthropics/hh-rlhf" |
| C18 | 196 | `grattafiori2024llama3`, `yang2025qwen3`, `gemmateam2025gemma3` | model families | SUPPORTED | Titles: "The Llama 3 Herd of Models", "Qwen3 Technical Report", "Gemma 3 Technical Report" |
| C19 | 197–198 | `rafailov2023dpo` | "DPO-recovered versions" | SUPPORTED (method source) | Abstract: "The resulting algorithm, which we call Direct Preference Optimization (DPO)" |
| C20 | 215–216 | `cohen1960kappa` | Cohen's κ | SUPPORTED | Crossref title "A Coefficient of Agreement for Nominal Scales" |
| C21 | 220–221 | `vishnubhotla2026grounded` | "Judges are known to disagree on fine-tuned models' outputs" | SUPPORTED | Abstract: "(2) automated safety judgments are unreliable for such incoherent outputs, and (3) the conclusions about the effects of fine-tuning can change depending on the choice of safety benchmark as well as the safety evaluator" |
| C22 | 226–227 | `gao2026alarm` | "Ranking is not an alarm" | SUPPORTED | Abstract: "Activation probes can predict safety-relevant properties of language models with high … AUROC, but deployed agent monitors make thresholded alarm decisions under tight false-alarm budgets. These are different estimands." |
| C23 | 255 | `qi2023finetuning`, `he2024safedata`, `wei2024brittleness` | "Benign fine-tuning degrades safety" | qi, he SUPPORTED (C1, C2); wei WEAK (E6, optional) | Wei abstract mentions it only as motivation: "susceptibility to jailbreaking and even non-malicious fine-tuning"; its finding is that safety-critical regions are sparse: "comprising about 3% at the parameter level and 2.5% at the rank level. Removing these regions compromises safety" |
| C24 | 256 | `arditi2024refusal`, `du2025procon` | "the refusal direction drifts during tuning" | SUPPORTED | as C3, C4 |
| C25 | 256–257 | `guo2026routing` | "benign tuning disrupts how safety information is routed to refusal" | SUPPORTED | Abstract: "alignment makes the safety geometry flatter while preserving an output-routing pathway. After 100 benign fine-tuning examples, this pathway is selectively re-sharpened"; "safety failure is best understood as a disruption of a low-rank output-routing mechanism" |
| C26 | 257–258 | `zou2023repe` | "Reading concepts from activation directions follows representation engineering" | SUPPORTED | Abstract: "RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena" |
| C27 | 258–259 | `peng2024landscape` | fine-tuning compromises safety once it leaves a "safety basin"; measured by perturbing weights | SUPPORTED | Abstract: "random perturbations to model weights maintain the safety level of the original aligned model within its local neighborhood. However, outside this local region, safety is fully compromised"; "finetuning compromises safety by dragging the model away from the safety basin" |
| C28 | 259–260 | `nghiem2026trait` | "tracks seven trait directions during LoRA tuning of four 7–9B models and validates a classifier on 468 held-out checkpoints" | SUPPORTED except "classifier" (E3) | Abstract: "a fixed coordinate system from seven alignment-relevant activation directions … during LoRA finetuning of four open-source 7-9B language models"; full text §3.2: "468 held-out checkpoints from 36 training runs on the held-out data"; §4.2: "we construct an EM monitor by fitting a regressor from the 7D drift vector"; "We use Ridge, Gradient Boosted Regression (GBR), and Random Forest (RF) regressors"; "The monitor flags a checkpoint when its predicted EM rate exceeds this threshold" |
| C29 | 261–262 | `tsatsos2026probes` | "Harmful-request probes transfer from a base model to its distilled and instruction-tuned successors with little loss of AUC" | SUPPORTED | Abstract: "apply it, unchanged, to two successors of that base: the DeepSeek-R1 distilled reasoning model and the developer's instruction-tuned chat model"; "on XSTest the transferred probe's AUC changes by only −0.011 to +0.027, and on an independent OR-Bench evaluation … by −0.015 to +0.053" |
| C30 | 263–264 | `ruellan2026replay` | "preserving the base refusal direction is not a reliable proxy for safety in every model" | SUPPORTED | Abstract: "This dissociation between models indicates that preserving the base refusal direction is not a reliable proxy for behavioral safety." |
| C31 | 264 | `duan2026monitors` | "frozen probes can go stale after updates …, consistent with H6" | first half SUPPORTED; "consistent with H6" **OVERREACH** (E2) | Abstract: "fine-tuning-style updates frequently make probes stale", but also "Fragility is highly monitor-dependent, with privacy/PII probes most affected and refusal-compliance probes comparatively stable". Duan freezes a probe across updates of one model; H6 re-calibrates the direction on each aligned model and freezes only the thresholds across families (main.tex 198: "We calibrated on each aligned model and scored all 35 models with Global's thresholds") |
| C32 | 265 | `gao2026alarm` | "High AUROC does not make a valid alarm policy" | SUPPORTED | Abstract: "joint activation-observable monitors reach AUROC 0.957 and 0.935 … yet their locked 5%/10% detection rates are only .642/.742 and .719/.782"; "no setting reaches alarm-policy validity" |
| C33 | 265–267 | `hurtado2026abliterated` | "an activation refusal gap combined with a weight-difference signal transfers a threshold across held-out families for abliterated models" | SUPPORTED | Abstract: "We combine two cheap internal signals, a reference-anchored activation refusal-gap and a weight-recovery energy of the base-to-candidate weight difference"; "a Youden-calibrated threshold transfers to held-out families at balanced accuracy 0.89 (FPR 0.11), missing only 4 of 57" ("a much larger change than benign drift" is the paper's own comment, not attributed) |
| C34 | 268 | `bao2026trace` | "the weights of harmful fine-tunes carry a readable trace" | SUPPORTED | Abstract: "harmful-compliance SFT induces a continuous, objective-dependent ordering in checkpoint-update space"; "TRACE, a weights-only auditing method" |
| C35 | 268–269 | `hsu2024safelora`, `du2025procon` | "Prevention methods constrain tuning instead" | du SUPPORTED; hsu imprecise (E4) | Du abstract: "a projection-constrained loss term that regularizes the projection magnitude of each training sample's hidden state onto the r-direction". Hsu abstract: "the projection of LoRA weights from selected layers to the safety-aligned subspace … Safe LoRA is a training-free and data-free approach" (it changes the update after training, it does not constrain tuning) |
| C36 | 269–270 | `seth2026safetune`, `seth2026circuitkit` | "provide the auditing, repair and circuit tools SafetyPulse builds on" | SUPPORTED | SafeTune title: "…for Auditing and Repairing Safety Drift in Fine-Tuned LLMs"; CircuitKIT abstract: "a source-available library that connects the circuit-analysis workflow" |
| C37 | 289–290 | `hawkins2026multilingual` | "benign multilingual fine-tuning moves safety differently per language" | SUPPORTED | Abstract: "safety outcomes are highly sensitive to both the choice of fine-tuning language and the evaluation language"; "Multilingual safety drift … occurs heterogeneously across languages and models" |

(37 rows, split by key where the evidence differs. Every one of the 32 cited keys appears in at least one row.)

## 3. Proposed edits

### E1 — main.tex lines 51–52 (CLAIM UNSUPPORTED for `guo2026routing`)

Old:
```
models is mediated largely by a single residual-stream direction \citep{arditi2024refusal}, and benign fine-tuning moves
it \citep{du2025procon,guo2026routing}, which makes the strength of that direction a natural candidate.
```
New (keeps both citations, each for what it says; +8 words):
```
models is mediated largely by a single residual-stream direction \citep{arditi2024refusal}, and benign fine-tuning moves
it \citep{du2025procon} and disrupts how safety is routed to refusal \citep{guo2026routing}, which makes the strength of
that direction a natural candidate.
```
Zero-length alternative: replace `\citep{du2025procon,guo2026routing}` by `\citep{du2025procon}` (Guo stays cited in
Related work, line 257). Source: C4, C5.

### E2 — main.tex line 264 (OVERREACH for `duan2026monitors`)

Old:
```
\citep{ruellan2026replay}, and frozen probes can go stale after updates \citep{duan2026monitors}, consistent with H6.
```
New:
```
\citep{ruellan2026replay}, and frozen activation probes often go stale after fine-tuning, although refusal probes were
comparatively stable \citep{duan2026monitors}; H6 shows that frozen thresholds can fail across model families.
```
Shorter alternative: `…, and frozen probes can go stale after fine-tuning \citep{duan2026monitors}, a risk that H6
shows for frozen thresholds across families.` Source: C31 (Duan abstract), main.tex line 198 (H6 recalibrates the
direction per aligned model).

### E3 — main.tex line 260 (`nghiem2026trait`: regressor, not classifier)

Old: `7--9B models and validates a classifier on 468 held-out checkpoints \citep{nghiem2026trait}; we test frozen thresholds of`
New: `7--9B models and validates a thresholded regressor on 468 held-out checkpoints \citep{nghiem2026trait}; we test frozen thresholds of`
Source: C28 (full text §4.2). Low priority.

### E4 — main.tex lines 268–269 (`hsu2024safelora`: post-hoc projection)

Old:
```
weights of harmful fine-tunes carry a readable trace \citep{bao2026trace}. Prevention methods constrain tuning instead
\citep{hsu2024safelora,du2025procon}; SafeTune \citep{seth2026safetune} and CircuitKIT \citep{seth2026circuitkit} provide
```
New:
```
weights of harmful fine-tunes carry a readable trace \citep{bao2026trace}. Prevention methods constrain tuning
\citep{du2025procon} or project the update back towards the aligned model \citep{hsu2024safelora}; SafeTune
\citep{seth2026safetune} and CircuitKIT \citep{seth2026circuitkit} provide
```
Source: C35. Low priority.

### E5 — main.tex line 159 (Wilson interval used, entry not cited)

Old: `for Spot. Logit: refusal-token score. Wilson 95\% intervals, e.g.\ H6 false alarms 0.55--0.95 (\cissep{}) vs.\ 0.09--0.53`
New: `for Spot. Logit: refusal-token score. Wilson 95\% intervals \citep{wilson1927probable}, e.g.\ H6 false alarms 0.55--0.95 (\cissep{}) vs.\ 0.09--0.53`
Source: `custom.bib` already holds `wilson1927probable` (row 37, Crossref-verified).

### E6 — main.tex line 255 (optional: `wei2024brittleness`)

Old: `Benign fine-tuning degrades safety \citep{qi2023finetuning,he2024safedata,wei2024brittleness}; the refusal direction`
New: `Benign fine-tuning degrades safety \citep{qi2023finetuning,he2024safedata}, and the weights that carry it are sparse \citep{wei2024brittleness}; the refusal direction`
Source: C23. Optional: the current wording is defensible (Wei's abstract states the premise), but Wei is not evidence for it.

### E7 — paper/make_bib.py (FIELD ERROR in the rendered reference list)

`acl_natbib.bst` lower-cases unbraced title words. `paper/main.pdf` prints: "The llama 3 herd of models",
"Llama guard: LLM-based…", "Tiny aya: Bridging…", "Stanford alpaca: …", "…model updates? benchmarking, …",
"What is in your safe data? identifying …", "Has this checkpoint been abliterated? a two-signal …".

Old (make_bib.py, in the loop that writes each entry):
```
    title = re.sub(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*)\b", r"{\1}", title)   # keep acronyms / CamelCase
```
New:
```
    title = re.sub(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*)\b", r"{\1}", title)   # keep acronyms / CamelCase
    title = re.sub(r"\b(Llama Guard|Llama|Tiny Aya)\b", r"{\1}", title)        # proper names the style lower-cases
    title = re.sub(r"([?!]) ([A-Z])", r"\1 {\2}", title)                         # capital after a question mark
```
And in `MANUAL`:
Old: `  title = {Stanford Alpaca: An Instruction-following {LLaMA} model},`
New: `  title = {Stanford {Alpaca}: An Instruction-following {LLaMA} model},`

Tested on the 35 arXiv titles (scratch script, same regexes): exactly six titles change —
`he2024safedata` "Safe Data? {I}dentifying", `duan2026monitors` "Updates? {B}enchmarking",
`inan2023llamaguard` "{Llama Guard}: {LLM}-based", `grattafiori2024llama3` "The {Llama} 3 Herd",
`salamanca2026tinyaya` "{Tiny Aya}: Bridging", `hurtado2026abliterated` "Abliterated? {A} Two-Signal".
Then rerun `python paper/make_bib.py` and `cd paper && tectonic main.tex`.

### E8 — optional venue notes (all verified as stated; none is an error)

| Key | Venue | Source |
|---|---|---|
| `he2024safedata` | COLM 2024 | arXiv PDF v2 page header "Published as a conference paper at COLM 2024" |
| `guo2026routing` | Findings of EMNLP 2026 | arXiv comments "Accepted to the Findings of EMNLP 2026" |
| `hsu2024safelora` | NeurIPS 2024 | arXiv comments "camera-ready version accepted for NeurIPS 2024" |
| `peng2024landscape` | NeurIPS 2024 | arXiv comments "NeurIPS'24" |
| `hawkins2026multilingual` | ICML 2026 | arXiv journal_ref "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026" |
| `nghiem2026trait` | COLM 2026 AIW workshop | arXiv comments "COLM AIW 2026 workshop" |
| `gao2026alarm` | DAI 2026 | arXiv comments "Accepted at the Eighth International Conference on Distributed Artificial Intelligence (DAI 2026)" |
| `vanmiltenburg2021preregistering` | add `pages = {613--623}`, `doi = {10.18653/v1/2021.naacl-main.51}` | ACL Anthology page |

Proposed make_bib.py change (adds a `note` field; `acl_natbib.bst` prints `note` for `@misc`):
after the `ARXIV = {...}` dict add
```
VENUE = {"he2024safedata": "COLM 2024", "guo2026routing": "Findings of EMNLP 2026",
         "hsu2024safelora": "NeurIPS 2024", "peng2024landscape": "NeurIPS 2024",
         "hawkins2026multilingual": "ICML 2026"}
```
and in the f-string that writes each entry, after `url = {{https://arxiv.org/abs/{aid}}}` insert
`+ (f",\n  note = {{{VENUE[key]}}}" if key in VENUE else "")` before the closing `\n}}`. Workshop and DAI notes are
left out on purpose (less useful to readers); add them if wanted. Not tested against the .bst output here.

### E9 — optional: Llama Guard 3 1B source

`inan2023llamaguard` describes the original Llama-2-7B Llama Guard. If a source for the 1B model is wanted, arXiv
2411.17713 (Fedorov et al., "Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations",
20 authors, v1 2024-11-18) mentions Llama Guard 3-1B as the larger counterpart of the INT4 model. Proposal: add key
`fedorov2024llamaguard3` = `2411.17713` to `ARXIV` in make_bib.py and change main.tex line 88
`\citep{inan2023llamaguard}` -> `\citep{inan2023llamaguard,fedorov2024llamaguard3}`. Optional; not needed for correctness.

### E10 — docs/12_RELATED_WORK.md agreement

Agreement: every entry docs/12 covers matches the bib and the paper's usage (Nghiem: "468 held-out checkpoints",
"random-forest monitor" — which supports E3; Guo: "leaves safety information present … but disrupts its routing to
refusal" — which supports E1; Duan: "refusal probes are comparatively stable" — which supports E2; He et al.:
"often lists, bullet points or math questions"; Hawkins; Ruellan; Hsu "NeurIPS 2024"; XSTest "NAACL 2024").

Gap: docs/12 has no row for 9 works the paper cites in Related work or Results: `tsatsos2026probes`, `gao2026alarm`,
`hurtado2026abliterated`, `bao2026trace`, `vishnubhotla2026grounded` (make_bib.py: "added after the related-work
check (abstract pages verified 2026-11-21)"), `zou2023repe`, `peng2024landscape` ("suggested in an internal review"),
and the toolkits `seth2026safetune` (docs/12 mentions SafeTune only as the restore method) and `salamanca2026tinyaya`.
Proposed addendum for docs/12 (append after the 2026-11-13 table):

```
### Added 2026-11-22 (citation audit, lane 10; checked against the arXiv abstracts)

| Work | What it does | Relation to SafetyPulse |
|---|---|---|
| Tsatsos, *Linear Probes Transfer from Base to Distilled Language Models despite Rotation*, arXiv [2610.35586](https://arxiv.org/abs/2610.35586) | XSTest-trained harmful-request probes on four base models transfer unchanged to distilled and instruction-tuned successors; AUC changes −0.011 to +0.027 despite 17°–35° rotation | Within-family transfer, as in H1 |
| Gao, *From Probe Scores to Alarm Policies*, arXiv [2610.04575](https://arxiv.org/abs/2610.04575) (DAI 2026, per arXiv) | High-AUROC activation monitors miss locked detection targets; "no setting reaches alarm-policy validity" | Ranking ≠ alarm; matches our per-family AUROC vs threshold analysis |
| Hurtado, *Has This Checkpoint Been Abliterated?*, arXiv [2607.01854](https://arxiv.org/abs/2607.01854) | Activation refusal-gap + weight-difference energy; threshold transfers to held-out families (balanced acc. 0.89) for abliterations | Whole-checkpoint audit of a much larger change than benign drift |
| Bao et al., *Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates*, arXiv [2610.07518](https://arxiv.org/abs/2610.07518) | Weights-only TRACE score of harmful-objective SFT | Weight-space audit; ours reads activations |
| Vishnubhotla et al., *Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability*, arXiv [2606.03648](https://arxiv.org/abs/2606.03648) | Automated safety judges are unreliable on incoherent outputs; conclusions depend on the evaluator | Why we re-judged with Llama Guard 3 8B |
| Zou et al., *Representation Engineering*, arXiv [2310.01405](https://arxiv.org/abs/2310.01405) | Monitoring and manipulating concepts via population-level representations | Background for reading a direction |
| Peng et al., *Navigating the Safety Landscape*, arXiv [2405.17374](https://arxiv.org/abs/2405.17374) (NeurIPS 2024) | "Safety basin" in weight space; VISAGE metric by weight perturbation | Weight-space risk measure; ours is activation-based |
```

## Open issues

- C16: the SafeTune model cards are gated; this audit confirmed the four aligned repos and their base models from
  the public Hub API but did not re-read the cards for "full fine-tunes" and "DPO-recovered". The team's H6 records
  (docs/13 §3d, kaggle/executed H6 runs) are the source for those words; worth one look by someone with access.
- E8 make_bib.py `note` patch was not run through BibTeX/tectonic.
- Venue information for older entries (e.g. Qi et al., Arditi et al., HarmBench, DPO, LoRA) was not added because
  arXiv's metadata for them states no venue; adding venues from memory would be unverified.
