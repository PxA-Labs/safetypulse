# l03 (round 3) — final citation audit: `paper/custom.bib`, `paper/main.tex` and the round-2/3 proposals

Audit only, 2026-11-23. No change to `paper/main.tex`, `paper/custom.bib` or `paper/make_bib.py`; everything
below is a proposal with exact text. No HF job (lane GPU cap $0).

## Scope and method

- Base: `origin/stage2/wf-integration` at `9d99ae4` (`paper/custom.bib` last changed in `6ab84a7`).
- Proposals read: every file in `docs/report/workflow/paper_proposals/` on that base (`CONSOLIDATED.md`,
  `consolidated.patch`, `l01` ... `l18-*`), plus `l04-fine-lead.md` on `origin/wf/r2-lane4` and `l13-intervals.md` on
  `origin/wf/r2-lane13`. Round-3 branches present at the end of this audit: `origin/wf/r3-lane1`, `-lane6`, `-lane7`
  (no proposal file differs from the integration branch) and `-lane8` (older base, only deletions relative to it). None
  adds a citation.
- Keys: every `\cite*{...}` in `main.tex` (comments stripped, multi-line safe), compared with the 39 `@` keys in
  `custom.bib`.
- arXiv: one script queried `https://export.arxiv.org/api/query?id_list=<ids>` for all 35 bib arXiv ids plus 4 new ids,
  and compared title, full author list (name by name after removing TeX accents), author count and v1 year with each
  entry. Abstract pages (`https://arxiv.org/abs/<id>`) were fetched where the API and the page could differ
  (2606.07631, 2407.21783, 2503.19786).
- DOIs: Crossref (`https://api.crossref.org/works/<doi>`) for the 3 bib DOIs, the van Miltenburg ACL DOI and 3 new
  DOIs; compared title, authors, journal, volume, issue, pages, year.
- Model cards: `meta-llama/Llama-Guard-3-8B`, `meta-llama/Llama-Guard-3-1B`, `google/shieldgemma-9b` (gated; README
  downloaded with the project's HF login), `facebook/nllb-200-distilled-600M`, dataset `LLM-LAT/harmful-dataset`.
- Llama 3 paper PDF (`https://arxiv.org/pdf/2407.21783v3`), text extracted with pypdf, searched for "Llama Guard 3".

## Counts

| Set | Items | ok | mismatch | unverifiable |
|---|---|---|---|---|
| `custom.bib` entries | 39 | 37 | 2 | 0 |
| Citations introduced or needed by round-2/3 proposals | 10 | 8 | 1 | 1 |
| **Total** | **49** | **45** | **3** | **1** |

"ok" = title, author list, year (and journal/volume/pages for DOIs) agree with the source; case-only differences and
TeX encodings of the same letter (`{\c C}elebi` = Çelebi) count as ok. Separately, 7 rendered-title casing defects
come from `acl_natbib.bst` + `make_bib.py`, not from wrong metadata (section 4).

## 1. Keys: cited vs. present

- Cited in `main.tex` at `9d99ae4`: **34 distinct keys** (`\citep`, plus `\citealp{salamanca2026tinyaya}` line 57 and
  `\citealp{cohen1960kappa}` line 216). **Cited but missing from the bib: none.**
- **In the bib but not cited (5):** `rottger2024xstest`, `lee2026relocates`, `yu2026safeseek`, `syed2023attribution`,
  `wilson1927probable`. Harmless (BibTeX prints only cited entries). `wilson1927probable` becomes cited if S5 of
  `CONSOLIDATED.md` (`consolidated.patch` adds `\citep{wilson1927probable}` at line 159) is applied, which is
  recommended: `main.tex` line 159 says "Wilson 95\% intervals" without a reference. (l10's round-1 report counted 32
  cited keys at an older commit; the current file has 34.)
- **Keys used by proposals but not in the bib (3):** `zeng2024shieldgemma` (l03, l18-limitations-ethics),
  `cobbe2021gsm8k` (l18-limitations-ethics, l18-checklist G8), `fedorov2024llamaguard3` (l10 E9 / CONSOLIDATED X3).
  Two further keys are proposed here: `nllbteam2022nllb`, `clopper1934use`, `efron1979bootstrap` (section 5).
- Key/year inconsistency (cosmetic, uncited): `rottger2024xstest` has `year = {2023}` (arXiv v1 2023-08-02, correct);
  only the key says 2024.

## 2. `custom.bib` entries (39)

Source URL is the one queried; the abstract page is `https://arxiv.org/abs/<id>`. n = author count.

| Key | Source checked | Status | Evidence / exact fix |
|---|---|---|---|
| qi2023finetuning | https://arxiv.org/abs/2310.03693 | ok | n=7, Xiangyu Qi, v1 2023-10-05 |
| arditi2024refusal | https://arxiv.org/abs/2406.11717 | ok | n=7, Andy Arditi, v1 2024-06-17 |
| wei2024brittleness | https://arxiv.org/abs/2402.05162 | ok | n=9, Boyi Wei, v1 2024-02-07 |
| mazeika2024harmbench | https://arxiv.org/abs/2402.04249 | ok | n=12, Mantas Mazeika, v1 2024-02-06 |
| rottger2024xstest | https://arxiv.org/abs/2308.01263 | ok (uncited) | n=6, Paul Röttger, v1 2023-08-02; arXiv comment "Accepted at NAACL 2024 (Main Conference)" |
| seth2026circuitkit | https://arxiv.org/abs/2607.19317 | ok | n=4, Pratinav Seth, v1 2026-07-21; space in "CircuitKIT :" is arXiv's own |
| nghiem2026trait | https://arxiv.org/abs/2606.07631 | **mismatch** | API lists "Hal Daumé"; abstract page lists "Hal Daumé III". Fix: `Hal Daum{\'e}` -> `Daum{\'e} III, Hal` (section 3) |
| du2025procon | https://arxiv.org/abs/2509.06795 | ok | n=9, Yanrui Du, v1 2025-09-08 |
| guo2026routing | https://arxiv.org/abs/2609.01455 | ok | n=5, Yitong Guo, v1 2026-09-01; comment "Accepted to the Findings of EMNLP 2026" |
| lee2026relocates | https://arxiv.org/abs/2610.00320 | ok (uncited) | n=6, Jungseob Lee, v1 2026-09-29 |
| hsu2024safelora | https://arxiv.org/abs/2405.16833 | ok | n=6, Chia-Yi Hsu, v1 2024-05-27; comment "camera-ready version accepted for NeurIPS 2024" |
| yu2026safeseek | https://arxiv.org/abs/2603.23268 | ok (uncited) | n=9, Miao Yu, v1 2026-03-24 |
| ruellan2026replay | https://arxiv.org/abs/2610.38001 | ok | n=2, George Ruellan, v1 2026-10-27 |
| hawkins2026multilingual | https://arxiv.org/abs/2606.28843 | ok | n=11, Will Hawkins, v1 2026-06-27; journal_ref "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026" |
| he2024safedata | https://arxiv.org/abs/2404.01099 | ok | n=3, Luxi He, v1 2024-04-01; render fix: capital after "?" (section 4) |
| duan2026monitors | https://arxiv.org/abs/2606.15980 | ok | n=1, Evan Duan, v1 2026-06-14; render fix: capital after "?" |
| hu2021lora | https://arxiv.org/abs/2106.09685 | ok | n=8, Edward J. Hu, v1 2021-06-17 |
| inan2023llamaguard | https://arxiv.org/abs/2312.06674 | ok | n=11, Hakan Inan, v1 2023-12-07; render fix: `{Llama Guard}` |
| grattafiori2024llama3 | https://arxiv.org/abs/2407.21783 | **mismatch** (hidden) | API n=561. The abstract page's author line reads "Guangyi (Jack) Zhang" and "Yu (Sid) Wang", but arXiv parses each into two authors ("Guangyi", "Zhang"; "Yu", "Wang": separate links and `citation_author` tags), and the API and bib follow (bib positions 340-341, 548-549). Parse artifact, so the true count is 559. Not printed (style truncates after 19 names) but the rendered "and 542 others" should be "and 540 others". Fix (optional): in the bib author list `Guangyi and Zhang` -> `Guangyi Zhang` and `Yu and Wang` -> `Yu Wang` (via a name-fix map in `make_bib.py`). Render fix: `{Llama}` |
| syed2023attribution | https://arxiv.org/abs/2310.10348 | ok (uncited) | n=3, Aaquib Syed, v1 2023-10-16 |
| salamanca2026tinyaya | https://arxiv.org/abs/2603.11510 | ok | n=26, Alejandro R. Salamanca, v1 2026-03-12; render fix: `{Tiny Aya}` |
| seth2026safetune | https://arxiv.org/abs/2609.22153 | ok | n=4, Pratinav Seth, v1 2026-08-26 |
| rafailov2023dpo | https://arxiv.org/abs/2305.18290 | ok | n=6, Rafael Rafailov, v1 2023-05-29 |
| bai2022hhrlhf | https://arxiv.org/abs/2204.05862 | ok | n=31, Yuntao Bai, v1 2022-04-12 |
| yang2025qwen3 | https://arxiv.org/abs/2505.09388 | ok | n=60, An Yang, v1 2025-05-14 |
| gemmateam2025gemma3 | https://arxiv.org/abs/2503.19786 | ok (note) | n=216, "Gemma Team" first. The abstract page itself lists "Dmitry" and "Lepikhin" as two authors and "Jean-bastien Grill"; the bib follows the source. "Jean-bastien Grill" is author 14 and **is printed** (the style prints 19 of >20 names), contrary to l18-format O2 ("hidden"). No change recommended without checking the report PDF |
| zou2023advbench | https://arxiv.org/abs/2307.15043 | ok | n=6, Andy Zou, v1 2023-07-27 |
| tsatsos2026probes | https://arxiv.org/abs/2610.35586 | ok | n=1, Marios Tsatsos, v1 2026-10-26 |
| gao2026alarm | https://arxiv.org/abs/2610.04575 | ok | n=1, Xueping Gao, v1 2026-10-03; comment "Accepted at ... DAI 2026" |
| hurtado2026abliterated | https://arxiv.org/abs/2607.01854 | ok | n=1, Gabriel Hurtado, v1 2026-07-02; render fix: capital after "?" |
| bao2026trace | https://arxiv.org/abs/2610.07518 | ok | n=4, Ziqun Bao, v1 2026-10-05 |
| vishnubhotla2026grounded | https://arxiv.org/abs/2606.03648 | ok | n=4, Krishnapriya Vishnubhotla, v1 2026-06-02 |
| zou2023repe | https://arxiv.org/abs/2310.01405 | ok | n=21, Andy Zou, v1 2023-10-02 |
| peng2024landscape | https://arxiv.org/abs/2405.17374 | ok | n=4, ShengYun Peng, v1 2024-05-27; comment "NeurIPS'24" |
| taori2023alpaca | https://github.com/tatsu-lab/stanford_alpaca (README citation block) | ok | same title, 8 authors, 2023; render fix: `Stanford {Alpaca}` |
| wilson1927probable | https://doi.org/10.1080/01621459.1927.10502953 (Crossref) | ok | Edwin B. Wilson; JASA 22(158) 209-212, 1927 |
| cohen1960kappa | https://doi.org/10.1177/001316446002000104 (Crossref) | ok | Jacob Cohen; Educ. Psychol. Meas. 20(1) 37-46, 1960 |
| vanmiltenburg2021preregistering | https://arxiv.org/abs/2103.06944 and https://doi.org/10.18653/v1/2021.naacl-main.51 (Crossref) | ok | n=3, Emiel van Miltenburg; Crossref: NAACL-HLT 2021 proceedings, pages 613-623. Optional completion (l10 E8, confirmed): add `pages = {613--623}, doi = {10.18653/v1/2021.naacl-main.51},` |
| nosek2018preregistration | https://doi.org/10.1073/pnas.1708274114 (Crossref) | ok | n=4, Nosek; PNAS 115(11) 2600-2606, 2018 |

Venue notes proposed by l10 E8 (COLM 2024, Findings of EMNLP 2026, NeurIPS 2024 x2, ICML 2026): the four arXiv
comments/journal_ref values above were re-fetched today and match l10's quotes; the COLM 2024 note for
`he2024safedata` comes from the PDF header (l10), not re-checked here.

## 3. "Hal Daumé" vs "Daumé III" (open question from CONSOLIDATED.md, lane 9)

| Form | Source that supports it |
|---|---|
| `Hal Daumé` | arXiv API `export.arxiv.org/api/query?id_list=2606.07631` (`<name>Hal Daumé</name>`); this is what `make_bib.py` reads, hence the current bib |
| `Hal Daumé III` | arXiv abstract page `https://arxiv.org/abs/2606.07631`: author line "Hal Daumé III" and meta tag `citation_author` = `Daumé III, Hal` |
| `Daum{\'e} III, Hal` | ACL Anthology's own BibTeX for his papers, e.g. `https://aclanthology.org/P07-1033.bib`: `author = "Daum{\'e} III, Hal"` |

Resolution: the name is **Hal Daumé III**; the API drops the suffix. Use `Daum{\'e} III, Hal`. Under `acl_natbib.bst`
the name format is `"{ff~}{vv~}{ll}{, jj}"` (line 569), so this form (suffix inside the last name) renders
"Hal Daumé III", whereas the BibTeX Jr form `Daum{\'e}, III, Hal` would render "Hal Daumé, III" with a comma. Do not
write `Hal Daum{\'e} III` (BibTeX would read "III" as the last name).

Exact fix in `custom.bib` (entry `nghiem2026trait`), old -> new:
`  author = {Huy Nghiem and Sy-Tuyen Ho and Sarah Wiegreffe and Hal Daum{\'e}},`
-> `  author = {Huy Nghiem and Sy-Tuyen Ho and Sarah Wiegreffe and Daum{\'e} III, Hal},`
Because `custom.bib` is regenerated, make the change in `make_bib.py` (section 4, B3), else it is overwritten.

## 4. `paper/make_bib.py` casing and name bugs (report only; no code changed)

`acl_natbib.bst` applies `"t" change.case$` to titles of `@misc`, `@article` and `@inproceedings`
(`FUNCTION {format.title}`, lines 672-678): every unbraced letter is lower-cased except the first character and the
first after a colon + space.

- **B1 (confirmed, matches l10 E7 / l18 O2).** The brace rule (line 102,
  `\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*)\b`) only protects words with two capitals. Single-capital proper names are
  lowered: "The llama 3 herd of models" (`grattafiori2024llama3`), "Llama guard: ..." (`inan2023llamaguard`),
  "Tiny aya: ..." (`salamanca2026tinyaya`), and in `MANUAL` "Stanford alpaca: ..." (`taori2023alpaca`).
  `Gemma 3` and `Qwen3` are title-initial, so unaffected.
- **B2 (confirmed).** A capital after "?" is lowered (BibTeX only keeps one after ":"): `he2024safedata`
  ("... safe data? identifying"), `duan2026monitors` ("... updates? benchmarking"), `hurtado2026abliterated`
  ("... abliterated? a two-signal"). l10 E7's two added regexes fix B1+B2 for exactly these six arXiv titles; l18 O2's
  variant also braces `Gemma`/`Qwen3`, harmless. Either works; l10's is the tested one.
- **B3 (name suffix).** `make_bib.py` has no way to override a name the API gets wrong; the API drops "III" (section 3)
  and splits names that contain a parenthesis (`grattafiori2024llama3`). Suggested: a `NAME_FIX` dict applied to each
  author string after `tex()`, e.g. `{"Hal Daum{\\'e}": "Daum{\\'e} III, Hal"}`, and, for 2407.21783, merging
  "Guangyi"+"Zhang" and "Yu"+"Wang". Untested.
- **B4 (latent, no current entry affected).** In `tex()`, the `special` map's `"å"` (and the intent for any accented
  letter) never fires for letters that NFD decomposes: `unicodedata.normalize("NFD", s)` splits `å` into `a` + ring
  before the lookup, and combining marks outside `acc` (ring, breve, macron, ogonek, dot above, double acute) are then
  silently dropped by the `ord(ch) < 128` filter. A scan of all 39 fetched arXiv records found no such mark today.
  Suggested: add `"̊": "r "`, `"̆": "u "`, `"̄": "="`, `"̨": "k "`, `"̇": "."`,
  `"̋": "H "` to `acc`, or warn on any dropped non-ASCII character.
- **B5 (cosmetic).** Group-author detection only braces names ending in "Team"; an author like
  "Llama Team, AI @ Meta" (Meta's own citation blocks) would be parsed as last name "Llama Team". Not in the bib today;
  relevant only if a Meta model-card entry is added (section 5, entry F).

## 5. Citations introduced or needed by round-2/3 proposals (10)

| Item (proposal) | Proposed key | Source checked | Status | Exact fix / recommendation |
|---|---|---|---|---|
| ShieldGemma 9B (l03-third-judge; l18-limitations-ethics) | `zeng2024shieldgemma` | https://arxiv.org/abs/2407.21772 (API); `google/shieldgemma-9b` card citation block | ok | l03's entry has the correct title and all 12 authors in order. Style fix only: l03 uses `@article` + `journal = {arXiv preprint ...}`; the bib uses `@misc` + `eprint`. Use entry A, or add `"zeng2024shieldgemma": "2407.21772"` to `ARXIV` in `make_bib.py` (the generator will then also brace `{ShieldGemma}`, `{AI}`; `Gemma` needs B1's brace rule) |
| Llama Guard 3 8B, the second judge (`main.tex` 215, 329; l03, l13) | existing `grattafiori2024llama3` | `meta-llama/Llama-Guard-3-8B` card, "# Citation" block = arXiv 2407.21783; PDF v3 §5.4.7 "System Level Safety": "we develop and release a new classifier, Llama Guard 3, which is a Llama 3 8B model fine-tuned for safety classification" | ok | No new entry. Optional: line 215 "with Llama Guard 3 8B" -> "with Llama Guard 3 8B \citep{grattafiori2024llama3}" |
| Llama Guard 3 1B, the primary judge (`main.tex` 87-88; l10 E9 / X3) | `fedorov2024llamaguard3` | https://arxiv.org/abs/2411.17713 (API, abstract) | ok | Title, 20 authors, v1 2024-11-18 verified. Its abstract is about Llama Guard 3-1B-INT4 and names Llama Guard 3-1B only as "its larger counterpart", so it is a secondary source for the 1B model. The 1B card's own citation is a URL entry (entry F). Recommend: keep `\citep{inan2023llamaguard}` and add F (or B) only if a 1B-specific source is wanted |
| GSM8K (l18-limitations-ethics; l18-checklist G8) | `cobbe2021gsm8k` | https://arxiv.org/abs/2110.14168 (API) | **mismatch** (draft entry) | l18's draft lists 5 authors "and others"; arXiv lists 12. Use entry C (full list from the API) |
| NLLB-200 distilled 600M, translation model of Job 20 (l06-h4-multilingual: "(NLLB-200, translation quality not measured)") | `nllbteam2022nllb` | https://arxiv.org/abs/2207.04672 (API); `facebook/nllb-200-distilled-600M` card: "Paper ... NLLB Team et al, No Language Left Behind: Scaling Human-Centered Machine Translation, Arxiv, 2022"; peer-reviewed version https://doi.org/10.1038/s41586-024-07335-x (Crossref: "Scaling neural machine translation to 200 languages", Nature 630(8018) 841-846, 2024) | ok | Use entry D (the paper the model card names). In l06's sentence: "(NLLB-200, translation quality not measured)" -> "(NLLB-200 \citep{nllbteam2022nllb}, translation quality not measured)" |
| Wilson interval (`main.tex` 159; l10 E5, l13 Edit 2/3, l18) | existing `wilson1927probable` | https://doi.org/10.1080/01621459.1927.10502953 (Crossref) | ok | Already in bib; apply S5 (`\citep{wilson1927probable}`). In l13's appendix table caption add the same citation after "Wilson" if S5 is not applied |
| Clopper-Pearson interval (l13 Edit 3, table column) | `clopper1934use` | https://doi.org/10.1093/biomet/26.4.404 (Crossref) | ok | Entry E1. In l13's caption: "95\% intervals (exploratory; ..." -> "95\% Wilson \citep{wilson1927probable} and Clopper--Pearson \citep{clopper1934use} intervals and bootstrap \citep{efron1979bootstrap} intervals (exploratory; ..." (only if l13 Edit 3 is accepted) |
| Bootstrap intervals (l13 Edit 1 and 3; `review_followups.md`) | `efron1979bootstrap` | https://doi.org/10.1214/aos/1176344552 (Crossref) | ok | Entry E2. General reference for the bootstrap only; no source for the family-clustered variant was verified, so none is proposed |
| Cohen's kappa (l03 text "$\kappa$ 0.26 ...") | existing `cohen1960kappa` | https://doi.org/10.1177/001316446002000104 | ok | Already cited at line 216; nothing to add |
| LLM-LAT refusal data used in the control (l18-checklist G8; `main.tex` 83, 169) | none | https://huggingface.co/datasets/LLM-LAT/harmful-dataset (card 363 bytes; API sha `8bfba31bc6d93a5b71808fee5275ef4b6330ed91`, last modified 2024-07-24) | **unverifiable** | The card names no paper and no licence, so no primary source can be cited. Do not attribute it to a paper. Recommend a URL citation with access date and revision (entry G) or a footnote URL |

Also mentioned in proposals without a citation and not needing one here: Dolly (in `main.tex` already, uncited, as
one of the SafeTune fine-tune domains; outside this lane's scope, noted only).

## 6. Ready-to-paste BibTeX (built only from the fetched metadata)

Author lists, titles and years are copied from the arXiv API or Crossref responses fetched 2026-11-23; accents are
written the way `make_bib.py`'s `tex()` writes them; braces only protect capitals. For the arXiv entries the simplest
route is adding the id to `ARXIV` in `make_bib.py` and regenerating, which produces the same fields.

A. ShieldGemma (arXiv 2407.21772; 12 authors; v1 2024-07-31)
```
@misc{zeng2024shieldgemma,
  title = {{ShieldGemma}: Generative {AI} Content Moderation Based on {Gemma}},
  author = {Wenjun Zeng and Yuchi Liu and Ryan Mullins and Ludovic Peran and Joe Fernandez and Hamza Harkous and Karthik Narasimhan and Drew Proud and Piyush Kumar and Bhaktipriya Radharapu and Olivia Sturman and Oscar Wahltinez},
  year = {2024},
  eprint = {2407.21772},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2407.21772}
}
```

B. Llama Guard 3-1B-INT4 (arXiv 2411.17713; 20 authors; v1 2024-11-18) — optional
```
@misc{fedorov2024llamaguard3,
  title = {{Llama Guard} 3-1B-{INT4}: Compact and Efficient Safeguard for Human-{AI} Conversations},
  author = {Igor Fedorov and Kate Plawiak and Lemeng Wu and Tarek Elgamal and Naveen Suda and Eric Smith and Hongyuan Zhan and Jianfeng Chi and Yuriy Hulovatyy and Kimish Patel and Zechun Liu and Changsheng Zhao and Yangyang Shi and Tijmen Blankevoort and Mahesh Pasupuleti and Bilge Soran and Zacharie Delpierre Coudert and Rachad Alao and Raghuraman Krishnamoorthi and Vikas Chandra},
  year = {2024},
  eprint = {2411.17713},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2411.17713}
}
```

C. GSM8K (arXiv 2110.14168; 12 authors; v1 2021-10-27) — replaces l18's "and others" draft
```
@misc{cobbe2021gsm8k,
  title = {Training Verifiers to Solve Math Word Problems},
  author = {Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
  year = {2021},
  eprint = {2110.14168},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2110.14168}
}
```

D. NLLB (arXiv 2207.04672; 39 authors incl. the group "NLLB Team" first; v1 2022-07-11)
```
@misc{nllbteam2022nllb,
  title = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author = {{NLLB Team} and Marta R. Costa-juss{\`a} and James Cross and Onur {\c C}elebi and Maha Elbayad and Kenneth Heafield and Kevin Heffernan and Elahe Kalbassi and Janice Lam and Daniel Licht and Jean Maillard and Anna Sun and Skyler Wang and Guillaume Wenzek and Al Youngblood and Bapi Akula and Loic Barrault and Gabriel Mejia Gonzalez and Prangthip Hansanti and John Hoffman and Semarley Jarrett and Kaushik Ram Sadagopan and Dirk Rowe and Shannon Spruit and Chau Tran and Pierre Andrews and Necip Fazil Ayan and Shruti Bhosale and Sergey Edunov and Angela Fan and Cynthia Gao and Vedanuj Goswami and Francisco Guzm{\'a}n and Philipp Koehn and Alexandre Mourachko and Christophe Ropers and Safiyyah Saleem and Holger Schwenk and Jeff Wang},
  year = {2022},
  eprint = {2207.04672},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2207.04672}
}
```

E1. Clopper and Pearson (Crossref, DOI 10.1093/biomet/26.4.404). Crossref stores the title in capitals; only the case
is changed here.
```
@article{clopper1934use,
  title = {The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial},
  author = {Clopper, C. J. and Pearson, E. S.},
  journal = {Biometrika},
  volume = {26}, number = {4}, pages = {404--413}, year = {1934},
  doi = {10.1093/biomet/26.4.404}
}
```

E2. Efron (Crossref, DOI 10.1214/aos/1176344552)
```
@article{efron1979bootstrap,
  title = {Bootstrap Methods: Another Look at the Jackknife},
  author = {Efron, B.},
  journal = {The Annals of Statistics},
  volume = {7}, number = {1}, pages = {1--26}, year = {1979},
  doi = {10.1214/aos/1176344552}
}
```

F. Llama Guard 3 1B model card — only if a 1B-specific source is wanted. Built from the card's own citation block
(`meta-llama/Llama-Guard-3-1B` README, "## Citation": author "Llama Team, AI @ Meta", title "The Llama 3 Family of
Models", howpublished the PurpleLlama MODEL_CARD URL, year 2024). The author is double-braced so BibTeX does not split
it at the comma (B5); `note` adds the access date.
```
@misc{metallamaguard3,
  author = {{Llama Team, AI @ Meta}},
  title = {The {Llama} 3 Family of Models},
  howpublished = {\url{https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md}},
  year = {2024},
  note = {Llama Guard 3-1B model card, accessed 2026-11-23}
}
```

G. LLM-LAT refusal data — no primary source; URL entry only
```
@misc{llmlat2024harmful,
  title = {{LLM-LAT/harmful-dataset}},
  howpublished = {\url{https://huggingface.co/datasets/LLM-LAT/harmful-dataset}},
  year = {2024},
  note = {Hugging Face dataset, revision 8bfba31, accessed 2026-11-23}
}
```
(`year` = the Hub's last-modified date, 2024-07-24; the card gives no author, so none is written.)

## Not done / uncertainty

- The Gemma 3 and Llama 3 report PDFs were not checked for the split names beyond the abstract pages; the Llama 3 fix
  rests on the abstract page's author line, the Gemma names are left as the source gives them.
- `he2024safedata`'s COLM 2024 note (l10 E8) was not re-checked here (it comes from the PDF header, not arXiv metadata).
- None of the BibTeX above was compiled through `acl_natbib.bst` in this lane.
