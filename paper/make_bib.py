"""Build paper/custom.bib from the arXiv API (authors, titles and years as arXiv lists them), plus a few classic
non-arXiv references written out by hand below. Run: python paper/make_bib.py   (needs network access)."""
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

ARXIV = {   # key: arXiv id   (venue notes per docs/12_RELATED_WORK.md)
    "qi2023finetuning": "2310.03693", "arditi2024refusal": "2406.11717", "wei2024brittleness": "2402.05162",
    "mazeika2024harmbench": "2402.04249", "rottger2024xstest": "2308.01263", "seth2026circuitkit": "2607.19317",
    "nghiem2026trait": "2606.07631", "du2025procon": "2509.06795", "guo2026routing": "2609.01455",
    "lee2026relocates": "2610.00320", "hsu2024safelora": "2405.16833", "yu2026safeseek": "2603.23268",
    "ruellan2026replay": "2610.38001", "hawkins2026multilingual": "2606.28843", "he2024safedata": "2404.01099",
    "duan2026monitors": "2606.15980", "hu2021lora": "2106.09685", "inan2023llamaguard": "2312.06674",
    "grattafiori2024llama3": "2407.21783", "syed2023attribution": "2310.10348", "salamanca2026tinyaya": "2603.11510",
    "seth2026safetune": "2609.22153", "rafailov2023dpo": "2305.18290", "bai2022hhrlhf": "2204.05862",
    "yang2025qwen3": "2505.09388", "gemmateam2025gemma3": "2503.19786", "zou2023advbench": "2307.15043",
    # added after the related-work check (abstract pages verified 2026-11-21)
    "tsatsos2026probes": "2610.35586", "gao2026alarm": "2610.04575", "hurtado2026abliterated": "2607.01854",
    "bao2026trace": "2610.07518", "vishnubhotla2026grounded": "2606.03648",
    # suggested in an internal review (abstract pages verified 2026-11-22)
    "zou2023repe": "2310.01405", "peng2024landscape": "2405.17374",
    # added for the 2026-11-22 follow-ups and the GSM8K citation (abstract pages verified 2026-11-23)
    "cobbe2021gsm8k": "2110.14168", "zeng2024shieldgemma": "2407.21772", "nllbteam2022nllb": "2207.04672",
    # Lexsi call 2026-11-24: cite every toolkit and model used (abstract pages verified 2026-11-24)
    "bhattacharjee2026curatorkit": "2606.21631", "dang2024ayaexpanse": "2412.04261",
    # 2026-11-24 experiments E1-E5: benchmarks and quantization (arXiv API titles/authors checked 2026-11-25)
    "zhou2023ifeval": "2311.07911", "artetxe2019xquad": "1910.11856", "frantar2022gptq": "2210.17323",
    "biderman2024lessons": "2405.14782", "hendrycks2020mmlu": "2009.03300",
}
# names the arXiv API gets wrong (abstract page of 2606.07631: "Hal Daumé III"; ACL Anthology form)
NAME_FIX = {"Hal Daum{\\'e}": "Daum{\\'e} III, Hal"}
MANUAL = r"""
@software{lexsi2026auditkit,
  title = {{AuditKIT}},
  author = {{Lexsi Labs}},
  year = {2026},
  version = {1.0.0},
  howpublished = {GitHub repository},
  url = {https://github.com/Lexsi-Labs/AuditKIT}
}
@software{lexsi2026aligntune,
  title = {{AlignTune}: A Modular Toolkit for Post Training Alignment of {LLMs}},
  author = {{Lexsi Labs}},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/Lexsi-Labs/aligntune}
}
@software{lmeval2026software,
  title = {{lm-evaluation-harness}},
  author = {{EleutherAI}},
  year = {2026},
  version = {0.4.13},
  howpublished = {GitHub repository},
  url = {https://github.com/EleutherAI/lm-evaluation-harness}
}
@software{vllm2026llmcompressor,
  title = {{LLM Compressor}},
  author = {{The vLLM Project}},
  year = {2026},
  version = {0.15.0},
  howpublished = {GitHub repository},
  url = {https://github.com/vllm-project/llm-compressor}
}
@software{vllm2026compressedtensors,
  title = {compressed-tensors},
  author = {{The vLLM Project}},
  year = {2026},
  version = {0.20.1},
  howpublished = {GitHub repository},
  url = {https://github.com/vllm-project/compressed-tensors}
}
@misc{taori2023alpaca,
  title = {Stanford {Alpaca}: An Instruction-following {LLaMA} model},
  author = {Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and Dubois, Yann and Li, Xuechen and Guestrin, Carlos and Liang, Percy and Hashimoto, Tatsunori B.},
  year = {2023},
  howpublished = {GitHub repository},
  url = {https://github.com/tatsu-lab/stanford_alpaca}
}
@article{wilson1927probable,
  title = {Probable Inference, the Law of Succession, and Statistical Inference},
  author = {Wilson, Edwin B.},
  journal = {Journal of the American Statistical Association},
  volume = {22}, number = {158}, pages = {209--212}, year = {1927},
  doi = {10.1080/01621459.1927.10502953}
}
@article{cohen1960kappa,
  title = {A Coefficient of Agreement for Nominal Scales},
  author = {Cohen, Jacob},
  journal = {Educational and Psychological Measurement},
  volume = {20}, number = {1}, pages = {37--46}, year = {1960},
  doi = {10.1177/001316446002000104}
}
@inproceedings{vanmiltenburg2021preregistering,
  title = {Preregistering {NLP} Research},
  author = {van Miltenburg, Emiel and van der Lee, Chris and Krahmer, Emiel},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2021},
  eprint = {2103.06944}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2103.06944}
}
@article{nosek2018preregistration,
  title = {The Preregistration Revolution},
  author = {Nosek, Brian A. and Ebersole, Charles R. and DeHaven, Alexander C. and Mellor, David T.},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {115}, number = {11}, pages = {2600--2606}, year = {2018},
  doi = {10.1073/pnas.1708274114}
}
"""


def tex(s):
    """BibTeX-safe ASCII: accents as TeX commands (Unicode is not allowed in BibTeX entries)."""
    acc = {"́": "'", "̀": "`", "̈": '"', "̂": "^", "̃": "~", "̧": "c ", "̌": "v "}
    special = {"ø": "\\o", "Ø": "\\O", "ł": "\\l", "Ł": "\\L", "ß": "\\ss", "æ": "\\ae", "å": "\\aa", "ı": "\\i"}
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if ch in special:   # letters with no decomposition (ø, ł, ...) would otherwise be dropped
            out.append("{" + special[ch] + "}")
        elif ch in acc and out:
            out[-1] = "{\\" + acc[ch] + out[-1] + "}"
        elif ord(ch) < 128:
            out.append(ch)
    return "".join(out).replace("&", r"\&")


def fetch(ids):
    q = ",".join(ids)
    x = urllib.request.urlopen(f"https://export.arxiv.org/api/query?id_list={q}&max_results={len(ids)}").read().decode()
    meta = {}
    for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
        aid = re.search(r"<id>https?://arxiv.org/abs/([0-9.]+)", e).group(1)
        meta[aid] = {"title": re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip(),
                     "authors": re.findall(r"<name>(.*?)</name>", e),
                     "year": re.search(r"<published>(\d{4})", e).group(1)}
    return meta


meta = fetch(list(ARXIV.values()))
missing = [k for k, v in ARXIV.items() if v not in meta]
assert not missing, missing
out = []
for key, aid in ARXIV.items():
    m = meta[aid]
    # group authors ("Gemma Team") are one name: brace them so BibTeX does not split first/last
    authors = " and ".join("{" + tex(a).strip() + "}" if tex(a).strip().endswith("Team") else tex(a).strip()
                           for a in m["authors"])
    for wrong, right in NAME_FIX.items():
        authors = authors.replace(wrong, right)
    title = tex(m["title"]).replace("{", "").replace("}", "")
    title = re.sub(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*)\b", r"{\1}", title)   # keep acronyms / CamelCase
    title = re.sub(r"\b(Llama Guard|Llama|Tiny Aya|Gemma)\b", r"{\1}", title)  # proper names the style lower-cases
    title = re.sub(r"([?!]) ([A-Z])", r"\1 {\2}", title)                       # capital after a question mark
    out.append(f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{m['year']}}},\n"
               f"  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}")
Path(__file__).with_name("custom.bib").write_text("\n".join(out) + "\n" + MANUAL)
for key, aid in ARXIV.items():
    print(f"{key:28s} {aid}  {meta[aid]['year']}  {meta[aid]['authors'][0]}: {meta[aid]['title'][:70]}")
