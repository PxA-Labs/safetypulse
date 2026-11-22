# Release-policy audit, round 2 (lane 19, 2026-11-22)

Scope: harden `tools/check_release_policy.py` for the new HF job outputs (Job 15 onward) and executed notebooks, run
it on the integration branch and on every pushed round-2 branch that adds job files, and audit the visibility of
every Hub store the workflow uses. No job was launched (GPU cap $0). No Hub visibility, file or collection was
changed: the Hub audit used read-only `HfApi` calls.

## 1. Checker changes

New rules (existing name, size and answer-field rules are unchanged):

| Rule | Applies to | Fails on | Exempt |
|---|---|---|---|
| Long free text | `kaggle/executed/12_hf_jobs/files/jobN*` with N >= 15 (any later job number, files or subfolders), `.json` / `.jsonl` / `.csv` / `.tsv` | any string value, list item, CSV cell, dict key or CSV header longer than 80 characters | one whitespace-free token that is a hex hash, a path, a URL or a model id (`org/name`) |
| Text-named fields | same files | a field named `answer`, `response`, `completion`, `generation`, `output_text`, `prompt_text`, `translation` (any `xx_` prefix such as `model_`, `hi_`; plurals) holding text | empty, `null`, numbers, the placeholder `[withheld: judged harmful]` |
| Unparseable output | same files | a JSON file or JSONL line that does not parse (its content cannot be checked) | none |
| Notebook prose | every `.ipynb` | a `stream`, `display_data` or `execute_result` (`text/plain`, `text/markdown`) output line longer than 80 characters with at least 12 words and at least 95% letters (any script, including Devanagari), marks, spaces or sentence punctuation | lines starting like a log message (`LOG_LINE`: Warning/Note/Error/INFO, `[tag]`, `!!`, absl `I0000`, `/path/file.py:12:` Python warnings, the pipeline's own "No HF_TOKEN" and "N earlier rows were scored" notices); `error` outputs (tracebacks) |

Reports name the file, field (or cell number and output kind) and line, and never print the value or output text.
The checker stays standard-library only, so the CI step (`python tools/check_release_policy.py`) needs no install.

Known limits: the notebook rule is a heuristic. Short answers (under 80 characters or 12 words), prose wrapped in a
dict repr (`{'answer': '...'}` lowers the letter ratio) and prose that starts with a log prefix such as "Note:" are
not caught there; job outputs are covered by the stricter JSON rules instead. Older job files (Jobs 1-14) keep the
old rules: `job4_build.json` holds benign harmless-sample answers and prompts as keys, and Jobs 5-7 store CLI
commands, all published before this rule.

### Tests that fail on the old checker

The new tests in `tests/test_release_policy.py` use synthetic placeholder strings only. Run against the old checker
(copy of `origin/stage2/wf-integration` `eb262cd`): 9 failed, 25 passed, 1 skipped (the repository test, which needs
a git checkout). Failure lines, quoted:

```
FAILED tests/test_release_policy.py::test_strict_job_file_scope - AttributeError: module 'check_release_policy' has no attribute 'is_strict_job_file'
FAILED tests/test_release_policy.py::test_job_output_text_fields_are_blocked[translation]    assert [] == [... "job output: text in field 'translation' (line 2)"]
FAILED tests/test_release_policy.py::test_job_output_text_fields_are_blocked[output_text]    assert [] == [... "job output: text in field 'output_text' (line 2)"]
FAILED tests/test_release_policy.py::test_job_output_text_fields_are_blocked[prompt_text]    assert [] == [... "job output: text in field 'prompt_text' (line 2)"]
FAILED tests/test_release_policy.py::test_job_output_text_fields_are_blocked[hi_translation] assert [] == [...]
FAILED tests/test_release_policy.py::test_job_output_text_fields_are_blocked[translations]   assert [] == [...]
FAILED tests/test_release_policy.py::test_job_output_long_free_text_is_blocked   assert [] == [... "job output: long text in field 'notes[]' (line 1)" ...] (4 items)
FAILED tests/test_release_policy.py::test_unparseable_job_output_is_blocked     AssertionError: assert [] == [... 'job output: unparseable JSON (line 1)']
FAILED tests/test_release_policy.py::test_notebook_free_text_outputs_are_blocked
```

With the new checker: 35 passed (`tests/test_release_policy.py`).

## 2. Checker results per branch

Command, run from a detached scratch worktree of each branch: `python3 tools/check_release_policy.py
origin/stage2/wf-integration` (new checker). Every scan below reports 0 violations, with the new and with the old
checker.

| Branch | Head | Job / notebook files added vs integration | Changed | Tracked | Added in history | Result |
|---|---|---|---|---|---|---|
| `stage2/wf-integration` | `eb262cd` | (base; includes `job15_*` and 101 notebooks) | 0 | 580 | 0 | pass |
| `wf/r2-lane2` | `cf7140a` | `job16_h6_testsplit.jsonl` | 7 | 585 | 5 | pass |
| `wf/r2-lane3` | `5e91599` | `job17_third_judge_h6.jsonl` | 7 | 585 | 5 | pass |
| `wf/r2-lane4` | `52ae95a` | `job18_07_finetune_checkpoints.ipynb`, `job18_08_monitor_checkpoints.ipynb`, `job18_dw.jsonl`, `job18_job.json`, `job18_monitor.jsonl`, `job18_summary.json`, `job18_train_log.json` | 15 | 592 | 12 | pass |
| `wf/r2-lane5` | `9f07db7` | `job19_h5_localise.jsonl` | 7 | 585 | 5 | pass |
| `wf/r2-lane6` | `a0f5de2` | `job20_asr.jsonl`, `job20_directions.json`, `job20_translations.json` | 9 | 587 | 7 | pass |

All five branches were pushed at the time of the run (fetched 2026-11-22); none was skipped. As a cross-check that
the pass is not a checker gap, a separate key survey of each new job file found no string longer than 80 characters
in any of them; `job20_translations.json` holds only translation statistics (counts, character ratios, SHA-256 of
each language's prompt file, MT model and revision), not translated text. Other pushed round-2 branches
(`wf/r2-lane1`, `-8` to `-13`) were not in this lane's list and were not checked here.

## 3. Hub audit (read only)

Queried 2026-11-22 with `HfApi` (`dataset_info` / `model_info`, `list_models` / `list_datasets` / `list_spaces`
with `author=`, `bucket_info`, `list_buckets`, `get_collection`, `list_repo_files`). Names were matched with
`/harm|ft[_-]harm|harmdense/i`.

### Workflow stores

| Store | Type | `private` | Note |
|---|---|---|---|
| `purvansh01/safetypulse-jobs` | dataset | True | job outputs; 165 files; one harm-named file, `data/eval_harm.jsonl` (evaluation prompts, private repo) |
| `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` | model | True | the only `safetypulse-wf-*` repo in the org (models, datasets and spaces listed); 168 files, none harm-named |
| `pxaLabs-lexsi/jobs-artifacts` | bucket | True | `bucket_info`; it is the only bucket `list_buckets("pxaLabs-lexsi")` returns |
| `pxaLabs-lexsi/safetypulse-workflow-models-private-6b030dd4c3a7647760ae6f16` | collection | True | 1 item: `pxaLabs-lexsi/safetypulse-wf-h3fine-lora-alpaca` (private) |
| `pxaLabs-lexsi/safetypulse-workflow-datasets-private-6b030dd671c703296d096ac3` | collection | True | empty |

### Harm-named files in public repos

All repos of `purvansh01` (18 models, 13 datasets, 10 spaces) and `pxaLabs-lexsi` (1 model, private) were listed.
No public repo has a harm adapter, `ft_harm` or `harmdense` file, and no repo name contains "harm". Two public files
match the pattern; neither is a withheld artifact:

| Public repo | File | Why it is allowed |
|---|---|---|
| `purvansh01/safetypulse-eval` (dataset) | `eval_harm.jsonl` | the harmful-request evaluation prompts, published on purpose ("Every prompt ... is published", docs/11 section 4); prompts, not answers |
| `purvansh01/safetypulse-pairs` (dataset) | `unpaired_v1/harmless_pool.jsonl` | the harmless prompt pool; matched only because "harmless" contains "harm" |

The withheld artifacts are in `purvansh01/safetypulse-withheld` (private: True, 81 of 85 files harm-named), as
docs/11 describes. `purvansh01/safetypulse-earth-probe-bundle` is also private. Every other SafetyPulse repo of
`purvansh01` is public and harm-free by file name; the public non-SafetyPulse repos (conflict-env models and
several spaces) have no harm-named files.

Nothing needs changing on the Hub. The two private collections remain the deviation the round-1 ledger already
records; the user decides on them.
