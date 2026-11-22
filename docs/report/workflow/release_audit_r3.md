# Release-policy audit, round 3 (lane 19, 2026-11-23)

Scope: owner decision D3 (`docs/report/workflow/owner_decisions.md`) records that `kaggle/executed/` is public with
the GitHub repository and that `tools/check_release_policy.py` did not look for compliance phrases or tokens in
notebook outputs, nor at `.log` files at all. This round extends the checker to those files, then runs it on the
integration head and on every branch that adds HF job outputs. No job was launched (GPU cap $0), no Hub state was
changed and no committed data file was edited or deleted.

## 1. Checker changes (`tools/check_release_policy.py`)

| Rule | Applies to | Fails on | Reported as |
|---|---|---|---|
| Access token | every UTF-8 text file; notebook cell sources and text outputs | `hf_` + 30 or more alphanumerics, `KGAT_` + 16 or more, `ghp_` + 30 or more, `github_pat_` + 30 or more (not preceded by a letter or digit) | kind (`hf_` ...), file, cell / output kind, line, column, line hash; never the value |
| Compliance opener | notebook outputs (`stream`, `execute_result` / `display_data` `text/plain` and `text/markdown`; images and `error` outputs skipped); `*.log` / `*.txt` under `kaggle/executed/` | a line containing `Sure, here`, `Step 1:`, `here's how to`, `Here is how` (the D3 phrase list), `Sure! Here`, `Certainly, here`, `Certainly! Here`, `Of course, here`, `Absolutely, here`, `Absolutely! Here`; case-insensitive, curly apostrophe treated as straight | file, cell / output kind, line, line hash |
| Free text | same notebook outputs (unchanged round-2 rule) and now also those run logs | a line over 80 characters, at least 12 words, at least 95% letters / marks / spaces / sentence punctuation, not starting like a log message (`LOG_LINE`) | file, cell / output kind, line, line hash |

The existing rules (weights, harm checkpoints, withheld file names, answer-named fields, strict job-output text,
size) are unchanged.

**Allowlist.** `ALLOWED_SNIPPETS` holds reviewed false positives of the three line rules as
`(repository path, sha256 of the stripped line)`. Every finding prints that hash (`sha256 <64 hex>`), so a reviewer
copies it without the text ever entering the allowlist or the CI log. A key matches one line in one file only; the
same line in another file is still reported. The shipped allowlist is empty, and a test asserts it holds 64-hex
hashes only. The path-level `ALLOWED` set and the `release-policy-reviewed` PR label work as before.

**Format change.** Free-text findings in notebooks now end in `; sha256 <hex>)`, for example
`free text in notebook output (cell 1, stream stdout, line 2; sha256 ...)`. Nothing parses these strings except the
tests, which were updated.

### Tests (written first)

Commit `a79a89f` added 22 tests to `tests/test_release_policy.py` (synthetic strings only; tokens are put together
at run time so that the file holds no token-shaped literal) and updated the expected notebook reason format. At that
commit, with the old checker: **22 failed, 36 passed**. Of the failures, 19 were `AttributeError: ... no attribute
'snippet_sha256'`, one was missing `ALLOWED_SNIPPETS`, one was missing `COMPLIANCE_OPENERS`, and
`test_main_output_never_contains_the_token` failed because `main` returned 0 (the leak went undetected).

Covered: a token of each kind in each of the three text output kinds, reported by cell, line and column; a token in
an image output is ignored; each D3 opener; the D3 list is a subset of `COMPLIANCE_OPENERS`; `.log` and `.txt` run
logs with a token, an opener and a prose line; clean logs, a log outside `kaggle/executed/` (only tokens apply) and a
non-UTF-8 log pass; a token in a source file; **redaction**: `main()` on a git repository with a token in a notebook
and a log exits 1, names both files, and its stdout and stderr hold neither the token nor any 16-character piece of
it; the allowlist suppresses only the keyed (path, hash) pair.

Implementation commit `dc756bb`: `tests/test_release_policy.py` 58 passed.

## 2. Scans

Method: each ref's tree was extracted with `git archive` into a scratch directory and passed to `check()` from this
branch's checker (every tracked file: name, size, answer-field, job-output, notebook, run-log and token rules), plus
`name_violations()` over every path added in `origin/stage2/wf-integration..<ref>`. The integration head was also
checked with the CLI (`python3 tools/check_release_policy.py origin/stage2/wf-integration` on this branch: 0
violations in 1 changed, 618 tracked, 0 history paths). Fetched 2026-11-23; the second fetch at the end of this lane
showed the same heads, so the table is the final state.

| Branch | SHA | Tracked files | Notebooks | Run logs | New job / notebook / log files vs integration | Findings |
|---|---|---|---|---|---|---|
| `stage2/wf-integration` | `9d99ae4` | 618 | 101 | 83 | (base) | 0 |
| `wf/r2-lane4` | `22b9c16` | 642 | 105 | 83 | `job18_07_finetune_checkpoints.ipynb`, `job18_08_monitor_checkpoints.ipynb`, `job18_dw.jsonl`, `job18_job.json`, `job18_monitor.jsonl`, `job18_summary.json`, `job18_train_log.json`, `job21_07_finetune_checkpoints.ipynb`, `job21_08_monitor_checkpoints.ipynb`, `job21_dw.jsonl`, `job21_job.json`, `job21_monitor.jsonl`, `job21_summary.json`, `job21_train_log.json` (all under `kaggle/executed/12_hf_jobs/files/`) | 0 |
| `wf/r3-lane10` | `f1d2482` | 620 | 101 | 83 | none yet (Job 24 outputs not committed) | 0 |
| `wf/r3-lane12` | `0cc34f0` | 622 | 101 | 83 | none yet (Job 23 outputs not committed) | 0 |
| `wf/r3-lane6` | `a056ffb` | 619 | 101 | 83 | none (only `docs/report/workflow/figure_sources.md` changed) | 0 |
| `wf/r3-lane19` (this branch) | see head commit | 618 + this file | 101 | 83 | none | 0 |

`wf/r3-lane4` was not on `origin` at either fetch, so it was not scanned; its Job 21 notebooks are already on
`wf/r2-lane4` (rows above) and were scanned there.

Volume scanned on the integration tree: 8,334 notebook output lines and 2,297 run-log lines. As a cross-check that
the zero is not a matching gap, a looser case-insensitive search of the same lines for the words `sure`, `here is`,
`here's` or `step 1` found 0 lines. A `grep -E` of every tracked file for the four token patterns also found none.

**Findings: 0 on every branch scanned.** (File:cell list: empty.)

## 3. Owner decision, not a violation: Job 20 `summary.json`

Lane 6 did not commit Job 20's `summary.json` (`kaggle/executed/12_hf_jobs/RUN_SUMMARY.md`, Job 20 entry, on
`wf/r3-lane6`): it holds one 105-character string, its `rule` field (the alarm rule text written by
`tools/hf_jobs/job20_h4_multilingual.py` line 251), which the strict job-output rule (no free text over 80 characters) rejects.
It is job-script text, not model or translated text. It stays in the private jobs repository and is recomputed by
`tools/h4_multilingual.py` from the committed files. LEDGER round 2, open item 7 asks the owner whether to commit it.
This is a pending owner decision, not a release-policy violation; if the owner approves, the file can be committed
with its path in `ALLOWED` or with the rule string removed from the copy.

## 4. Verdict

Pass. No token, compliance opener, free-text line, withheld file or answer field in the integration head
`9d99ae4` or in any scanned branch. The checker now covers notebook outputs and run logs with redacted reports and a
hash-keyed allowlist.

Known limits: the opener and prose rules are heuristics. A short harmful answer that opens differently (for example
"To do this, first ...") or sits inside a dict repr is not caught; job outputs are covered by the stricter JSON rules
instead. Token patterns cover the three providers this project uses plus GitHub fine-grained tokens. Old-style
Kaggle `kaggle.json` keys (32 hex characters with no prefix) look like any hex hash and are not detectable this way.
