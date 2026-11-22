"""Public release of the Stage 2 Hub repos (docs/10, docs/11 release policy).

Usage: python tools/hub_stage2_release.py STEP [STEP ...]
Steps, in order (each is idempotent):
  configs      rewrite every adapter_config.json: base_model_name_or_path = the Hub model id (training recorded the
               Kaggle mirror path /tmp/sp_models/...), revision = the pinned base revision. One commit per model repo.
  manifest     update the SHA-256 and size of those configs in each experiment's MANIFEST.json, with a note.
  cards        upload docs/hub_cards/stage2/<repo>.md as each repo's README.md.
  tag          create <tag>-v1.1 at the current head (the <tag>-v1 tags stay on the verified original uploads).
  publish      make the repos public.
  collections  add the repos to the SafetyPulse model and data collections, with a note per item.
  verify       anonymous check: public, tag resolves, configs name the Hub id, manifest SHAs match the files.
  carddiff     read-only: unified diff of each repo's Hub README.md against its git card (also covers
               safetypulse-h6-safetune, whose card is produced by tools/h6_card.py and released by hand).

Never touches the repos that stay private (safetypulse-jobs, safetypulse-withheld, safetypulse-earth-probe-bundle).
Uses the local `hf auth login`; prints no token.
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "docs/hub_cards/stage2"
OWNER = "purvansh01"
EARTH = ("CohereLabs/tiny-aya-earth", "0624fafbe200f930b89566fc4ec4da314ba48d89")
GLOBAL = ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324")
# repo -> (type, base model or None, logs repo holding its MANIFEST or None, version tag, collection note)
REPOS = {
    "safetypulse-earth-lora-alpaca": ("model", EARTH, "safetypulse-earth-logs", "h1-v1",
                                      "Stage 2 · H1: Tiny Aya Earth, Alpaca seeds 1–3, 78 checkpoints (confirmed)"),
    "safetypulse-earth-lora-none": ("model", EARTH, "safetypulse-earth-logs", "h1-v1",
                                    "Stage 2 · H1: Tiny Aya Earth, matched control, 52 checkpoints"),
    "safetypulse-h3-lora-alpaca": ("model", GLOBAL, "safetypulse-h3-logs", "h3-v1",
                                   "Stage 2 · H3: Alpaca seeds 4–6 at 10-step spacing, 78 checkpoints (refuted: ties ‖ΔW‖)"),
    "safetypulse-h2-lora-alpacalist": ("model", GLOBAL, "safetypulse-h2-logs", "h2-v1",
                                       "Stage 2 · H2: list-formatted Alpaca, seeds 1–3, 78 checkpoints (confirmed)"),
    "safetypulse-earth-logs": ("dataset", None, None, "h1-v1", "Stage 2 · H1 logs: per-checkpoint CIS and ASR on Tiny Aya Earth"),
    "safetypulse-h3-logs": ("dataset", None, None, "h3-v1", "Stage 2 · H3 logs: CIS, ASR and ‖ΔW‖ per checkpoint"),
    "safetypulse-h2-logs": ("dataset", None, None, "h2-v1", "Stage 2 · H2 logs and the list-formatted Alpaca training data"),
    "safetypulse-judge8b": ("dataset", None, None, "judge8b-v1",
                            "Stage 2 · Judge robustness: Llama Guard 3 1B vs 8B on all H1/H3 checkpoints"),
}
NEVER_PUBLIC = {"safetypulse-jobs", "safetypulse-withheld", "safetypulse-earth-probe-bundle"}
COLLECTIONS = {"model": f"{OWNER}/safetypulse-models-6af00b8f56c773a80af9f7e0",
               "dataset": f"{OWNER}/safetypulse-data-6af00b9124ee9d5022a36482"}
FIX_NOTE = ("2026-11-21: adapter_config.json rewritten for the public release (base_model_name_or_path = the Hub id "
            "instead of the training machine's local mirror path; revision = the pinned base revision). Adapter "
            "weights unchanged. The verified original upload stays at tag <tag>-v1.")
api = HfApi()
assert not NEVER_PUBLIC & set(REPOS)


def rid(name):
    return f"{OWNER}/{name}"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fixed_config(raw, base):
    cfg = json.loads(raw)
    cfg["base_model_name_or_path"], cfg["revision"] = base
    return (json.dumps(cfg, indent=2, sort_keys=True) + "\n").encode()


def configs_of(name):
    return [f for f in api.list_repo_files(rid(name)) if f.endswith("adapter_config.json")]


def step_configs():
    for name, (rtype, base, *_ ) in REPOS.items():
        if rtype != "model":
            continue
        files = configs_of(name)
        with ThreadPoolExecutor(16) as ex:
            raws = list(ex.map(lambda f: Path(hf_hub_download(rid(name), f, force_download=True)).read_bytes(), files))
        ops = [CommitOperationAdd(f, fixed_config(r, base))
               for f, r in zip(files, raws, strict=True) if fixed_config(r, base) != r]
        if ops:
            api.create_commit(rid(name), ops, commit_message=f"Adapter configs: base model {base[0]} (Hub id) and revision")
        print(name, len(files), "configs,", len(ops), "rewritten")


def step_manifest():
    for logs in sorted({v[2] for v in REPOS.values() if v[2]}):
        p = hf_hub_download(rid(logs), "MANIFEST.json", repo_type="dataset", force_download=True)
        m = json.loads(Path(p).read_text())
        changed = 0
        for name, (_rtype, _base, lg, tag, _) in REPOS.items():
            if lg != logs:
                continue
            files = m["repos"][rid(name)]["files"]
            for f in [f for f in files if f.endswith("adapter_config.json")]:
                b = Path(hf_hub_download(rid(name), f)).read_bytes()
                if files[f]["sha256"] != sha(b):
                    files[f] = {"sha256": sha(b), "bytes": len(b)}
                    changed += 1
            note = FIX_NOTE.replace("<tag>", tag.rsplit("-", 1)[0])
        if changed or "post_release" not in m:
            m["post_release"] = [note]
            api.upload_file(path_or_fileobj=(json.dumps(m, indent=1) + "\n").encode(), path_in_repo="MANIFEST.json",
                            repo_id=rid(logs), repo_type="dataset",
                            commit_message="MANIFEST: checksums of the rewritten adapter configs")
        print(logs, changed, "checksums updated")


def step_cards():
    for name, (rtype, *_ ) in REPOS.items():
        card = CARDS / f"{name}.md"
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=rid(name), repo_type=rtype,
                        commit_message="Card for the public release")
        print("card", name)


def step_tag():
    for name, (rtype, _, _, tag, _) in REPOS.items():
        new = tag + ".1"
        tags = [t.name for t in api.list_repo_refs(rid(name), repo_type=rtype).tags]
        if new in tags:
            api.delete_tag(rid(name), tag=new, repo_type=rtype)
        api.create_tag(rid(name), tag=new, repo_type=rtype,
                       tag_message="Public release: corrected adapter configs and cards; weights identical to " + tag)
        print(name, "tag", new)


def step_publish():
    for name, (rtype, *_ ) in REPOS.items():
        api.update_repo_settings(rid(name), private=False, repo_type=rtype)
        print("public", name)


def step_collections():
    for rtype, slug in COLLECTIONS.items():
        have = {i.item_id for i in api.get_collection(slug).items}
        for name, (t, *_, note) in REPOS.items():
            if t == rtype and rid(name) not in have:
                api.add_collection_item(slug, item_id=rid(name), item_type=rtype, note=note)
                print("collection", rtype, name)


def step_verify():
    anon = HfApi(token=False)
    ok = True
    for name, (rtype, base, logs, tag, _) in REPOS.items():
        info = anon.repo_info(rid(name), repo_type=rtype)
        head = info.sha
        tagged = anon.repo_info(rid(name), repo_type=rtype, revision=tag + ".1").sha
        line = f"{name}: public {not info.private}, {tag}.1 == head {tagged == head}"
        ok &= (not info.private) and tagged == head
        if rtype == "model":
            m = json.loads(Path(hf_hub_download(rid(logs), "MANIFEST.json", repo_type="dataset", token=False,
                                                force_download=True)).read_text())["repos"][rid(name)]["files"]
            files = [f for f in anon.list_repo_files(rid(name)) if f.endswith("adapter_config.json")]
            with ThreadPoolExecutor(16) as ex:
                raws = list(ex.map(lambda f: Path(hf_hub_download(rid(name), f, token=False)).read_bytes(), files))
            bases = {json.loads(r)["base_model_name_or_path"] for r in raws}
            match = all(m[f]["sha256"] == sha(r) for f, r in zip(files, raws, strict=True))
            ok &= bases == {base[0]} and match
            line += f", configs {len(files)} base {bases}, manifest match {match}"
        print(line)
    print("ALL OK" if ok else "PROBLEMS FOUND")
    return ok


def step_carddiff():
    import difflib
    repos = {name: v[0] for name, v in REPOS.items()} | {"safetypulse-h6-safetune": "dataset"}
    same = True
    for name, rtype in repos.items():
        hub = Path(hf_hub_download(rid(name), "README.md", repo_type=rtype, force_download=True)).read_text()
        git = (CARDS / f"{name}.md").read_text()
        diff = list(difflib.unified_diff(hub.splitlines(True), git.splitlines(True),
                                         f"hub:{name}/README.md", f"git:docs/hub_cards/stage2/{name}.md"))
        same &= not diff
        print(f"{name}: {'same' if not diff else 'DIFFERS'}")
        sys.stdout.writelines(diff)
    return same


if __name__ == "__main__":
    steps = sys.argv[1:] or ["verify"]
    for s in steps:
        globals()[f"step_{s}"]()
