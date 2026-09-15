#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(*a): return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
def counterpart(path):
    p=Path(path); n=p.name
    other=n[:-6]+".md" if n.endswith(".vi.md") else n[:-3]+".vi.md"
    return str(p.with_name(other)).replace("\\","/")
def banner(path):
    p=Path(path); n=p.name
    if n.endswith(".vi.md"):
        return f"> 🌐 Language / Ngôn ngữ: [English]({n[:-6]}.md) | **Tiếng Việt**"
    return f"> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt]({n[:-3]}.vi.md)"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--history",action="store_true"); args=ap.parse_args()
    docs=[x for x in git("ls-files","*.md").splitlines() if x]; ds=set(docs); err=[]
    for p in docs:
        q=counterpart(p)
        if q not in ds: err.append(f"{p}: missing {q}"); continue
        top="\n".join((ROOT/p).read_text(encoding="utf-8").splitlines()[:8])
        if banner(p) not in top: err.append(f"{p}: missing exact language banner")
    if args.history:
        for c in git("rev-list","--reverse","HEAD").splitlines():
            changed={x for x in git("diff-tree","--root","--no-commit-id","--name-only","-r",c).splitlines() if x.endswith(".md")}
            for p in changed:
                if counterpart(p) not in changed: err.append(f"{c[:12]}: {p} changed without {counterpart(p)}")
    if err: raise SystemExit("\n".join(sorted(err)))
    print("BILINGUAL_DOCUMENTATION_POLICY=PASS")
if __name__=="__main__": main()
