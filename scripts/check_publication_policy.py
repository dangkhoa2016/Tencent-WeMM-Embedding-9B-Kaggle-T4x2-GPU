#!/usr/bin/env python3
"""Stage validation for the corrected public documentation suite.

Validates the documentation-suite commit against the corrected public
history: canonical identity, commit count, merge-free history, complete
bilingual documentation pairs, license header, notebook authority using the
corrected runtime pin, and the CI guardrails introduced from this commit.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ["Đăng Khoa", "i.am@dangkhoa.dev", "Đăng Khoa", "i.am@dangkhoa.dev"]
RUNTIME_SHA = "224f07cd1d6eb174d3532c9eaeeb9abd606a857f"
DOC_PAIRS = [
"docs/index","docs/architecture","docs/features","docs/kaggle","docs/python-runtime",
"docs/api","docs/model-and-data","docs/qdrant","docs/retrieval-and-evaluation",
"docs/reproducibility","docs/limitations","docs/troubleshooting","docs/development",
"docs/roadmap","docs/releases/v1.0.0","CHANGELOG",
]
def git(*a): return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
def fail(m): raise SystemExit(m)
commits=git("rev-list","--reverse","HEAD").splitlines()
if len(commits)!=37: fail(f"expected 37 commits, got {len(commits)}")
if git("rev-list","--merges","HEAD"): fail("merge commit found")
for c in commits:
    if git("show","-s","--format=%an%x00%ae%x00%cn%x00%ce",c).split("\0")!=AUTHOR:
        fail(f"identity mismatch at {c}")
if git("show","-s","--format=%s","HEAD")!="feat(project): publish public demo and documentation suite":
    fail("unexpected final commit subject")
required=[
"LICENSE",".github/CODEOWNERS",".github/CONTRIBUTING.md",".github/CONTRIBUTING.vi.md",
".github/CODE_OF_CONDUCT.md",".github/CODE_OF_CONDUCT.vi.md",".github/SECURITY.md",".github/SECURITY.vi.md",
".github/SUPPORT.md",".github/SUPPORT.vi.md",".github/pull_request_template.md",".github/pull_request_template.vi.md",
".github/ISSUE_TEMPLATE/bug_report.yml",".github/ISSUE_TEMPLATE/bug_report_vi.yml",
".github/ISSUE_TEMPLATE/feature_request.yml",".github/ISSUE_TEMPLATE/feature_request_vi.yml",
".github/ISSUE_TEMPLATE/config.yml",".github/workflows/ci.yml",
"scripts/check_bilingual_docs.py","scripts/check_doc_links.py","scripts/check_publication_policy.py",
]
for base in DOC_PAIRS:
    required += [f"{base}.md",f"{base}.vi.md"]
missing=[p for p in required if not (ROOT/p).exists()]
if missing: fail(f"missing required files: {missing}")
lic=(ROOT/"LICENSE").read_text()
if not lic.startswith("MIT License\n\nCopyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>\n"):
    fail("MIT license header mismatch")
nb=(ROOT/"notebooks/kaggle-production-demo-thin.ipynb").read_text()
if RUNTIME_SHA not in nb or "requirements-demo.txt" not in nb:
    fail("notebook authority changed")
ci=(ROOT/".github/workflows/ci.yml").read_text()
for pin in ["actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"]:
    if pin not in ci: fail(f"missing immutable action pin {pin}")
if "permissions:\n  contents: read" not in ci: fail("CI permissions not read-only")
if "python scripts/check_doc_links.py" not in ci: fail("CI does not validate Markdown links")
print("PUBLICATION_REPOSITORY_POLICY=PASS")
print("RUNTIME_PIN="+RUNTIME_SHA)
print("CURATED_COMMIT_COUNT=37")
print(f"CANONICAL_DOC_PAIRS={len(DOC_PAIRS)}")
