#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUTHOR=["Đăng Khoa","i.am@dangkhoa.dev","Đăng Khoa","i.am@dangkhoa.dev"]
RUNTIME_SHA="d04bcd3e601b449b67d09ff1132cab965619d858"
PREFIX=[
"b5bf930919d58a887ccfe88eb33b3ddb3fd5aa3b","0c301c507ee80a3ebe00126d7bbe00c9f5b6561c",
"b1a2059a6257f2c52c3713e279ae5574c88d4265","d600c548204d8de7123b97f6105631cf69d46883",
"cd08aa011b1b584ea22e1a1b6bd8471a4f68d719","a99aa3599024a069fd01e4882b2708a5bfc2756b",
"ac490e59c58d294c2670c2c4493c0aba04eed671","fe1fa2d0a107fcd304030cf290d2291de63ef725",
"87126d1ee2dbdc5886ef3c47d80483617e3084a1","1eea2351b4f5ba393969515d49150a0c6333ced9",
"5b985082492f9546f21064bb99c93b42aaacc9d3","d217c51ac0a2e112fd8463fd03b46a68f9b82cb0",
"78823e26630cbbd9fa32dce6dbf8c486eb9845cf","1ca871770147cf476a1b96cd64a0fa53334bf5ab",
"47792a9d06056de40a618433a4cc2e3425661470","aee5afe581cae20d38a16a2e4a5133b88a790237",
"2b7508599cfc354e4dc32524a5b45a3825e01090","e0b8dd47f1a065399ca2a44f3d528425a52186d4",
"ca1bc2ca4dd3bc481b805da98ed11db394e7a5c9","f82b2b754809bf39b86d432de3ae08d924b890cd",
"31332a3138a2dc5800b7de6bf5b2052e809c12e8","7b97b0fdf81ec73d73bee604526b1bf73e365cb6",
"61bac5fd08d248344dda1770592c86bb76fe8987","016e4ac0ac05fde3f3500a5f908c545f2f2a5add",
"438b071d1b55b63ddc8fe5323c7e3732ad99f638","d04bcd3e601b449b67d09ff1132cab965619d858",
]
DOC_PAIRS=[
"docs/index","docs/architecture","docs/features","docs/kaggle","docs/python-runtime",
"docs/api","docs/model-and-data","docs/qdrant","docs/retrieval-and-evaluation",
"docs/reproducibility","docs/limitations","docs/troubleshooting","docs/development",
"docs/roadmap","docs/releases/v1.0.0","CHANGELOG",
]
def git(*a): return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
def fail(m): raise SystemExit(m)
commits=git("rev-list","--reverse","HEAD").splitlines()
if len(commits)!=28: fail(f"expected 28 commits, got {len(commits)}")
if commits[:26]!=PREFIX: fail("frozen first 26 commits changed")
if git("rev-list","--merges","HEAD"): fail("merge commit found")
for c in commits:
    if git("show","-s","--format=%an%x00%ae%x00%cn%x00%ce",c).split("\0")!=AUTHOR:
        fail(f"identity mismatch at {c}")
if git("show","-s","--format=%s","HEAD")!="docs: publish bilingual project guide":
    fail("unexpected final commit subject")
paths={x for x in git("diff-tree","--root","--no-commit-id","--name-only","-r","HEAD").splitlines() if x}
if paths!={"README.md","README.vi.md"}: fail(f"final commit paths={sorted(paths)}")
if git("show","-s","--format=%s","HEAD^")!="feat(project): publish public demo and documentation suite":
    fail("unexpected penultimate commit subject")
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
if (ROOT/"VERSION").read_text().strip()!="1.0.0": fail("VERSION mismatch")
lic=(ROOT/"LICENSE").read_text()
if not lic.startswith("MIT License\n\nCopyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>\n"):
    fail("MIT license header mismatch")
nb=(ROOT/"notebooks/kaggle-production-demo-thin.ipynb").read_text()
if RUNTIME_SHA not in nb or "requirements-demo.txt" not in nb:
    fail("notebook authority changed")
for p in ["wemm_kaggle/search","tests/search","requirements-search.txt","kaggle/run-search.sh","scripts/serve_search.py","scripts/ingest_wikidata.py","scripts/benchmark_search.py"]:
    if (ROOT/p).exists(): fail(f"legacy residue returned: {p}")
ci=(ROOT/".github/workflows/ci.yml").read_text()
for pin in ["actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"]:
    if pin not in ci: fail(f"missing immutable action pin {pin}")
if "permissions:\n  contents: read" not in ci: fail("CI permissions not read-only")
if "python scripts/check_doc_links.py" not in ci: fail("CI does not validate Markdown links")
print("PUBLICATION_REPOSITORY_POLICY=PASS")
print("RUNTIME_PIN="+RUNTIME_SHA)
print("CURATED_COMMIT_COUNT=28")
print("FROZEN_PREFIX_COMMITS=26")
print(f"CANONICAL_DOC_PAIRS={len(DOC_PAIRS)}")
