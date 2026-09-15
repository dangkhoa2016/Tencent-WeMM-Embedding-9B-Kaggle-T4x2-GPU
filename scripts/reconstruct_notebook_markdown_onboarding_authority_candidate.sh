#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

FROZEN_26="d04bcd3e601b449b67d09ff1132cab965619d858"
OLD_27="9ab67ac923547cb52c90ec7feb2a3900e8e79dee"
OLD_28="62531c22f865276a335dcfef9a6336d3513ed4a0"
CORRECTIVE_BRANCH="origin/candidate/notebook-markdown-onboarding-corrective-20260915"
CORRECTIVE_CONTENT_COMMIT="61c7577b6b230cb5f8b556a557c088438ab08cb9"
CANDIDATE_BRANCH="candidate/notebook-markdown-onboarding-authority-20260915"

AUTHOR_NAME="Đăng Khoa"
AUTHOR_EMAIL="i.am@dangkhoa.dev"
CANONICAL_DATE="2026-09-15T13:05:53Z"

NOTEBOOK="notebooks/kaggle-production-demo-thin.ipynb"
TEST_FILE="tests/test_modular_public_notebook.py"

echo "=== FETCH AUTHORITY INPUTS ==="
git fetch origin --prune --tags
git fetch origin candidate/notebook-markdown-onboarding-corrective-20260915

git merge-base --is-ancestor "$CORRECTIVE_CONTENT_COMMIT" "$CORRECTIVE_BRANCH"

echo "FROZEN_26=$FROZEN_26"
echo "OLD_27=$OLD_27"
echo "OLD_28=$OLD_28"
echo "CORRECTIVE_BRANCH_HEAD=$(git rev-parse "$CORRECTIVE_BRANCH")"
echo "CORRECTIVE_CONTENT_COMMIT=$CORRECTIVE_CONTENT_COMMIT"

echo "=== VERIFY EXISTING CANONICAL TOPOLOGY ==="
test "$(git rev-parse "$OLD_27^")" = "$FROZEN_26"
test "$(git rev-parse "$OLD_28^")" = "$OLD_27"
test "$(git rev-list --count "$FROZEN_26")" = "26"
test "$(git show -s --format=%s "$OLD_27")" = "feat(project): publish public demo and documentation suite"
test "$(git show -s --format=%s "$OLD_28")" = "docs: publish bilingual project guide"

OLD28_FILES="$(git diff-tree --no-commit-id --name-only -r "$OLD_28" | sort)"
EXPECTED_OLD28_FILES=$'README.md\nREADME.vi.md'
test "$OLD28_FILES" = "$EXPECTED_OLD28_FILES"

WT="$(mktemp -d /tmp/wemm-authority-reconstruct.XXXXXX)"
cleanup() {
  git -C "$REPO_ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || true
  rm -rf "$WT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== CREATE TEMPORARY WORKTREE ==="
git worktree add --detach "$WT" "$FROZEN_26" >/dev/null
cd "$WT"

git config user.name "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"

echo "=== RECONSTRUCT COMMIT 27 TREE ==="
git read-tree --reset -u "$OLD_27"
git checkout "$CORRECTIVE_CONTENT_COMMIT" -- "$NOTEBOOK" "$TEST_FILE"
git diff --cached --check

echo "Commit-27 changes vs previous canonical commit 27:"
git diff --cached --name-status "$OLD_27"

EXPECTED_OVERLAY=$'notebooks/kaggle-production-demo-thin.ipynb\ntests/test_modular_public_notebook.py'
ACTUAL_OVERLAY="$(git diff --cached --name-only "$OLD_27" | sort)"
test "$ACTUAL_OVERLAY" = "$EXPECTED_OVERLAY"

env \
  GIT_AUTHOR_NAME="$AUTHOR_NAME" \
  GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
  GIT_AUTHOR_DATE="$CANONICAL_DATE" \
  GIT_COMMITTER_NAME="$AUTHOR_NAME" \
  GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
  GIT_COMMITTER_DATE="$CANONICAL_DATE" \
  git commit -q -m "feat(project): publish public demo and documentation suite"

NEW_27="$(git rev-parse HEAD)"

echo "=== RECONSTRUCT COMMIT 28 ==="
git checkout "$OLD_28" -- README.md README.vi.md

STAGED_28="$(git diff --cached --name-only | sort)"
EXPECTED_28=$'README.md\nREADME.vi.md'
test "$STAGED_28" = "$EXPECTED_28"
git diff --cached --check

env \
  GIT_AUTHOR_NAME="$AUTHOR_NAME" \
  GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
  GIT_AUTHOR_DATE="$CANONICAL_DATE" \
  GIT_COMMITTER_NAME="$AUTHOR_NAME" \
  GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
  GIT_COMMITTER_DATE="$CANONICAL_DATE" \
  git commit -q -m "docs: publish bilingual project guide"

NEW_28="$(git rev-parse HEAD)"

echo "=== AUTHORITY STRUCTURAL VERIFY ==="
test "$(git rev-list --count HEAD)" = "28"
test "$(git rev-parse "$NEW_27^")" = "$FROZEN_26"
test "$(git rev-parse "$NEW_28^")" = "$NEW_27"
test "$(git show -s --format=%s "$NEW_27")" = "feat(project): publish public demo and documentation suite"
test "$(git show -s --format=%s "$NEW_28")" = "docs: publish bilingual project guide"

for sha in "$NEW_27" "$NEW_28"; do
  test "$(git show -s --format='%an <%ae>' "$sha")" = "$AUTHOR_NAME <$AUTHOR_EMAIL>"
  test "$(git show -s --format='%cn <%ce>' "$sha")" = "$AUTHOR_NAME <$AUTHOR_EMAIL>"
done

NEW28_FILES="$(git diff-tree --no-commit-id --name-only -r "$NEW_28" | sort)"
test "$NEW28_FILES" = "$EXPECTED_28"

echo "=== NOTEBOOK PRESENTATION VERIFY ==="
python - <<'PY'
import json
from pathlib import Path

p = Path("notebooks/kaggle-production-demo-thin.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
cells = nb["cells"]
md = [c for c in cells if c["cell_type"] == "markdown"]
code = [c for c in cells if c["cell_type"] == "code"]

assert len(cells) == 10
assert len(md) == 5
assert len(code) == 5
assert [c["cell_type"] for c in cells] == ["markdown", "code"] * 5
assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)

setup = "".join(md[0]["source"])
for marker in (
    "Add Input",
    "GPU T4 ×2",
    "Internet = ON",
    "dangkhoa2016/tencent-wemm-embedding-9b",
    "dangkhoa2016/wemm-embedding-9b-v1-qdrant-snapshots",
    "/kaggle/input",
):
    assert marker in setup, marker

assert "".join(code[2]["source"]).strip() == "image_results = run_step7a(demo)"
assert "".join(code[3]["source"]).strip() == "visual_results = run_step7b(demo)"

meta = nb["metadata"]["wemm_public_demo"]
assert meta["runtime_commit"] == "d04bcd3e601b449b67d09ff1132cab965619d858"
assert meta["science_reopened"] is False
assert meta["frozen_examples_changed"] is False
assert meta["presentation_markdown_cells"] == 5
assert meta["executable_code_cells"] == 5
assert meta["step_7a_7b_separate_code_cells"] is True
assert meta["markdown_onboarding_restored"] is True
assert meta["per_phase_guidance_expanded"] is True

print("NOTEBOOK_MARKDOWN_ONBOARDING=PASS")
print("NOTEBOOK_LAYOUT=5_MARKDOWN_PLUS_5_CODE")
print("STEP_7A_7B_SEPARATE_CELLS=PASS")
PY

echo "=== POLICY + STRUCTURAL TESTS ==="
python scripts/check_bilingual_docs.py --history
python scripts/check_doc_links.py
python scripts/check_publication_policy.py
python -m pytest -q tests/test_modular_public_notebook.py
python -m compileall -q wemm_runtime wemm_kaggle wemm_notebook scripts
bash -n kaggle/*.sh
git diff --check

echo "=== PUSH AUTHORITY CANDIDATE ONLY ==="
git push --force origin HEAD:"refs/heads/$CANDIDATE_BRANCH"

echo "=== FINAL AUTHORITY SUMMARY ==="
echo "RECONSTRUCT_NOTEBOOK_MARKDOWN_ONBOARDING_AUTHORITY=PASS"
echo "AUTHORITY_CANDIDATE=$CANDIDATE_BRANCH"
echo "AUTHORITY_COMMIT_27=$NEW_27"
echo "AUTHORITY_CANDIDATE_SHA=$NEW_28"
echo "AUTHORITY_CANDIDATE_COMMIT_COUNT=$(git rev-list --count HEAD)"
echo "AUTHORITY_CANDIDATE_IDENTITY=$(git show -s --format='%an <%ae> / %cn <%ce>' HEAD)"
echo "FROZEN_PREFIX_HEAD=$FROZEN_26"
echo "NOTEBOOK_LAYOUT=5_MARKDOWN_PLUS_5_CODE"
echo "MARKDOWN_ONBOARDING_RESTORED=PASS"
echo "STEP_7A_7B_SEPARATE_CELLS=PASS"
echo
echo "NEXT: wait for GitHub Actions CI on $CANDIDATE_BRANCH."
echo "DO NOT update main or v1.0.0 until candidate CI passes."
