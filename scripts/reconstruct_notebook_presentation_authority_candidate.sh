#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

FROZEN="d04bcd3e601b449b67d09ff1132cab965619d858"
CURRENT_PROJECT="2cfd14cb60a495230877142c83e8d21dfdeb59a7"
CURRENT_FINAL="cdd15001fd272afbd29fbf093e7c020edcbe8046"
CORRECTIVE_REF="origin/ops/notebook-presentation-cells-20260915"
TARGET="candidate/notebook-presentation-authority-20260915"

git fetch origin main ops/notebook-presentation-cells-20260915

TMP="$(mktemp -d -t wemm-notebook-presentation-authority.XXXXXX)"
cleanup() {
  git -C "$REPO_ROOT" worktree remove --force "$TMP" >/dev/null 2>&1 || true
  rm -rf "$TMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git worktree add --detach "$TMP" "$FROZEN" >/dev/null
cd "$TMP"

git config user.name "Đăng Khoa"
git config user.email "i.am@dangkhoa.dev"

# Recreate canonical commit 27 from the current accepted project tree,
# replacing only the presentation notebook + structural guard test.
git checkout "$CURRENT_PROJECT" -- .
git checkout "$CORRECTIVE_REF" --   notebooks/kaggle-production-demo-thin.ipynb   tests/test_modular_public_notebook.py

git add -A
git commit -m "feat(project): publish public demo and documentation suite"

# Recreate canonical commit 28 as README-only.
git checkout "$CURRENT_FINAL" -- README.md README.vi.md
git add README.md README.vi.md
git commit -m "docs: publish bilingual project guide"

test "$(git rev-list --count HEAD)" = "28"
test -z "$(git rev-list --merges HEAD)"
test "$(git show -s --format=%an HEAD)" = "Đăng Khoa"
test "$(git show -s --format=%ae HEAD)" = "i.am@dangkhoa.dev"
test "$(git show -s --format=%cn HEAD)" = "Đăng Khoa"
test "$(git show -s --format=%ce HEAD)" = "i.am@dangkhoa.dev"

mapfile -t FINAL_PATHS < <(
  git diff-tree --root --no-commit-id --name-only -r HEAD | sort
)
test "${#FINAL_PATHS[@]}" = "2"
test "${FINAL_PATHS[0]}" = "README.md"
test "${FINAL_PATHS[1]}" = "README.vi.md"

python scripts/check_publication_policy.py
python -m pytest -q tests/test_modular_public_notebook.py

git push --force origin "HEAD:refs/heads/$TARGET"

echo "RECONSTRUCT_NOTEBOOK_PRESENTATION_AUTHORITY=PASS"
echo "AUTHORITY_CANDIDATE=$TARGET"
echo "AUTHORITY_CANDIDATE_SHA=$(git rev-parse HEAD)"
echo "AUTHORITY_CANDIDATE_COMMIT_COUNT=$(git rev-list --count HEAD)"
echo "AUTHORITY_CANDIDATE_IDENTITY=$(git show -s --format='%an <%ae> / %cn <%ce>' HEAD)"
echo "AUTHORITY_NOTEBOOK_LAYOUT=8_MARKDOWN_PLUS_1_ATOMIC_CODE"
