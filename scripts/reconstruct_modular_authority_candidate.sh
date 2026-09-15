#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

FROZEN="d04bcd3e601b449b67d09ff1132cab965619d858"
OLD_PROJECT="99e3b66706887d4892093263ef6aaf692b8363b9"
OLD_FINAL="37b1ef78b1b7bfa46bbeda144848eb8987b24938"
WIP_REF="origin/ops/modular-atomic-runner-20260915"
TARGET="candidate/modular-atomic-runner-authority-20260915"

git fetch origin main ops/modular-atomic-runner-20260915

TMP="$(mktemp -d -t wemm-modular-authority.XXXXXX)"
cleanup() {
  git -C "$REPO_ROOT" worktree remove --force "$TMP" >/dev/null 2>&1 || true
  rm -rf "$TMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git worktree add --detach "$TMP" "$FROZEN" >/dev/null
cd "$TMP"

git config user.name "Đăng Khoa"
git config user.email "i.am@dangkhoa.dev"

git checkout "$OLD_PROJECT" -- .
git checkout "$WIP_REF" --   wemm_notebook/__init__.py   wemm_notebook/bootstrap.py   wemm_notebook/runner.py   wemm_notebook/text_showcase.py   wemm_notebook/image_showcase.py   wemm_notebook/visual_showcase.py   wemm_notebook/closeout.py   notebooks/kaggle-production-demo-thin.ipynb   tests/test_modular_public_notebook.py

git add -A
git commit -m "feat(project): publish public demo and documentation suite"

git checkout "$OLD_FINAL" -- README.md README.vi.md
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

git push --force origin "HEAD:refs/heads/$TARGET"

echo "RECONSTRUCT_MODULAR_AUTHORITY=PASS"
echo "AUTHORITY_CANDIDATE=$TARGET"
echo "AUTHORITY_CANDIDATE_SHA=$(git rev-parse HEAD)"
echo "AUTHORITY_CANDIDATE_COMMIT_COUNT=$(git rev-list --count HEAD)"
echo "AUTHORITY_CANDIDATE_IDENTITY=$(git show -s --format='%an <%ae> / %cn <%ce>' HEAD)"
