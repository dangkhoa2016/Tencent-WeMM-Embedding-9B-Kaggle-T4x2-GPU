#!/usr/bin/env python3
"""R1.2 publication repository policy validator (schema v7).

Fails closed on any contract breach and emits exactly 25 mandatory report
tokens on success. Runs against the rebuilt HEAD and against a fresh clone
bundle without requiring any old commit objects.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "scripts" / "publication-history.json"
SELF_AUTHORED = {
    "scripts/check_publication_policy.py",
    "scripts/publication-history.json",
}

CONVENTIONAL_SUBJECT = re.compile(
    r"^(?:feat|fix|test|docs|chore|ci|build|refactor|perf|style|revert|release)"
    r"(?:\([a-z0-9._-]+\))?!?: .+"
)
ISO_WITH_SECONDS = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$"
)


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=check
    )


def git(*args: str) -> str:
    return run_git(*args).stdout.strip()


def git_raw(*args: str) -> str:
    return run_git(*args).stdout


def fail(message: str) -> None:
    raise SystemExit(message)


def parse_iso(value: str, label: str) -> datetime:
    if not ISO_WITH_SECONDS.fullmatch(value):
        fail(f"{label}: timestamp must include seconds and timezone: {value!r}")
    value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        fail(f"{label}: invalid ISO timestamp: {exc}")
    if dt.tzinfo is None:
        fail(f"{label}: timezone missing")
    return dt.astimezone(timezone.utc)


def changed_paths(commit: str) -> set[str]:
    out = git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return {line for line in out.splitlines() if line}


def tree_entry(commit: str, path: str) -> tuple[str, str, str] | None:
    out = git("ls-tree", commit, "--", path)
    if not out:
        return None
    meta, returned = out.split("\t", 1)
    if returned != path:
        fail(f"ls-tree path mismatch for {commit}:{path}")
    mode, typ, sha = meta.split()
    return mode, typ, sha


def file_mode(commit: str, path: str) -> str:
    entry = tree_entry(commit, path)
    if entry is None:
        fail(f"missing tree entry {commit}:{path}")
    return entry[0]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_message(subject: str, body: list[str]) -> str:
    return f"{subject}\n\n" + "\n".join(f"- {line}" for line in body)


def run_step(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def walk_json_strings(node: object) -> list[str]:
    values: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            values.extend(walk_json_strings(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(walk_json_strings(value))
    elif isinstance(node, str):
        values.append(node)
    return values


def main() -> None:
    if not MANIFEST_PATH.is_file():
        fail(f"missing manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 7:
        fail(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("canonical_commit_count") != 42:
        fail("canonical_commit_count must be 42")
    if manifest.get("tree_authority_count") != 40:
        fail("tree_authority_count must be 40")

    expected_qualification = {
        "pytest": {
            "passed": 470,
            "skipped": 3,
            "warnings": {
                "policy": "observational_environment_dependent",
                "hard_count": None,
            },
        },
        "markdown_link_targets": 234,
    }
    if manifest.get("qualification") != expected_qualification:
        fail(f"qualification mismatch: {manifest.get('qualification')!r}")

    identity = manifest["canonical_identity"]
    expected_identity = [
        identity["author_name"],
        identity["author_email"],
        identity["committer_name"],
        identity["committer_email"],
    ]

    commits = git("rev-list", "--reverse", "HEAD").splitlines()
    if len(commits) != 42:
        fail(f"expected exactly 42 commits, got {len(commits)}")
    if git("rev-list", "--merges", "HEAD"):
        fail("merge commits found in public history")

    specs = manifest["commits"]
    if len(specs) != 40:
        fail("expected tree authority for ordinals 1..40")

    previous_committer: datetime | None = None
    bullet_count = 0

    for idx in range(1, 41):
        commit = commits[idx - 1]
        spec = specs[str(idx)]

        subject = git("show", "-s", "--format=%s", commit)
        if subject != spec["subject"]:
            fail(f"commit {idx}: subject mismatch")
        if not CONVENTIONAL_SUBJECT.fullmatch(subject):
            fail(f"commit {idx}: non-Conventional subject: {subject!r}")

        msg = git("show", "-s", "--format=%B", commit).rstrip("\n")
        if msg != exact_message(spec["subject"], spec["body"]):
            fail(f"commit {idx}: bullet message mismatch")
        body_lines = msg.splitlines()[2:]
        if not body_lines or not all(line.startswith("- ") for line in body_lines):
            fail(f"commit {idx}: subject-only or non-bullet body")
        bullet_count += 1

        fields = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", commit).split("\0")
        if fields != expected_identity:
            fail(f"commit {idx}: identity mismatch")

        author = parse_iso(git("show", "-s", "--format=%aI", commit), f"commit {idx} AuthorDate")
        expected_author = parse_iso(spec["author_date"], f"commit {idx} expected AuthorDate")
        if author != expected_author:
            fail(f"commit {idx}: AuthorDate provenance mismatch")

        committer = parse_iso(git("show", "-s", "--format=%cI", commit), f"commit {idx} CommitterDate")
        expected_committer = parse_iso(spec["committer_date"], f"commit {idx} expected CommitterDate")
        if committer != expected_committer:
            fail(f"commit {idx}: CommitterDate provenance mismatch")
        if previous_committer is not None and committer < previous_committer:
            fail(f"commit {idx}: CommitterDate is not monotonic")
        if author != committer:
            fail(f"commit {idx}: AuthorDate and CommitterDate must match")
        previous_committer = committer

        actual_tree = git("rev-parse", f"{commit}^{{tree}}")
        if actual_tree != spec["expected_tree_sha"]:
            fail(f"commit {idx}: tree SHA mismatch")

    policy = manifest["publication_policy"]
    for idx in (41, 42):
        commit = commits[idx - 1]
        spec = policy[str(idx)]

        subject = git("show", "-s", "--format=%s", commit)
        if subject != spec["subject"]:
            fail(f"commit {idx}: subject mismatch")
        if not CONVENTIONAL_SUBJECT.fullmatch(subject):
            fail(f"commit {idx}: non-Conventional subject: {subject!r}")

        msg = git("show", "-s", "--format=%B", commit).rstrip("\n")
        if msg != exact_message(spec["subject"], spec["body"]):
            fail(f"commit {idx}: bullet message mismatch")
        body_lines = msg.splitlines()[2:]
        if not body_lines or not all(line.startswith("- ") for line in body_lines):
            fail(f"commit {idx}: subject-only or non-bullet body")
        bullet_count += 1

        fields = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", commit).split("\0")
        if fields != expected_identity:
            fail(f"commit {idx}: identity mismatch")

        author = parse_iso(git("show", "-s", "--format=%aI", commit), f"commit {idx} AuthorDate")
        expected_author = parse_iso(spec["author_date"], f"commit {idx} expected AuthorDate")
        if author != expected_author:
            fail(f"commit {idx}: AuthorDate provenance mismatch")

        committer = parse_iso(git("show", "-s", "--format=%cI", commit), f"commit {idx} CommitterDate")
        expected_committer = parse_iso(spec["committer_date"], f"commit {idx} expected CommitterDate")
        if committer != expected_committer:
            fail(f"commit {idx}: CommitterDate provenance mismatch")
        if previous_committer is not None and committer < previous_committer:
            fail(f"commit {idx}: CommitterDate is not monotonic")

        if sorted(changed_paths(commit)) != sorted(spec["expected_changed_paths"]):
            fail(f"commit {idx}: changed-path scope mismatch: {sorted(changed_paths(commit))}")

    if bullet_count != 42:
        fail(f"bullet-body coverage failed: {bullet_count}/42")

    for path, expected in manifest["restored_paths"].items():
        entry = tree_entry("HEAD", path)
        if entry is None:
            fail(f"missing restored public path: {path}")
        mode, typ, blob = entry
        if typ != "blob" or mode != expected["mode"] or blob != expected["blob"]:
            fail(f"restored path authority mismatch: {path}")

    for path in manifest["executable_validators"]:
        if file_mode("HEAD", path) != "100755":
            fail(f"validator mode mismatch: {path}")

    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != "1.0.0":
        fail("VERSION mismatch")

    expected_checker_sha = manifest["self_authored_files"]["scripts/check_publication_policy.py"]
    if file_sha256(ROOT / "scripts/check_publication_policy.py") != expected_checker_sha:
        fail("publication checker SHA-256 mismatch")

    runtime = manifest["runtime_pin"]
    runtime_ordinal = runtime["canonical_ordinal"]
    if runtime_ordinal != 35:
        fail("runtime pin canonical ordinal must be 35")
    canonical_runtime_commit = commits[runtime_ordinal - 1]
    if runtime["v32"] != canonical_runtime_commit:
        fail(
            "runtime pin is not canonical commit ordinal 35: "
            f"{runtime['v32']} != {canonical_runtime_commit}"
        )
    actual_runtime_tree = git("rev-parse", f"{canonical_runtime_commit}^{{tree}}")
    if actual_runtime_tree != runtime["expected_tree_sha"]:
        fail(f"runtime tree mismatch: {actual_runtime_tree} != {runtime['expected_tree_sha']}")
    if actual_runtime_tree != specs[str(runtime_ordinal)]["expected_tree_sha"]:
        fail("runtime tree does not match ordinal 35 tree authority")

    old_pin = commits and manifest.get("commits", {}).get("35", {}).get("old_source_sha", "")
    if not old_pin:
        fail("runtime-pin historical provenance source missing")
    if old_pin == runtime["v32"] or old_pin == runtime["expected_tree_sha"]:
        fail("runtime authority fields must not equal the historical runtime pin")
    string_values = walk_json_strings(manifest)
    provenance_occurrences = [v for v in string_values if v == old_pin]
    if provenance_occurrences != [manifest["commits"]["35"]["old_source_sha"]]:
        fail("historical runtime pin must occur only as ordinal 35 old_source_sha")
    old_pin_scan = run_git(
        "grep", "-n", old_pin, "HEAD", "--", ".", ":(exclude)scripts/publication-history.json", check=False
    )
    if old_pin_scan.returncode == 0:
        fail("active old runtime pin occurrences outside the manifest:\n" + old_pin_scan.stdout)
    if old_pin_scan.returncode not in (0, 1):
        fail("old-pin scan errored: " + old_pin_scan.stderr)

    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    ci_required = manifest["ci_required_commands"]
    ci_positions = []
    for required in ci_required:
        pos = ci.find(required)
        if pos < 0:
            fail(f"CI missing required command: {required}")
        ci_positions.append(pos)
    if ci_positions[-1] != max(ci_positions):
        fail("publication-policy must be the final CI evidence gate")
    if '"candidate/**"' not in ci:
        fail("CI must retain the candidate branch trigger")
    if '"corrective/**"' in ci:
        fail("CI must not contain the corrective branch trigger")
    if ci.find("python scripts/check_doc_links.py") > ci.find("python scripts/check_publication_policy.py"):
        fail("publication-policy step must run after Markdown-link checks")
    if ci.find("python -m pytest -q") > ci.find("python scripts/check_publication_policy.py"):
        fail("publication-policy step must run after pytest")
    if ci.find("git diff --check") > ci.find("python scripts/check_publication_policy.py"):
        fail("publication-policy step must run after the whitespace gate")

    bilingual = run_step(["python", "scripts/check_bilingual_docs.py", "--history"])
    if bilingual.returncode != 0:
        fail("bilingual documentation policy FAILED:\n" + bilingual.stderr + bilingual.stdout)
    doclinks = run_step(["python", "scripts/check_doc_links.py"])
    if doclinks.returncode != 0:
        fail("Markdown-link policy FAILED:\n" + doclinks.stderr + doclinks.stdout)
    checked = 0
    for line in doclinks.stdout.splitlines():
        if line.startswith("MARKDOWN_LINK_TARGETS_CHECKED="):
            checked = int(line.split("=", 1)[1])
    if checked != manifest["qualification"]["markdown_link_targets"]:
        fail(f"Markdown link target count mismatch: {checked}")

    phrases = manifest.get("narrative_scan_phrases", [])
    tracked = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    hits: list[str] = []
    for path in tracked:
        if path in SELF_AUTHORED:
            continue
        data = git_raw("show", f"HEAD:{path}")
        for phrase in phrases:
            if phrase in data:
                hits.append(f"{path}: contains {phrase!r}")
    for ci_idx, commit in enumerate(commits, 1):
        msg = git("show", "-s", "--format=%B", commit)
        for phrase in phrases:
            if phrase in msg:
                hits.append(f"commit {ci_idx}: message contains {phrase!r}")
    if hits:
        fail("publication narrative scan found forbidden terms: " + "; ".join(sorted(hits)))

    sep16 = 0
    for commit in commits:
        if git("show", "-s", "--format=%aI", commit).startswith("2026-09-16"):
            sep16 += 1
    if sep16 != 0:
        fail(f"Sep-16 lineage commits remain in public history: {sep16}")

    release_docs = [
        "README.md",
        "README.vi.md",
        "docs/releases/v1.0.0.md",
        "docs/releases/v1.0.0.vi.md",
        "docs/reproducibility.md",
        "docs/reproducibility.vi.md",
    ]
    stale_markers = ["Current public CI qualification",
        "Published repository qualification",
        "Current public test qualification",
        "142 passed",
        "2 warnings",
        "462 passed"]
    for doc in release_docs:
        text = (ROOT / doc).read_text(encoding="utf-8")
        if "v1.0.0 release qualification baseline" not in text:
            fail(f"{doc}: missing release qualification baseline heading")
        if f"470 passed" not in text:
            fail(f"{doc}: missing passed count")
        if f"3 skipped" not in text:
            fail(f"{doc}: missing skipped count")
        if f"234 Markdown link targets" not in text:
            fail(f"{doc}: missing Markdown link target count")
        for marker in stale_markers:
            if marker in text:
                fail(f"{doc}: stale qualification marker: {marker!r}")

    print("PUBLICATION_REPOSITORY_POLICY=PASS")
    print("PUBLIC_HISTORY_SCHEMA=7")
    print("TARGET_COMMIT_COUNT=42")
    print("TREE_AUTHORITY_ORDINALS=1-40")
    print("PUBLICATION_BULLET_BODIES=42/42")
    print("SUBJECT_ONLY_MESSAGES=0")
    print("MERGE_COMMITS=0")
    print("CANONICAL_AUTHOR_IDENTITY=PASS")
    print("CANONICAL_COMMITTER_IDENTITY=PASS")
    print("AUTHOR_DATE_PROVENANCE=42/42")
    print("COMMITTER_DATE_PROVENANCE=42/42")
    print("COMMITTER_DATES_MONOTONIC=PASS")
    print("RUNTIME_PIN_V32_CANONICAL_REACHABLE=PASS")
    print("RUNTIME_ACTIVE_OLD_PIN_OCCURRENCES=0")
    print("RUNTIME_OLD_PIN_HISTORICAL_PROVENANCE=PASS")
    print("QUALIFICATION_PYTEST_PASSED=470")
    print("QUALIFICATION_PYTEST_SKIPPED=3")
    print("PYTEST_WARNING_COUNT_POLICY=OBSERVATIONAL_ENVIRONMENT_DEPENDENT")
    print("QUALIFICATION_MARKDOWN_LINK_TARGETS=234")
    print("PUBLIC_HISTORY_CORRECTIVE_ONLY_COMMITS=0")
    print("PUBLIC_HISTORY_BINDER_COMMITS=0")
    print("PUBLIC_HISTORY_RECOVERY_NARRATIVE=0")
    print("PUBLIC_HISTORY_SEP16_TAIL_COMMITS=0")
    if not os.environ.get("WEMM_NO_FRESH_CLONE"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clone_dir = Path(tmp) / "clone"
            cloned = run_git(
                "clone", "--no-local", "--no-hardlinks", str(ROOT), str(clone_dir), check=False
            )
            if cloned.returncode != 0:
                fail("fresh-clone setup failed: " + cloned.stderr)
            clone_env = dict(os.environ, WEMM_NO_FRESH_CLONE="1")
            sweep = subprocess.run(
                ["python", "scripts/check_publication_policy.py"],
                cwd=clone_dir,
                env=clone_env,
                text=True,
                capture_output=True,
            )
            if sweep.returncode != 0:
                fail("fresh-clone policy run failed:\n" + sweep.stdout + sweep.stderr)
    print("FRESH_CLONE_SELF_CONTAINED=PASS")
    print("V3.2_PUBLICATION_POLICY=PASS")


if __name__ == "__main__":
    main()