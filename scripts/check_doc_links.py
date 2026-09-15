#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SCHEMES = ("http://", "https://", "mailto:", "tel:")

def markdown_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.md") if ".git" not in path.parts)

def normalize_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split(" ", 1)[0]
    return unquote(target)

def check_target(source: Path, raw: str) -> str | None:
    target = normalize_target(raw)
    if not target or target.startswith("#") or target.startswith(SCHEMES):
        return None
    local = target.split("#", 1)[0].split("?", 1)[0]
    if not local:
        return None
    resolved = (ROOT / local.lstrip("/")) if local.startswith("/") else (source.parent / local)
    try:
        resolved = resolved.resolve()
        resolved.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return f"{source.relative_to(ROOT)}: link escapes repository: {target}"
    if not resolved.exists():
        return f"{source.relative_to(ROOT)}: missing local target: {target}"
    return None

def main() -> None:
    errors: list[str] = []
    checked = 0
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for regex in (LINK_RE, IMAGE_RE):
            for match in regex.finditer(text):
                checked += 1
                error = check_target(source, match.group(1))
                if error:
                    errors.append(error)
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print("MARKDOWN_LOCAL_LINKS=PASS")
    print(f"MARKDOWN_LINK_TARGETS_CHECKED={checked}")

if __name__ == "__main__":
    main()
