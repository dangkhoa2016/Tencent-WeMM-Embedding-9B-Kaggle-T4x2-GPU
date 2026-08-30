"""Exact canonical bilingual base-description text contract.

Contract ID: ``wemm-v030-canonical-bilingual-base-description-v1``.

Builds a language text from ordered structured components ``label`` then
``description``. Component semantics (per component):

* ``None`` -> ABSENT
* non-``str`` -> PRESENT with error code ``INVALID_FIELD_TYPE``
* ``str`` containing a codepoint in ``FORBIDDEN_CONTROL_SET`` -> PRESENT with
  error code ``FORBIDDEN_CONTROL_CHARACTER``
* ``str`` containing at least one codepoint outside
  ``PRESENCE_WHITESPACE_SET`` -> PRESENT
* otherwise (only pinned whitespace codepoints) -> ABSENT

The assembled text is ``". ".join(present_components) + "."``. Present
components are emitted verbatim (no Unicode normalization, no strip/trim, no
punctuation deduplication). When no component is present a
``TextBuildError`` with code ``EMPTY_LANGUAGE_TEXT`` is raised.

Only the structured fields ``label_en``, ``description_en``, ``label_vi``,
``description_vi`` may feed this builder. No alias, translation, QID, or
synthetic fallbacks are permitted.
"""
from __future__ import annotations

from typing import Optional, Tuple

CONTRACT_ID = "wemm-v030-canonical-bilingual-base-description-v1"

C0 = set(range(0x0000, 0x0020))
DEL = {0x007F}
C1 = set(range(0x0080, 0x00A0))

# Pinned 29-codepoint presence whitespace set (equals Python str.strip() set).
PRESENCE_WHITESPACE_SET = frozenset(
    [
        *range(0x0009, 0x000E),
        *range(0x001C, 0x0020),
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    ]
)

# Control codepoints that are allowed extraction whitespace.
ALLOWED_WHITESPACE_CONTROL_SET = frozenset(
    [
        *range(0x0009, 0x000E),
        *range(0x001C, 0x0020),
        0x0085,
    ]
)

# Forbidden control set: C0 | DEL | C1 minus allowed controls (55 codepoints).
FORBIDDEN_CONTROL_SET = frozenset((C0 | DEL | C1) - ALLOWED_WHITESPACE_CONTROL_SET)

STATUS_ABSENT = "ABSENT"
STATUS_PRESENT = "PRESENT"
CODE_INVALID_FIELD_TYPE = "INVALID_FIELD_TYPE"
CODE_FORBIDDEN_CONTROL_CHARACTER = "FORBIDDEN_CONTROL_CHARACTER"
CODE_EMPTY_LANGUAGE_TEXT = "EMPTY_LANGUAGE_TEXT"


class TextBuildError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code + (f": {detail}" if detail else ""))
        self.code = code


def component_status(value) -> Tuple[str, Optional[str]]:
    if value is None:
        return STATUS_ABSENT, None
    if not isinstance(value, str):
        return STATUS_PRESENT, CODE_INVALID_FIELD_TYPE
    if any(ord(ch) in FORBIDDEN_CONTROL_SET for ch in value):
        return STATUS_PRESENT, CODE_FORBIDDEN_CONTROL_CHARACTER
    if any(ord(ch) not in PRESENCE_WHITESPACE_SET for ch in value):
        return STATUS_PRESENT, None
    return STATUS_ABSENT, None


def build_language_text(label, description) -> str:
    parts = []
    for value in (label, description):
        status, err = component_status(value)
        if err is not None:
            raise TextBuildError(err)
        if status == STATUS_PRESENT:
            parts.append(value)
    if not parts:
        raise TextBuildError(CODE_EMPTY_LANGUAGE_TEXT)
    return ". ".join(parts) + "."
