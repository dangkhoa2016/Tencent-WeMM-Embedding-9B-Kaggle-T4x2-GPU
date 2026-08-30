"""Source-level tests for the canonical bilingual base-description contract.

Covers: pinned 29-codepoint whitespace set, forbidden 55-codepoint control set,
label-then-description order, null->absent, non-char->INVALID_FIELD_TYPE,
error precedence, verbatim emission, ". " join + single "." append, double
period preservation (no punctuation dedup), and EMPTY_LANGUAGE_TEXT.
"""
from __future__ import annotations

import pytest

from wemm_kaggle.search.base_description import (
    ALLOWED_WHITESPACE_CONTROL_SET,
    CONTRACT_ID,
    FORBIDDEN_CONTROL_SET,
    PRESENCE_WHITESPACE_SET,
    STATUS_ABSENT,
    STATUS_PRESENT,
    CODE_EMPTY_LANGUAGE_TEXT,
    CODE_FORBIDDEN_CONTROL_CHARACTER,
    CODE_INVALID_FIELD_TYPE,
    TextBuildError,
    build_language_text,
    component_status,
)

EXPECTED_CONTRACT_ID = "wemm-v030-canonical-bilingual-base-description-v1"

C0 = set(range(0x0000, 0x0020))
DEL = {0x007F}
C1 = set(range(0x0080, 0x00A0))

EXPECTED_PRESENCE_WHITESPACE_SET = frozenset(
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

EXPECTED_ALLOWED_CONTROLS = frozenset(
    [
        *range(0x0009, 0x000E),
        *range(0x001C, 0x0020),
        0x0085,
    ]
)

ALL_PINNED_WHITESPACE_CODEPOINTS = sorted(EXPECTED_PRESENCE_WHITESPACE_SET)
FORBIDDEN_CODEPOINTS = sorted((C0 | DEL | C1) - EXPECTED_ALLOWED_CONTROLS)


def test_contract_id_exact():
    assert CONTRACT_ID == EXPECTED_CONTRACT_ID


def test_pinned_whitespace_identity_set_exact():
    assert PRESENCE_WHITESPACE_SET == EXPECTED_PRESENCE_WHITESPACE_SET
    assert ALLOWED_WHITESPACE_CONTROL_SET == EXPECTED_ALLOWED_CONTROLS


def test_forbidden_control_set_exact():
    assert FORBIDDEN_CONTROL_SET == frozenset(FORBIDDEN_CODEPOINTS)
    assert len(FORBIDDEN_CONTROL_SET) == len(FORBIDDEN_CODEPOINTS)


@pytest.mark.parametrize("cp", ALL_PINNED_WHITESPACE_CODEPOINTS)
def test_pinned_whitespace_codepoint_alone_is_absent(cp):
    status, err = component_status(chr(cp))
    assert status == STATUS_ABSENT
    assert err is None


def test_pinned_whitespace_only_string_is_absent():
    chars = "".join(chr(cp) for cp in ALL_PINNED_WHITESPACE_CODEPOINTS)
    status, err = component_status(chars)
    assert status == STATUS_ABSENT
    assert err is None


def test_whitespace_only_emits_empty_language_text():
    with pytest.raises(TextBuildError) as exc:
        build_language_text(" \t\r\n\u00a0\u3000", None)
    assert exc.value.code == CODE_EMPTY_LANGUAGE_TEXT


@pytest.mark.parametrize("cp", FORBIDDEN_CODEPOINTS)
def test_forbidden_control_codepoint_rejected(cp):
    status, err = component_status(chr(cp))
    assert status == STATUS_PRESENT
    assert err == CODE_FORBIDDEN_CONTROL_CHARACTER


@pytest.mark.parametrize("cp", FORBIDDEN_CODEPOINTS)
def test_forbidden_control_in_emission_rejected(cp):
    with pytest.raises(TextBuildError) as exc:
        build_language_text("a" + chr(cp) + "b", None)
    assert exc.value.code == CODE_FORBIDDEN_CONTROL_CHARACTER


def test_order_label_then_description():
    assert build_language_text("l", "d") == "l. d."
    assert build_language_text(None, "d") == "d."
    assert build_language_text("l", None) == "l."


def test_null_to_absent_components():
    assert component_status(None) == (STATUS_ABSENT, None)


def test_nonstring_label_to_invalid_field_type():
    status, err = component_status(123)
    assert status == STATUS_PRESENT
    assert err == CODE_INVALID_FIELD_TYPE
    with pytest.raises(TextBuildError) as exc:
        build_language_text(123, "d")
    assert exc.value.code == CODE_INVALID_FIELD_TYPE


def test_nonstring_description_to_invalid_field_type():
    with pytest.raises(TextBuildError) as exc:
        build_language_text("l", 3.5)
    assert exc.value.code == CODE_INVALID_FIELD_TYPE


def test_precedence_nonstring_over_forbidden_control():
    with pytest.raises(TextBuildError) as exc:
        build_language_text(123, "a\x00b")
    assert exc.value.code == CODE_INVALID_FIELD_TYPE


def test_precedence_forbidden_control_over_presence():
    status, err = component_status("a\x00b")
    assert status == STATUS_PRESENT
    assert err == CODE_FORBIDDEN_CONTROL_CHARACTER


def test_verbatim_leading_trailing_whitespace_preserved():
    assert build_language_text("  ACME  ", None) == "  ACME  ."
    assert build_language_text("a", "  b  ") == "a.   b  ."


def test_verbatim_internal_whitespace_preserved():
    assert build_language_text("x\t y", None) == "x\t y."


def test_join_semantics_exact():
    assert build_language_text("a", "b") == "a. b."


def test_append_single_period_exact():
    assert build_language_text("a", None) == "a."
    assert build_language_text("a", None).count(".") == 1


def test_punctuation_double_period_not_deduplicated():
    assert build_language_text("Co., Ltd.", "company") == "Co., Ltd.. company."


def test_punctuation_trailing_period_preserved_before_join():
    assert build_language_text("End.", "Next") == "End.. Next."


def test_verbatim_leading_period_preserved():
    assert build_language_text(".lead", None) == ".lead."


def test_empty_language_text_when_both_absent():
    with pytest.raises(TextBuildError) as exc:
        build_language_text(None, None)
    assert exc.value.code == CODE_EMPTY_LANGUAGE_TEXT


def test_empty_language_text_error_is_value_error():
    with pytest.raises(ValueError):
        build_language_text(None, None)


def test_error_carries_code_attribute():
    try:
        build_language_text("\x07", None)
    except TextBuildError as exc:
        assert isinstance(exc, ValueError)
        assert exc.code == CODE_FORBIDDEN_CONTROL_CHARACTER


def test_present_normal_component():
    status, err = component_status("hello")
    assert status == STATUS_PRESENT
    assert err is None


def test_contract_reads_only_label_and_description_fields():
    import inspect
    import wemm_kaggle.search.base_description as mod
    sig = inspect.signature(mod.build_language_text)
    params = list(sig.parameters)
    assert params == ["label", "description"]
    func_src = inspect.getsource(mod.build_language_text) + inspect.getsource(mod.component_status)
    for needle in ("aliases", "document_", "qid", "unicodedata.normalize"):
        assert needle not in func_src


def test_no_strip_or_dedupe_in_emit_path():
    import inspect
    import wemm_kaggle.search.base_description as mod
    func_src = inspect.getsource(mod.build_language_text)
    assert ".strip(" not in func_src
    assert "dedup" not in func_src
    assert "unicodedata" not in func_src
