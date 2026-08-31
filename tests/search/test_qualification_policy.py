"""Focused tests for the corrected acceptance-policy deviation semantics."""

from __future__ import annotations

import json

import pytest

from scripts.evaluate_search_policy import APPROVED_OUTCOME, evaluate_report

RECALL = 0.99
MRR = 0.99


def _report(verdict: str = "FAIL") -> dict:
    combos = {}
    for combo in ("4096:en_vi", "4096:vi_en", "1024:en_vi", "1024:vi_en"):
        combos[combo] = {"recall_at_10": RECALL, "mrr_at_10": MRR}
    return {"verdict": verdict, "cross_lingual": combos}


def test_strict_pass_needs_no_deviation():
    result = evaluate_report(_report(verdict="PASS"))
    assert result["verdict"] == "PASS"
    assert result["deviation_required"] is False
    assert result["deviation_applied"] is False
    assert result["approval_basis"] == "STANDARD_PASS"
    assert result["all_combo_pass"] is True


def test_documented_deviation_applied_when_thresholds_pass():
    result = evaluate_report(_report(verdict="FAIL"))
    assert result["verdict"] == APPROVED_OUTCOME
    assert result["deviation_required"] is True
    assert result["deviation_applied"] is True
    assert result["strict_top10_verdict"] == "FAIL"
    assert result["all_combo_pass"] is True


def test_metrics_failure_never_applies_deviation():
    report = _report(verdict="FAIL")
    report["cross_lingual"]["1024:vi_en"] = {"recall_at_10": 0.4, "mrr_at_10": 0.3}
    result = evaluate_report(report)
    assert result["verdict"] == "FAIL"
    assert result["deviation_required"] is False
    assert result["deviation_applied"] is False
    assert result["all_combo_pass"] is False


def test_policy_cli_exits_nonzero_when_metrics_fail(tmp_path):
    from scripts.evaluate_search_policy import main

    report = _report(verdict="FAIL")
    report["cross_lingual"]["4096:en_vi"] = {"recall_at_10": 0.2, "mrr_at_10": 0.1}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    out_path = tmp_path / "out.json"
    assert main(["--report", str(report_path), "--out", str(out_path)]) == 1
    payload = json.loads(out_path.read_text())
    assert payload["verdict"] == "FAIL"
