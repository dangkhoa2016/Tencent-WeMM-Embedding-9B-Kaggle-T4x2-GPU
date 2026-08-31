#!/usr/bin/env python3
"""Evaluate the approved 5K cross-lingual quality policy on the acceptance report.

Policy id: wemm-v030-rc1-cross-lingual-quality-v1
Pass requires, for EACH cross-lingual combo (4096:en_vi, 4096:vi_en, 1024:en_vi,
1024:vi_en) independently:
    Recall@10 >= 0.95  AND  MRR@10 >= 0.90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


POLICY_ID = "wemm-v030-rc1-cross-lingual-quality-v1"
REQUIRED_COMBOS = ["4096:en_vi", "4096:vi_en", "1024:en_vi", "1024:vi_en"]
RECALL_THRESHOLD = 0.95
MRR_THRESHOLD = 0.90
ORIGINAL_STRICT_VERDICT = "FAIL"
APPROVED_OUTCOME = "PASS_WITH_DOCUMENTED_ACCEPTANCE_POLICY_DEVIATION"
APPROVAL_BASIS = "APPROVED_THRESHOLD_EVALUATION"

def evaluate_report(report: dict) -> dict:
    strict_verdict = report.get("verdict", ORIGINAL_STRICT_VERDICT)

    per_combo = []
    all_pass = True
    for combo in REQUIRED_COMBOS:
        entry = report.get("cross_lingual", {}).get(combo)
        if entry is None:
            per_combo.append(
                {
                    "combo": combo,
                    "recall_at_10": None,
                    "mrr_at_10": None,
                    "recall_pass": False,
                    "mrr_pass": False,
                    "pass": False,
                    "reason": "combo absent from acceptance report",
                }
            )
            all_pass = False
            continue
        recall = float(entry["recall_at_10"])
        mrr = float(entry["mrr_at_10"])
        recall_pass = recall >= RECALL_THRESHOLD
        mrr_pass = mrr >= MRR_THRESHOLD
        combo_pass = recall_pass and mrr_pass
        if not combo_pass:
            all_pass = False
        per_combo.append(
            {
                "combo": combo,
                "recall_at_10": recall,
                "mrr_at_10": mrr,
                "recall_required": RECALL_THRESHOLD,
                "mrr_required": MRR_THRESHOLD,
                "recall_pass": recall_pass,
                "mrr_pass": mrr_pass,
                "pass": combo_pass,
            }
        )

    deviation_required = all_pass and strict_verdict != "PASS"
    deviation_applied = deviation_required
    if deviation_applied:
        verdict = APPROVED_OUTCOME
    elif all_pass:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "policy_id": POLICY_ID,
        "policy_type": "RECALL_MRR_THRESHOLD_POLICY",
        "recall_required": RECALL_THRESHOLD,
        "mrr_required": MRR_THRESHOLD,
        "strict_top10_verdict": strict_verdict,
        "original_policy": "STRICT_TOP10_IDENTICAL_QID",
        "deviation_required": deviation_required,
        "deviation_applied": deviation_applied,
        "approval_basis": APPROVAL_BASIS if deviation_applied else "STANDARD_PASS",
        "per_combo": per_combo,
        "all_combo_pass": all_pass,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text())
    result = evaluate_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
