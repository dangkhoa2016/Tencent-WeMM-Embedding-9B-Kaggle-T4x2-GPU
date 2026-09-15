"""Human-first rendering for the frozen bilingual text showcase."""

from __future__ import annotations

import contextlib
import html
import io
from pathlib import Path

from IPython.display import HTML, display


RAW_LOG_PATH = Path("/kaggle/working/wemm-step6-bilingual-raw.log")


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _nearest_competitor(path: dict, expected_qid: str) -> dict | None:
    for hit in path.get("top3", []):
        if str(hit.get("qid")) != str(expected_qid):
            return hit
    return None


def _path_map(item: dict) -> dict[tuple[str, int], dict]:
    mapped = {}
    for path in item["runtime_paths"]:
        direction = "EN→VI" if path["vector_name"] == "vi" else "VI→EN"
        mapped[(direction, int(path["dimension"]))] = path
    return mapped


def _candidate_text(hit: dict) -> str:
    """Return full candidate text with no truncation."""
    vi = str(hit.get("label_vi") or "").strip()
    en = str(hit.get("label_en") or "").strip()
    if vi and en and vi != en:
        return f"VI: {vi}<br>EN: {en}"
    return vi or en or str(hit.get("qid") or "")


def _winner_competitor_cell(path: dict, expected_qid: str) -> str:
    score = path.get("expected_score")
    rank = path.get("rank")
    competitor = _nearest_competitor(path, expected_qid)

    if score is None or rank is None:
        return "<b>Result unavailable</b>"

    winner = next(
        (
            hit
            for hit in path.get("top3", [])
            if str(hit.get("qid")) == str(expected_qid)
        ),
        None,
    )
    winner_text = _candidate_text(winner or {"qid": expected_qid})
    status = "✅" if rank == 1 else "⚠️"

    if competitor is None:
        competitor_html = "<i>No competitor returned.</i>"
        margin_html = "n/a"
    else:
        margin = float(score) - float(competitor["score"])
        competitor_html = (
            f"<b>#{int(competitor['rank'])} {_esc(competitor['qid'])}</b><br>"
            f"{_candidate_text(competitor)}<br>"
            f"raw cosine <b>{float(competitor['score']):.6f}</b>"
        )
        margin_html = f"+{margin:.6f}"

    return f"""
    <div style="display:grid;grid-template-columns:1fr;gap:8px">
      <div style="border:1px solid #b9d8bd;border-radius:9px;padding:9px 10px">
        <div style="font-size:12px;font-weight:750;opacity:.75">TOP-1 WINNER / KẾT QUẢ #1</div>
        <div style="margin-top:4px"><b>{status} #{int(rank)} {_esc(expected_qid)}</b></div>
        <div style="margin-top:3px">{winner_text}</div>
        <div style="margin-top:4px">raw cosine <b>{float(score):.6f}</b></div>
      </div>
      <div style="border:1px solid #dedede;border-radius:9px;padding:9px 10px">
        <div style="font-size:12px;font-weight:750;opacity:.75">NEAREST COMPETITOR / ĐỐI THỦ GẦN NHẤT</div>
        <div style="margin-top:4px">{competitor_html}</div>
        <div style="margin-top:5px">winner margin <b>{margin_html}</b></div>
      </div>
    </div>
    """


def _technical_rows(item: dict, paths: dict) -> str:
    rows = []
    for direction in ("EN→VI", "VI→EN"):
        for dimension in (4096, 1024):
            path = paths[(direction, dimension)]
            for hit in path.get("top3", []):
                marker = "✅" if str(hit.get("qid")) == str(item["qid"]) else ""
                rows.append(
                    "<tr>"
                    f"<td>{_esc(direction)} · {dimension}d</td>"
                    f"<td>#{int(hit['rank'])}</td>"
                    f"<td>{marker} <b>{_esc(hit['qid'])}</b><br>{_candidate_text(hit)}</td>"
                    f"<td style='text-align:right;font-family:monospace'>{float(hit['score']):.6f}</td>"
                    "</tr>"
                )
    return "".join(rows)


def run_text_showcase(demo):
    """Run frozen text retrieval unchanged and render full human-readable evidence."""

    raw_stdout = io.StringIO()
    with contextlib.redirect_stdout(raw_stdout):
        text_results = demo.run_text()

    raw_log = raw_stdout.getvalue()
    RAW_LOG_PATH.write_text(raw_log, encoding="utf-8")

    display(
        HTML(
            "<div style='padding:14px 16px;border:1px solid #c9c9c9;border-radius:12px;margin:8px 0 18px 0'>"
            "<div style='font-size:22px;font-weight:700'>Bilingual semantic retrieval / Truy xuất ngữ nghĩa song ngữ</div>"
            "<div style='margin-top:6px;line-height:1.55'>"
            "Each example asks whether an English description retrieves the matching Vietnamese representation, "
            "and whether the Vietnamese description independently retrieves the matching English representation. "
            "The same test is repeated at 4096d and 1024d."
            "</div></div>"
        )
    )

    top1_counts = {
        ("EN→VI", 4096): 0,
        ("EN→VI", 1024): 0,
        ("VI→EN", 4096): 0,
        ("VI→EN", 1024): 0,
    }

    for index, item in enumerate(text_results, 1):
        paths = _path_map(item)
        for key, path in paths.items():
            if path.get("rank") == 1:
                top1_counts[key] += 1

        technical_rows = _technical_rows(item, paths)
        card = f"""
        <div style="border:1px solid #c9c9c9;border-radius:14px;padding:16px 18px;margin:14px 0 22px 0">
          <div style="font-size:13px;font-weight:700;letter-spacing:.04em;opacity:.72">
            EXAMPLE {index}/5 · CROSS-LANGUAGE SEMANTIC RETRIEVAL
          </div>
          <div style="font-size:24px;font-weight:750;margin-top:3px">
            {_esc(item['name'])}
            <span style="font-size:15px;font-weight:500;opacity:.7">({_esc(item['qid'])})</span>
          </div>
          <div style="font-size:13px;opacity:.72;margin:2px 0 14px 0">{_esc(item['category'])}</div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div style="border:1px solid #dddddd;border-radius:10px;padding:12px">
              <div style="font-weight:700;margin-bottom:6px">🇬🇧 English query — full embedded text</div>
              <div style="line-height:1.48">{_esc(item['text_en'])}</div>
              <div style="margin-top:10px;font-size:12px;opacity:.72">
                English meaning → embedding → search <b>Vietnamese vectors</b>
              </div>
            </div>
            <div style="border:1px solid #dddddd;border-radius:10px;padding:12px">
              <div style="font-weight:700;margin-bottom:6px">🇻🇳 Vietnamese query — full embedded text</div>
              <div style="line-height:1.48">{_esc(item['text_vi'])}</div>
              <div style="margin-top:10px;font-size:12px;opacity:.72">
                Vietnamese meaning → embedding → search <b>English vectors</b>
              </div>
            </div>
          </div>

          <div style="font-size:16px;font-weight:700;margin:16px 0 7px 0">
            Cross-language result / Kết quả truy xuất chéo ngôn ngữ
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:14px">
            <thead>
              <tr>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #cccccc">Direction</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #cccccc">4096d</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid #cccccc">1024d</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="padding:9px 8px;vertical-align:top"><b>🇬🇧 EN → 🇻🇳 VI</b></td>
                <td style="padding:9px 8px;vertical-align:top">{_winner_competitor_cell(paths[('EN→VI', 4096)], item['qid'])}</td>
                <td style="padding:9px 8px;vertical-align:top">{_winner_competitor_cell(paths[('EN→VI', 1024)], item['qid'])}</td>
              </tr>
              <tr>
                <td style="padding:9px 8px;vertical-align:top"><b>🇻🇳 VI → 🇬🇧 EN</b></td>
                <td style="padding:9px 8px;vertical-align:top">{_winner_competitor_cell(paths[('VI→EN', 4096)], item['qid'])}</td>
                <td style="padding:9px 8px;vertical-align:top">{_winner_competitor_cell(paths[('VI→EN', 1024)], item['qid'])}</td>
              </tr>
            </tbody>
          </table>

          <div style="margin-top:14px;padding:11px 12px;border:1px solid #dddddd;border-radius:10px;line-height:1.5">
            <b>What happened? / Điều gì vừa xảy ra?</b><br>
            The English description retrieved <b>{_esc(item['qid'])} — {_esc(item['name'])}</b> at rank #1
            from Vietnamese vectors, and the Vietnamese description independently retrieved the same entity at rank #1
            from English vectors. The result held at both <b>4096d</b> and <b>1024d</b>.
          </div>

          <div style="margin-top:12px;font-size:18px;font-weight:750">
            ✅ 4 / 4 cross-language paths TOP-1 — PASS
          </div>

          <details style="margin-top:14px">
            <summary style="cursor:pointer;font-weight:700">
              Technical details / Chi tiết kỹ thuật — full Top-3 text + raw cosine
            </summary>
            <div style="margin-top:8px">
              <table style="width:100%;border-collapse:collapse;font-size:12px">
                <thead>
                  <tr>
                    <th style="text-align:left;padding:6px;border-bottom:1px solid #cccccc">Path</th>
                    <th style="text-align:left;padding:6px;border-bottom:1px solid #cccccc">Rank</th>
                    <th style="text-align:left;padding:6px;border-bottom:1px solid #cccccc">Full candidate text</th>
                    <th style="text-align:right;padding:6px;border-bottom:1px solid #cccccc">Raw cosine</th>
                  </tr>
                </thead>
                <tbody>{technical_rows}</tbody>
              </table>
              <div style="font-size:12px;opacity:.72;margin-top:7px">
                Text is intentionally not truncated. Raw cosine is a similarity score, not a confidence percentage.
              </div>
            </div>
          </details>
        </div>
        """
        display(HTML(card))

    total_top1 = sum(top1_counts.values())
    if len(text_results) != 5 or total_top1 != 20:
        raise RuntimeError(
            "Human-first Step 6 summary contract failed: "
            f"entities={len(text_results)} top1_paths={total_top1}"
        )

    summary_html = f"""
    <div style="border:2px solid #b9b9b9;border-radius:14px;padding:16px 18px;margin:18px 0">
      <div style="font-size:23px;font-weight:750">Bilingual retrieval summary / Tổng kết truy xuất song ngữ</div>
      <div style="margin:7px 0 13px 0">
        <b>5 real entities</b> · <b>10 natural-language embeddings</b> · <b>20 cross-language retrieval paths</b>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:15px">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px;border-bottom:1px solid #cccccc">Direction</th>
            <th style="text-align:center;padding:8px;border-bottom:1px solid #cccccc">4096d</th>
            <th style="text-align:center;padding:8px;border-bottom:1px solid #cccccc">1024d</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:9px 8px"><b>🇬🇧 EN → 🇻🇳 VI</b></td>
            <td style="text-align:center;padding:9px 8px"><b>{top1_counts[('EN→VI', 4096)]}/5 ✅</b></td>
            <td style="text-align:center;padding:9px 8px"><b>{top1_counts[('EN→VI', 1024)]}/5 ✅</b></td>
          </tr>
          <tr>
            <td style="padding:9px 8px"><b>🇻🇳 VI → 🇬🇧 EN</b></td>
            <td style="text-align:center;padding:9px 8px"><b>{top1_counts[('VI→EN', 4096)]}/5 ✅</b></td>
            <td style="text-align:center;padding:9px 8px"><b>{top1_counts[('VI→EN', 1024)]}/5 ✅</b></td>
          </tr>
        </tbody>
      </table>
      <div style="font-size:21px;font-weight:800;margin-top:14px">✅ TOTAL: 20 / 20 TOP-1</div>
    </div>
    """
    display(HTML(summary_html))

    marker_prefixes = (
        "FROZEN_BILINGUAL_SHOWCASE=",
        "TEXT_SHOWCASE_ENTITIES=",
        "TEXT_SHOWCASE_EMBEDDING_REQUESTS=",
        "TEXT_SHOWCASE_QUERY_EXECUTIONS=",
        "TEXT_SHOWCASE_LANGUAGE_GATE_AUTHORITY=",
        "TEXT_SHOWCASE_ALL_TOP1=",
        "TEXT_SHOWCASE_STRICT_ALL_TOP3=",
        "TEXT_SHOWCASE_THRESHOLD_RELAXED=",
    )
    for line in raw_log.splitlines():
        if line.startswith(marker_prefixes):
            print(line, flush=True)

    print(f"STEP_6_RAW_AUDIT_LOG={RAW_LOG_PATH}", flush=True)
    print("STEP_6_PRESENTATION_LAYER=HUMAN_FIRST_FULL_TEXT", flush=True)
    print("STEP_6_NOTEBOOK_CELL_RETURN=PASS", flush=True)
    return text_results
