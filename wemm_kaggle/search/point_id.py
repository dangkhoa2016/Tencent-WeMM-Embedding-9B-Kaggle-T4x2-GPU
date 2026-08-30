from __future__ import annotations

import re

_QID_RE = re.compile(r"^Q([1-9][0-9]*)$")
_QDRANT_UINT64_MAX = (1 << 64) - 1


def qid_to_point_id(qid: str) -> int:
    if not isinstance(qid, str):
        raise ValueError(f"invalid QID (not a string): {qid!r}")
    match = _QID_RE.match(qid)
    if not match:
        raise ValueError(f"invalid Wikidata QID: {qid!r}")
    point_id = int(match.group(1))
    if point_id > _QDRANT_UINT64_MAX:
        raise ValueError(f"QID numeric suffix exceeds Qdrant uint64 range: {qid!r}")
    return point_id
