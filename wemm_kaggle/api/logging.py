from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def emit_event(event: str, **fields: Any) -> None:
    """Emit one JSONL service-log record on stdout.

    Never pass raw request text, image bytes, embedding vectors, or auth tokens
    into ``fields``; only safe numeric/status values belong there.
    """
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)