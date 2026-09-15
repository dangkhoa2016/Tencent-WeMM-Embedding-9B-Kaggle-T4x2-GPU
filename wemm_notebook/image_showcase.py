"""Thin wrapper for the frozen Step 7A image-to-text showcase."""

from __future__ import annotations

def run_image_showcase(demo):
    """Run the unchanged semantic image→text showcase."""

    print("STEP_7A_NOTEBOOK_CELL_ENTER=PASS", flush=True)
    return demo.run_image()
