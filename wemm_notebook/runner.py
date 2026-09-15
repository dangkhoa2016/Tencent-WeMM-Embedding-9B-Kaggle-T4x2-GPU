"""Atomic public-notebook orchestrator with fail-closed lifecycle ownership."""

from __future__ import annotations

from IPython.display import Markdown, display

from .bootstrap import bootstrap_runtime
from .closeout import run_closeout
from .image_showcase import run_image_showcase
from .text_showcase import run_text_showcase
from .visual_showcase import run_visual_showcase


_ACTIVE_DEMO = None


def _previous_notebook_demo():
    """Return a live demo left in the current IPython namespace, if any."""

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return None
        return shell.user_ns.get("demo")
    except Exception:
        return None


def _abort_live_demo(candidate, marker_prefix: str) -> None:
    if candidate is None or getattr(candidate, "closed", True):
        return
    print(f"{marker_prefix}=START", flush=True)
    candidate.abort()
    print(f"{marker_prefix}=PASS", flush=True)


def run_public_notebook():
    """Run bootstrap → setup → Step 6 → Step 7A → Step 7B → Step 8 atomically."""

    global _ACTIVE_DEMO

    print("ATOMIC_NOTEBOOK_RUNNER=START", flush=True)
    print("ATOMIC_NOTEBOOK_CODE_BOUNDARIES=ELIMINATED", flush=True)

    previous = _previous_notebook_demo()
    _abort_live_demo(previous, "ATOMIC_PREVIOUS_DEMO_CLEANUP")
    if _ACTIVE_DEMO is not previous:
        _abort_live_demo(_ACTIVE_DEMO, "ATOMIC_MODULE_DEMO_CLEANUP")
    _ACTIVE_DEMO = None

    demo = None
    try:
        bootstrap_runtime()

        display(
            Markdown(
                "## Steps 1/8–>5/8 — Chuẩn bị hệ thống / System setup\n\n"
                "**VI:** Thực hiện hardware preflight, source verification, Dataset "
                "verification, Qdrant reuse/restore và load model worker. Qdrant và GPU "
                "worker được giữ sống cho Steps 6, 7A và 7B.\n\n"
                "**EN:** Perform hardware preflight, source verification, Dataset "
                "verification, Qdrant reuse/restore, and model-worker loading. Qdrant and "
                "the GPU worker remain live for Steps 6, 7A, and 7B."
            )
        )

        from wemm_kaggle.public_demo import start_public_demo

        demo = start_public_demo()
        _ACTIVE_DEMO = demo

        display(
            Markdown(
                "## Step 6/8 — Trình diễn truy vấn văn bản / Text query showcase\n\n"
                "**VI:** Giữ nguyên 5 ví dụ EN↔VI và 20 retrieval paths đã khóa, nhưng "
                "render human-first với full query text, full candidate text và tách rõ "
                "TOP-1 winner khỏi nearest competitor.\n\n"
                "**EN:** Keep the same five frozen EN↔VI examples and 20 retrieval paths, "
                "while rendering full query/candidate text and visually separating the "
                "TOP-1 winner from the nearest competitor."
            )
        )
        text_results = run_text_showcase(demo)
        image_results = run_image_showcase(demo)
        visual_results = run_visual_showcase(demo)
        final_summary = run_closeout(demo, visual_results)

        _ACTIVE_DEMO = None
        return {
            "text_results": text_results,
            "image_results": image_results,
            "visual_results": visual_results,
            "final_summary": final_summary,
        }
    except BaseException:
        if demo is not None and not getattr(demo, "closed", True):
            try:
                demo.abort()
            finally:
                print("ATOMIC_FAILURE_DEMO_ABORT=PASS", flush=True)
        _ACTIVE_DEMO = None
        raise
