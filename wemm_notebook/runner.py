"""Sectioned public-notebook orchestrator with fail-closed lifecycle ownership."""

from __future__ import annotations

from .bootstrap import bootstrap_runtime
from .closeout import run_closeout
from .image_showcase import run_image_showcase
from .text_showcase import run_text_showcase
from .visual_showcase import run_visual_showcase


_ACTIVE_DEMO = None
_PHASE = "idle"
_VISUAL_RESULTS = None


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


def _clear_state() -> None:
    global _ACTIVE_DEMO, _PHASE, _VISUAL_RESULTS
    _ACTIVE_DEMO = None
    _PHASE = "idle"
    _VISUAL_RESULTS = None


def _fail_closed(demo, marker: str) -> None:
    try:
        if demo is not None and not getattr(demo, "closed", True):
            demo.abort()
            print(f"{marker}=PASS", flush=True)
    finally:
        _clear_state()


def _require_phase(demo, expected: str) -> None:
    if demo is None:
        raise RuntimeError("Public demo session is not initialized")
    if demo is not _ACTIVE_DEMO:
        raise RuntimeError("Notebook demo object does not match the active public session")
    if getattr(demo, "closed", True):
        raise RuntimeError("Public demo session is already closed")
    if _PHASE != expected:
        raise RuntimeError(
            f"Notebook phase order violation: observed={_PHASE} expected={expected}"
        )


def start_public_session():
    """Bootstrap frozen authority and complete Steps 1–5 setup."""

    global _ACTIVE_DEMO, _PHASE, _VISUAL_RESULTS

    previous = _previous_notebook_demo()
    _abort_live_demo(previous, "SECTIONED_PREVIOUS_DEMO_CLEANUP")
    if _ACTIVE_DEMO is not previous:
        _abort_live_demo(_ACTIVE_DEMO, "SECTIONED_MODULE_DEMO_CLEANUP")
    _clear_state()

    demo = None
    try:
        bootstrap_runtime()
        from wemm_kaggle.public_demo import start_public_demo

        demo = start_public_demo()
        _ACTIVE_DEMO = demo
        _PHASE = "setup"
        _VISUAL_RESULTS = None
        print("SECTIONED_NOTEBOOK_SESSION=START", flush=True)
        print("NOTEBOOK_PHASE_SETUP=PASS", flush=True)
        return demo
    except BaseException:
        _fail_closed(demo, "SECTIONED_SETUP_FAILURE_ABORT")
        raise


def run_step6(demo):
    """Run Step 6 bilingual text retrieval in its own notebook code cell."""

    global _PHASE
    _require_phase(demo, "setup")
    try:
        result = run_text_showcase(demo)
        _PHASE = "step6"
        print("NOTEBOOK_PHASE_STEP6=PASS", flush=True)
        return result
    except BaseException:
        _fail_closed(demo, "SECTIONED_STEP6_FAILURE_ABORT")
        raise


def run_step7a(demo):
    """Run Step 7A semantic image→text retrieval in its own code cell."""

    global _PHASE
    _require_phase(demo, "step6")
    try:
        result = run_image_showcase(demo)
        _PHASE = "step7a"
        print("NOTEBOOK_PHASE_STEP7A=PASS", flush=True)
        return result
    except BaseException:
        _fail_closed(demo, "SECTIONED_STEP7A_FAILURE_ABORT")
        raise


def run_step7b(demo):
    """Run Step 7B visual robustness retrieval in its own code cell."""

    global _PHASE, _VISUAL_RESULTS
    _require_phase(demo, "step7a")
    try:
        result = run_visual_showcase(demo)
        _VISUAL_RESULTS = result
        _PHASE = "step7b"
        print("NOTEBOOK_PHASE_STEP7B=PASS", flush=True)
        return result
    except BaseException:
        _fail_closed(demo, "SECTIONED_STEP7B_FAILURE_ABORT")
        raise


def run_step8(demo, visual_results=None):
    """Run Step 8 closeout in its own code cell."""

    global _PHASE
    _require_phase(demo, "step7b")
    results = visual_results if visual_results is not None else _VISUAL_RESULTS
    if results is None:
        raise RuntimeError("Step 7B results are unavailable for closeout")

    try:
        summary = run_closeout(demo, results)
        _PHASE = "closed"
        print("NOTEBOOK_PHASE_STEP8=PASS", flush=True)
        print("SECTIONED_NOTEBOOK_RUNNER=PASS", flush=True)
        print("NOTEBOOK_EXECUTABLE_CELLS=5", flush=True)
        print("STEP_7A_7B_SEPARATE_CELLS=PASS", flush=True)
        _clear_state()
        return summary
    except BaseException:
        _fail_closed(demo, "SECTIONED_STEP8_FAILURE_ABORT")
        raise


def abort_public_session(demo=None) -> None:
    """Explicitly abort a live sectioned session."""

    target = demo if demo is not None else _ACTIVE_DEMO
    _abort_live_demo(target, "SECTIONED_EXPLICIT_ABORT")
    _clear_state()


def run_public_notebook():
    """Compatibility helper that executes the same five phases sequentially."""

    print("LEGACY_SEQUENTIAL_COMPATIBILITY_RUNNER=START", flush=True)
    demo = start_public_session()
    text_results = run_step6(demo)
    image_results = run_step7a(demo)
    visual_results = run_step7b(demo)
    final_summary = run_step8(demo, visual_results)
    print("LEGACY_SEQUENTIAL_COMPATIBILITY_RUNNER=PASS", flush=True)
    return {
        "text_results": text_results,
        "image_results": image_results,
        "visual_results": visual_results,
        "final_summary": final_summary,
    }
