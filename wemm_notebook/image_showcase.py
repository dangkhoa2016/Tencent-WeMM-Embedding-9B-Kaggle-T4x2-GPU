"""Thin wrapper for the frozen Step 7A image-to-text showcase."""

from __future__ import annotations

from IPython.display import Markdown, display


def run_image_showcase(demo):
    """Run the unchanged semantic image→text showcase."""

    display(
        Markdown(
            "## Step 7A/8 — Truy xuất semantic đa phương thức / Semantic cross-modal retrieval\n\n"
            "**VI:** Đây là phần image→text gốc và vẫn được giữ nguyên. Nó chứng minh ảnh "
            "có thể truy xuất đúng entity text tiếng Anh và tiếng Việt ở cả 4096d và 1024d. "
            "Raw cosine của cross-modal space được hiển thị nguyên bản, không được rescale.\n\n"
            "**EN:** This is the original image→text showcase and remains unchanged. It "
            "proves that images retrieve the correct English and Vietnamese entity text at "
            "both 4096d and 1024d. Raw cross-modal cosine values are shown as-is and are "
            "never rescaled."
        )
    )
    print("STEP_7A_NOTEBOOK_CELL_ENTER=PASS", flush=True)
    return demo.run_image()
