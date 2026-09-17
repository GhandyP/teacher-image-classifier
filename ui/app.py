"""Streamlit shell for the local teacher review workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import ReviewStatus
from src.review_ui import (
    apply_item_edit,
    documents_with_images,
    filter_documents,
    load_workspace,
    save_workspace,
    summary_lines,
)


def _rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:
        st.experimental_rerun()


def main() -> None:
    """Run the local-only Streamlit review application."""

    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required for the teacher review UI; install it with 'pip install -e .[ui]'."
        ) from exc

    st.set_page_config(page_title="Teacher review")
    st.title("Teacher review")
    with st.sidebar:
        workspace_text = st.text_input("Session JSON path", value="reviews/session.json")
        status_filter = st.selectbox(
            "Document status",
            ["all", *(status.value for status in ReviewStatus)],
        )
        only_pending = st.checkbox("Only pending documents")

    workspace_path = Path(workspace_text).expanduser()
    try:
        documents = load_workspace(workspace_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    for line in summary_lines(documents):
        st.write(line)

    visible_documents = filter_documents(documents, status_filter, only_pending)
    for document, image_path in documents_with_images(visible_documents):
        student = document.student_reference or "(no student reference)"
        st.header(f"{document.evaluation_id} · {student} · {document.review_status.value}")
        if image_path is not None:
            st.image(str(image_path), use_container_width=True)
        else:
            st.info("Source image is unavailable locally.")

        edits = []
        for item in document.items:
            correction = st.text_area(
                f"{item.item_id} · {item.review_status.value}",
                value=item.teacher_correction or item.model_transcription,
                key=f"correction:{document.evaluation_id}:{item.item_id}",
            )
            status = st.selectbox(
                f"Status for {item.item_id}",
                [review_status.value for review_status in ReviewStatus],
                index=[review_status.value for review_status in ReviewStatus].index(item.review_status.value),
                key=f"status:{document.evaluation_id}:{item.item_id}",
            )
            edits.append((item.item_id, correction, status))

        if st.button("Save changes", key=f"save:{document.evaluation_id}"):
            try:
                updated = documents
                current = document
                for item_id, correction, status in edits:
                    current = apply_item_edit(
                        current,
                        item_id,
                        teacher_correction=correction,
                        review_status=status,
                    )
                updated = [current if candidate is document else candidate for candidate in documents]
                save_workspace(updated, workspace_path)
                st.toast("Changes saved")
                _rerun(st)
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
