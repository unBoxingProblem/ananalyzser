from __future__ import annotations

import streamlit as st

from src.cleaning.cleaning_operations import CleaningOperation, DESCRIPTIONS, MissingMode
from src.cleaning.cleaning_service import CleaningService
from src.dataset.dataset_registry import DatasetRegistry

OPERATION_LABELS = {
    CleaningOperation.DROP_MISSING: "Remove rows containing missing values",
    CleaningOperation.DROP_DUPLICATES: "Remove completely duplicated rows",
    CleaningOperation.DROP_EMPTY_ROWS: "Remove rows containing only missing values",
}


def render_cleaning_page(registry: DatasetRegistry, dataset_id: str | None) -> None:
    st.divider()
    st.subheader("Data Cleaning")
    if dataset_id is None:
        st.info("Load a dataset to preview and apply cleaning operations.")
        return

    service = CleaningService(registry)
    events = st.session_state.setdefault("cleaning_events", {})
    event = events.pop(dataset_id, None)
    if event:
        st.success(event)

    current = registry.get_dataframe(dataset_id)
    operation = st.selectbox(
        "Select Cleaning Operation",
        list(CleaningOperation),
        format_func=lambda value: OPERATION_LABELS[value],
        key=f"clean_operation_{dataset_id}",
    )
    st.write(DESCRIPTIONS[operation])

    missing_mode = MissingMode.ANY
    if operation == CleaningOperation.DROP_MISSING:
        choice = st.radio(
            "Remove rows when",
            ["At least one cell is missing", "Every cell is missing"],
            horizontal=True,
            key=f"missing_mode_{dataset_id}",
        )
        missing_mode = MissingMode.ANY if choice == "At least one cell is missing" else MissingMode.ALL
    elif operation == CleaningOperation.DROP_EMPTY_ROWS:
        st.caption("This removes only fully empty rows. Rows with even one populated cell are kept.")

    preview = service.preview(dataset_id, operation, missing_mode=missing_mode)
    count_cols = st.columns(3)
    count_cols[0].metric("Current rows", f"{preview.rows_before:,}")
    count_cols[1].metric("Rows to remove", f"{preview.rows_to_remove:,}")
    count_cols[2].metric("Rows after cleaning", f"{preview.rows_after:,}")

    if preview.rows_to_remove:
        st.markdown("**Preview of rows that would be removed** (up to 10 rows)")
        st.dataframe(preview.removed_rows_preview, use_container_width=True, hide_index=True)
    else:
        st.info("No rows match this operation in the current working dataset. Nothing will be applied.")

    if registry.has_cleaning_changes(dataset_id):
        controls = st.columns(2)
        if controls[0].button(
            "Undo most recent cleaning",
            disabled=not registry.can_undo_cleaning(dataset_id),
            key=f"undo_clean_{dataset_id}",
        ):
            if service.undo(dataset_id):
                events[dataset_id] = "Undid the most recent cleaning operation."
                st.rerun()
        if controls[1].button("Reset Cleaning", key=f"reset_clean_{dataset_id}"):
            service.reset(dataset_id)
            events[dataset_id] = "Reset cleaning; the original uploaded dataset has been restored."
            st.rerun()
    else:
        st.button("Undo most recent cleaning", disabled=True, key=f"undo_clean_{dataset_id}")
        st.button("Reset Cleaning", disabled=True, key=f"reset_clean_{dataset_id}")

    if st.button(
        "Apply cleaning operation",
        type="primary",
        disabled=preview.rows_to_remove == 0,
        key=f"apply_clean_{dataset_id}",
    ):
        try:
            result = service.apply(dataset_id, operation, missing_mode=missing_mode)
            events[dataset_id] = (
                f"{OPERATION_LABELS[result.operation]} complete: removed {result.rows_removed:,} rows "
                f"({result.rows_before:,} → {result.rows_after:,})."
            )
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))
