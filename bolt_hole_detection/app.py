from __future__ import annotations

from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from runtime_manager import DashboardRuntime, RuntimeSession
from state_machine import SystemState


st.set_page_config(page_title="Railway Bolt-Hole Detection", layout="wide")


@st.cache_resource
def get_runtime() -> DashboardRuntime:
    return DashboardRuntime()


runtime = get_runtime()
if "task_session" not in st.session_state:
    st.session_state.task_session = runtime.new_task_session(clear_database=True)

task_session: RuntimeSession = st.session_state.task_session
processor = task_session.processor

st.title("Railway Bolt-Hole Detection")

control_cols = st.columns(3)
with control_cols[0]:
    if st.button("Start Detection", width="stretch", disabled=processor.state_machine.is_running_state()):
        processor.start()
        st.toast("Detection started.")
        st.rerun()
with control_cols[1]:
    state = processor.state_machine.state
    if st.button("Stop Detection", width="stretch", disabled=state in {SystemState.IDLE, SystemState.STOPPED}):
        with st.spinner("Stopping worker threads..."):
            processor.stop()
        st.toast("Detection stopped.")
        st.rerun()
with control_cols[2]:
    if st.button("Generate Report", width="stretch"):
        report_path = processor.generate_report()
        st.success(f"Report generated: {report_path}")


@st.fragment(run_every=2.0)
def render_live_metrics() -> None:
    metrics = processor.snapshot_metrics()
    current_state = processor.state_machine.state

    if current_state == SystemState.WAITING_FOR_WINDOW:
        st.warning("Waiting for the SRT_BScan software window.")
    elif current_state == SystemState.CAPTURING and metrics.frames_processed == 0:
        st.info("Detection is armed. Play the SRT_BScan video; processing will begin automatically when motion appears.")
    elif current_state == SystemState.PROCESSING:
        st.success("Video motion detected. Detection is running.")

    status_cols = st.columns(6)
    status_cols[0].metric("Current State", current_state.value)
    status_cols[1].metric("Frames Processed", metrics.frames_processed)
    status_cols[2].metric("Visual Holes", metrics.holes_detected)
    status_cols[3].metric("Capture FPS", f"{metrics.capture_fps:.2f}")
    status_cols[4].metric("Processing FPS", f"{metrics.processing_fps:.2f}")
    status_cols[5].metric("OCR Events", metrics.ocr_events)

    queue_cols = st.columns(3)
    queue_cols[0].metric("Frame Queue", metrics.frame_queue_size)
    queue_cols[1].metric("Detection Queue", metrics.detection_queue_size)
    queue_cols[2].metric("Event Queue", metrics.event_queue_size)

    live = processor.snapshot_live_view()
    preview_col, table_col = st.columns([2, 1])
    with preview_col:
        st.subheader("Annotated B-scan")
        if live.annotated_bscan is None:
            st.info("Press Start Detection, then play the SRT_BScan video to see annotations here.")
        else:
            rgb_bscan = cv2.cvtColor(live.annotated_bscan, cv2.COLOR_BGR2RGB)
            st.image(rgb_bscan, caption=f"Latest B-scan frame: {live.frame_number}", width="stretch")
    with table_col:
        st.subheader("Bolt-hole distances")
        if live.records:
            st.dataframe(pd.DataFrame(live.records), width="stretch", hide_index=True)
        else:
            st.info("No bolt-hole distance records yet.")

    if metrics.last_error:
        st.error(metrics.last_error)

    st.caption(f"Dashboard refreshed: {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"Task prepared: {task_session.task_started_at.strftime('%H:%M:%S')}")


render_live_metrics()

st.caption("Control interface only. Live video rendering is intentionally disabled for low CPU usage.")
