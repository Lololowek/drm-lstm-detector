"""
dashboard.py
Streamlit live dashboard for the DRM replay-attack detector.
Polls the FastAPI server every 2 seconds.  No model inference here —
the server is the single source of truth.

Usage:
    streamlit run dashboard.py
"""

import os
import sys
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

_HERE = os.path.dirname(os.path.abspath(__file__))  # project root for Popen cwd


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    health:   dict
    sessions: list   # list of {session_id, n_events, classification}
    events:   list   # list of recent events from /events/recent


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_clock(epoch: float) -> str:
    """Unix epoch → local HH:MM:SS (for classified_at timestamps)."""
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def fmt_event_t(ts: float) -> str:
    """Relative session time → '+28.47s' string (not a clock time)."""
    return f"+{ts:.2f}s"


def is_classified(s: dict) -> bool:
    return isinstance(s.get("classification"), dict)


def is_attack(s: dict) -> bool:
    clf = s.get("classification")
    return isinstance(clf, dict) and clf.get("label") == 1


def compute_reuse_intervals(events: list) -> list:
    """Re-derive token_reuse_interval from event timestamps (same logic as inference.py)."""
    seen: dict[str, float] = {}
    result = []
    for ev in events:
        lid, ts = ev["licence_id"], float(ev["timestamp"])
        result.append(ts - seen[lid] if lid in seen else 0.0)
        seen[lid] = ts
    return result


def _apply_status_style(col: pd.Series) -> list:
    """Styler.apply column function — colours the 'status' cell."""
    colour_map = {
        "Normal":  "background-color: #d4edda",
        "Attack":  "background-color: #f8d7da",
        "Pending": "background-color: #e9ecef",
    }
    return [colour_map.get(v, "") for v in col]


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_state() -> AppState:
    """Single call that pulls all data from the server."""
    health   = requests.get(f"{API_URL}/health",                 timeout=2).json()
    sessions = requests.get(f"{API_URL}/sessions",               timeout=2).json()
    events   = requests.get(f"{API_URL}/events/recent?limit=60", timeout=2).json()
    return AppState(health=health, sessions=sessions, events=events)


# ── Section renders ───────────────────────────────────────────────────────────

def render_metrics(sessions: list):
    total      = len(sessions)
    classified = sum(1 for s in sessions if is_classified(s))
    attacks    = sum(1 for s in sessions if is_attack(s))
    normal     = classified - attacks

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions observed",        total)
    c2.metric("Classifications complete", classified)
    c3.metric("Attacks detected",         attacks,
              delta=attacks if attacks > 0 else None,
              delta_color="inverse")
    c4.metric("Normal sessions",          normal)


def render_controls(sessions: list):
    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        if st.button("Run demo traffic"):
            subprocess.Popen(
                [sys.executable, "traffic_client.py",
                "--host",            API_URL,
                "--normal_sessions", "8",
                "--attack_sessions", "3",
                "--rate",            "0.2",
                "--seed",            str(int(time.time()))],
                cwd=_HERE,
            )
            st.toast("Demo traffic launched")
        if st.button("Reset state"):
            requests.delete(f"{API_URL}/sessions", timeout=2)
            st.toast("Server state cleared")

    with col_info:
        threshold = next(
            (f"{s['classification']['threshold']:.2f}"
             for s in sessions if is_classified(s)),
            "—",
        )
        classified = sum(1 for s in sessions if is_classified(s))
        st.info(
            f"Model loaded · threshold {threshold} · "
            f"{classified}/{len(sessions)} sessions classified · auto-refresh 2s"
        )


def render_alerts(sessions: list):
    st.subheader("Alerts")
    attacks = sorted(
        (s for s in sessions if is_attack(s)),
        key=lambda s: s["classification"]["classified_at"],
        reverse=True,
    )
    if not attacks:
        st.info("No attacks detected yet — waiting for traffic.")
        return

    for sess in attacks:
        clf = sess["classification"]
        st.error(
            f"**{sess['session_id']}**  ·  "
            f"Attack probability: {clf['prob']:.3f}  ·  "
            f"Classified at: {fmt_clock(clf['classified_at'])}"
        )
        with st.expander("Event sequence"):
            detail = requests.get(
                f"{API_URL}/sessions/{sess['session_id']}", timeout=2
            ).json()
            evs = detail.get("events", [])
            if evs:
                counts: dict[str, int] = {}
                for ev in evs:
                    counts[ev["licence_id"]] = counts.get(ev["licence_id"], 0) + 1
                rows = [{"pos": ev["position"], "licence_id": ev["licence_id"],
                         "device_id": str(ev["device_id"]),
                         "time": fmt_event_t(ev["timestamp"])}
                        for ev in evs]
                df = pd.DataFrame(rows)
                def _highlight_reused(row, lc=counts):
                    bg = "#ffcccc" if lc.get(row["licence_id"], 0) > 1 else ""
                    return [f"background-color: {bg}" if bg else ""] * len(row)
                st.dataframe(
                    df.style.apply(_highlight_reused, axis=1),
                    width="stretch",
                )


def render_event_stream(events: list):
    st.subheader("Recent event stream")
    if not events:
        st.info("Waiting for events.")
        return

    def _status(ev: dict) -> str:
        clf = ev.get("classification_status")
        if not isinstance(clf, dict):
            return "Pending"
        return "Attack" if clf.get("label") == 1 else "Normal"

    rows = [
        {"time":       fmt_event_t(ev["timestamp"]),
         "session_id": ev["session_id"],
         "licence_id": ev["licence_id"],
         "device_id":  str(ev["device_id"]),
         "position":   ev["position"],
         "status":     _status(ev)}
        for ev in events[:30]
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.apply(_apply_status_style, subset=["status"]),
        width="stretch",
        height=420,
    )


def render_inspector(sessions: list):
    st.subheader("Session inspector")
    if not sessions:
        st.info("No sessions yet.")
        return

    all_sids = [s["session_id"] for s in reversed(sessions)]  # most recent first

    attack_list = [s for s in sessions if is_attack(s)]
    if attack_list:
        default = max(attack_list,
                      key=lambda s: s["classification"]["classified_at"])["session_id"]
    else:
        default = all_sids[0]

    selected = st.selectbox(
        "Select session", options=all_sids,
        index=all_sids.index(default) if default in all_sids else 0,
        key="inspector_select",
    )

    detail = requests.get(f"{API_URL}/sessions/{selected}", timeout=2).json()
    evs    = detail.get("events", [])
    clf    = detail.get("classification")

    m1, m2, m3, m4 = st.columns(4)
    if isinstance(clf, dict):
        m1.metric("Classification", "Attack" if clf["label"] == 1 else "Normal")
        m2.metric("Probability",    f"{clf['prob']:.3f}")
        m3.metric("Threshold",      f"{clf['threshold']:.3f}")
    else:
        m1.metric("Classification", "Pending")
        m2.metric("Probability",    "—")
        m3.metric("Threshold",      "—")
    m4.metric("Events buffered", len(evs))

    if not evs:
        return

    reuse = compute_reuse_intervals(evs)
    rows  = [{"pos": ev["position"], "licence_id": ev["licence_id"],
               "device_id": str(ev["device_id"]), "time": fmt_event_t(ev["timestamp"]),
               "reuse_s": round(r, 3)}
              for ev, r in zip(evs, reuse)]
    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.caption("Token reuse interval per event position  (red = non-zero reuse)")
    colors = ["#d62728" if r > 0 else "#aec7e8" for r in reuse]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.barh(range(len(reuse)), reuse, color=colors, height=0.6)
    ax.set_yticks(range(len(reuse)))
    ax.set_yticklabels([f"pos {i}" for i in range(len(reuse))])
    ax.set_xlabel("Reuse interval (seconds)")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _live_dashboard():
    try:
        state = fetch_state()
    except Exception:
        st.error(
            f"Cannot reach detector API at {API_URL}. "
            "Start it with:  uvicorn server:app --port 8000"
        )
        return

    render_metrics(state.sessions)
    st.divider()
    render_controls(state.sessions)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        render_alerts(state.sessions)
    with col_right:
        render_event_stream(state.events)

    st.divider()
    render_inspector(state.sessions)
    st.divider()


def main():
    st.set_page_config(
        page_title="DRM Replay Detector",
        page_icon="shield",
        layout="wide",
    )
    st.title("DRM Replay Attack Detector — Live Operations")
    st.caption("Astana IT University · Cybersecurity B058 · 2026")

    _live_dashboard()

    st.caption(
        "Backend: FastAPI · Model: LSTM (1×64, 4 features) · Dashboard polls every 2s"
    )


main()
