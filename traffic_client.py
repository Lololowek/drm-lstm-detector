"""
traffic_client.py
Realistic DRM traffic generator — drives the licence server for live demos.
Imports session generators from data_generator; owns no feature-engineering logic.

Usage:
    python traffic_client.py
    python traffic_client.py --normal_sessions 4 --attack_sessions 2 --rate 0.1
"""

import argparse
import sys
import time

import numpy as np
import requests

from data_generator import make_normal_session, make_replay_session

# Ensure Unicode checkmarks render on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post_event(host: str, session_id: str, event: dict) -> dict:
    """POST a single licence event to the server and return the JSON response."""
    payload = {
        "session_id": session_id,
        "licence_id": event["licence_id"],
        "device_id":  str(event["device"]),   # int bucket -> string device_id
        "timestamp":  event["timestamp"],
    }
    resp = requests.post(f"{host}/licence", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def run_session(
    host: str, session_id: str, events: list[dict], rate: float, truth_label: int
) -> int:
    """Send all 10 events, print a summary line, return 1 if correctly classified."""
    last = None
    for ev in events:
        last = post_event(host, session_id, ev)
        time.sleep(rate)

    clf       = (last or {}).get("classification") or {}
    predicted = clf.get("label", -1)
    prob      = clf.get("prob", 0.0)
    truth_str = "ATTACK" if truth_label == 1 else "NORMAL"
    pred_str  = "ATTACK" if predicted == 1  else "NORMAL"
    marker    = "✓" if predicted == truth_label else "✗"
    print(f"{session_id:<16} truth={truth_str:<7} predicted={pred_str:<7} prob={prob:.3f}  {marker}")
    return int(predicted == truth_label)


def main():
    parser = argparse.ArgumentParser(description="DRM traffic demo client")
    parser.add_argument("--host",              default="http://localhost:8000")
    parser.add_argument("--normal_sessions",   type=int,   default=8)
    parser.add_argument("--attack_sessions",   type=int,   default=2)
    parser.add_argument("--rate",              type=float, default=0.3,
                        help="Seconds between events within a session")
    parser.add_argument("--inter_session_gap", type=float, default=1.0,
                        help="Seconds between consecutive sessions")
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    sessions = []
    for i in range(args.normal_sessions):
        sid = f"demo_n_{i}"
        sessions.append((sid, make_normal_session(rng, sid), 0))
    for i in range(args.attack_sessions):
        sid = f"demo_a_{i}"
        sessions.append((sid, make_replay_session(rng, sid), 1))

    n_correct = 0
    for idx, (sid, events, label) in enumerate(sessions):
        n_correct += run_session(args.host, sid, events, args.rate, label)
        if idx < len(sessions) - 1:
            time.sleep(args.inter_session_gap)

    print(f"\n{n_correct}/{len(sessions)} sessions correctly classified.")


if __name__ == "__main__":
    main()
