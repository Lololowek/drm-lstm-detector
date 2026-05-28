"""
show_logs.py
Prints readable DRM session event logs for presentation screenshots.
Shows 2 normal sessions and 2 replay attack sessions side by side.

Usage:
    python show_logs.py
"""

import numpy as np

RNG = np.random.default_rng(42)
SEQ_LEN = 10

def make_normal_session(session_id):
    intervals = RNG.lognormal(mean=3.33, sigma=0.6, size=SEQ_LEN)
    timestamps = np.cumsum(intervals)
    timestamps -= timestamps[0]
    device = f"DEV_{RNG.integers(1000, 9999)}"
    rows = []
    seen = {}
    for i in range(SEQ_LEN):
        lid = f"LIC-{session_id}-{i:03d}"
        t = float(timestamps[i])
        reuse = 0.0 if lid not in seen else t - seen[lid]
        seen[lid] = t
        rows.append({
            "t": t, "licence_id": lid,
            "device": device,
            "reuse_interval": reuse,
            "label": "NORMAL"
        })
    return rows

def make_replay_session(session_id):
    intervals = RNG.lognormal(mean=3.33, sigma=0.6, size=SEQ_LEN)
    timestamps = np.cumsum(intervals)
    timestamps -= timestamps[0]
    device = f"DEV_{RNG.integers(1000, 9999)}"
    licence_ids = [f"LIC-{session_id}-{i:03d}" for i in range(SEQ_LEN)]
    # inject replay at position 7: reuse licence from position 2
    licence_ids[7] = licence_ids[2]
    rows = []
    seen = {}
    for i in range(SEQ_LEN):
        lid = licence_ids[i]
        t = float(timestamps[i])
        reuse = 0.0 if lid not in seen else t - seen[lid]
        seen[lid] = t
        label = "*** REPLAY ATTACK ***" if i == 7 else "normal"
        rows.append({
            "t": t, "licence_id": lid,
            "device": device,
            "reuse_interval": reuse,
            "label": label
        })
    return rows

def print_session(title, rows):
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")
    print(f"  {'#':<3} {'Time (s)':<10} {'Licence ID':<18} {'Device':<12} {'Reuse interval':<16} {'Status'}")
    print(f"  {'─'*3} {'─'*9} {'─'*17} {'─'*11} {'─'*15} {'─'*22}")
    for i, r in enumerate(rows):
        reuse_str = f"{r['reuse_interval']:.1f}s" if r['reuse_interval'] > 0 else "—"
        flag = "  <-- FLAGGED" if "REPLAY" in r['label'] else ""
        print(f"  {i+1:<3} {r['t']:<10.1f} {r['licence_id']:<18} {r['device']:<12} {reuse_str:<16} {r['label']}{flag}")
    print()

def main():
    print("\n" + "="*72)
    print("  DRM LICENCE EVENT LOG VIEWER — Intelligent Testing System")
    print("  Astana IT University · Cybersecurity · 2025")
    print("="*72)

    print("\n\n  ► NORMAL SESSIONS (no attack detected)\n")
    print_session("Session N-001 — normal streaming session", make_normal_session("N001"))
    print_session("Session N-002 — normal streaming session", make_normal_session("N002"))

    print("\n\n  ► ATTACK SESSIONS (replay attack injected)\n")
    print_session("Session A-001 — replay attack at event #7", make_replay_session("A001"))
    print_session("Session A-002 — replay attack at event #7", make_replay_session("A002"))

    print("="*72)
    print("  Note: 'Reuse interval' = time since this licence ID was last seen.")
    print("  A non-zero reuse interval is the primary signal the LSTM detects.")
    print("="*72 + "\n")

if __name__ == "__main__":
    main()
