"""
Alert system — the system's "voice".
On escalation: terminal message, sound, and a timestamped log entry.
"""

import time

LOG_FILE = "crowd_alerts.log"

def _play_sound():
    """Short beep."""
    try:
        import winsound
        winsound.Beep(1000, 300)          # 1000 Hz for 300 ms
    except ImportError:
        print("\a", end="", flush=True)   

def fire_alert(risk_label, density, camera_name):
    """Record and announce an escalation event."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {camera_name} | {risk_label} | density={density:.3f}/m2"

    # Write to the log file
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    # Announce in the terminal
    print("ALERT:", line)

    # Sound only for the most serious level
    if risk_label == "CRITICAL":
        _play_sound()