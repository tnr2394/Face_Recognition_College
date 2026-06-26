"""Shared recognition cooldown state between live_face_processor and recognition_watcher."""

import json
import os
import re
import time

COOLDOWN_STATE_FILE = "recognition_cooldown.json"
RECOGNITION_COOLDOWN_SEC = 300
COOLDOWN_IN_TRAINING = False


def _empty_state():
    return {"by_track": {}, "by_employee": {}}


def load_state():
    if not os.path.isfile(COOLDOWN_STATE_FILE):
        return _empty_state()
    try:
        with open(COOLDOWN_STATE_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_state()
        data.setdefault("by_track", {})
        data.setdefault("by_employee", {})
        data["by_track"] = {}  # ignored — ByteTrack reuses numeric IDs
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_state()


def save_state(state):
    with open(COOLDOWN_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def parse_track_id_from_folder(folder_name):
    """person_3 or person_3_b2_1719403200 -> 3"""
    match = re.match(r"^person_(\d+)", folder_name)
    return int(match.group(1)) if match else None


def is_in_cooldown(track_id=None, employee_id=None):
    """Check employee cooldown only (track IDs are reused by ByteTrack)."""
    if employee_id is None:
        return False
    now = time.time()
    state = load_state()
    expires = state.get("by_employee", {}).get(str(employee_id))
    return bool(expires and now < float(expires))


def set_cooldown(track_id=None, employee_id=None, duration=RECOGNITION_COOLDOWN_SEC):
    if employee_id is None:
        return
    expires = time.time() + duration
    state = load_state()
    state["by_employee"][str(employee_id)] = expires
    save_state(state)
