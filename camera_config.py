"""Load cameras.json — single source of RTSP + rotation for capture and Settings UI."""

import json
import os

CAMERAS_JSON = os.environ.get(
    "CAMERAS_JSON",
    "/var/www/html/Face_Recognition_College/cameras.json",
)

VALID_ROTATIONS = (0, 90, 180, 270)


def load_cameras(path=None):
    path = path or CAMERAS_JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("cameras.json must be an object")
    cameras = data.get("cameras") or []
    if not isinstance(cameras, list):
        raise ValueError("cameras.json cameras must be a list")
    return data


def get_camera(camera_id, path=None):
    data = load_cameras(path)
    for cam in data.get("cameras") or []:
        if str(cam.get("id", "")).strip() == str(camera_id).strip():
            return cam
    return None


def capture_source_for(camera_id, path=None):
    """Return CAPTURE_SOURCE style string: rtsp[:rotation]."""
    cam = get_camera(camera_id, path)
    if not cam:
        raise KeyError(f"Unknown camera id: {camera_id}")
    rtsp = str(cam.get("rtsp") or "").strip()
    if not rtsp:
        raise ValueError(f"Camera {camera_id} has empty rtsp")
    try:
        rotation = int(cam.get("rotation", 0)) % 360
    except (TypeError, ValueError):
        rotation = 0
    if rotation not in VALID_ROTATIONS:
        rotation = 0
    if rotation:
        return f"{rtsp}:{rotation}"
    return rtsp


def save_cameras(data, path=None):
    path = path or CAMERAS_JSON
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _parse_hhmm(value):
    """Return normalized HH:MM or raise ValueError."""
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time: {value!r}")
    return f"{hour:02d}:{minute:02d}"


def update_office(office_patch, path=None):
    """
    Merge shift_start / shift_end (and other office keys) into cameras.json office.
    Only known time fields are validated; unknown keys are ignored.
    """
    data = load_cameras(path)
    office = data.get("office")
    if not isinstance(office, dict):
        office = {}
        data["office"] = office
    patch = office_patch or {}
    if "shift_start" in patch and patch["shift_start"] is not None:
        office["shift_start"] = _parse_hhmm(patch["shift_start"])
    if "shift_end" in patch and patch["shift_end"] is not None:
        office["shift_end"] = _parse_hhmm(patch["shift_end"])
    save_cameras(data, path)
    return data


def update_camera_streams(updates, path=None, office=None):
    """
    updates: list of {id, rtsp?, rotation?}
    Only rtsp/rotation are changed; other fields preserved.
    Optional office: {shift_start?, shift_end?}
    """
    data = load_cameras(path)
    by_id = {str(c.get("id")): c for c in (data.get("cameras") or [])}
    for item in updates or []:
        cid = str(item.get("id", "")).strip()
        if not cid or cid not in by_id:
            continue
        cam = by_id[cid]
        if "rtsp" in item and item["rtsp"] is not None:
            cam["rtsp"] = str(item["rtsp"]).strip()
        if "rotation" in item and item["rotation"] is not None:
            try:
                rot = int(item["rotation"]) % 360
            except (TypeError, ValueError):
                rot = 0
            if rot not in VALID_ROTATIONS:
                rot = 0
            cam["rotation"] = rot
    if office:
        if not isinstance(data.get("office"), dict):
            data["office"] = {}
        if "shift_start" in office and office["shift_start"] is not None:
            data["office"]["shift_start"] = _parse_hhmm(office["shift_start"])
        if "shift_end" in office and office["shift_end"] is not None:
            data["office"]["shift_end"] = _parse_hhmm(office["shift_end"])
    save_cameras(data, path)
    return data
