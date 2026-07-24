"""Shared folder contracts and I/O helpers for the multi-service pipeline."""

import json
import os
import re
import shutil
import time
from pathlib import Path

import yaml

CONFIG_PATH = "pipeline_config.yaml"
READY_MARKER = ".ready"
PROCESSED_MARKER = ".processed"


def _deep_merge(base, overlay):
    """Merge overlay dict into base (overlay wins on conflicts)."""
    for key, value in overlay.items():
        if key == "profiles":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    profile_name = config.get("camera_profile")
    if profile_name:
        profiles = config.get("profiles") or {}
        overlay = profiles.get(profile_name)
        if overlay:
            _deep_merge(config, overlay)
        else:
            print(f"Warning: camera_profile '{profile_name}' has no profiles.{profile_name} section")
    return config


def clamp_bbox(x1, y1, x2, y2, frame_w, frame_h):
    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(x1 + 1, min(int(x2), frame_w))
    y2 = max(y1 + 1, min(int(y2), frame_h))
    return x1, y1, x2, y2


def frame_num_from_meta_filename(filename):
    match = re.match(r"frame_(\d+)_meta\.json", filename)
    return match.group(1) if match else None


def list_meta_files(person_dir):
    if not os.path.isdir(person_dir):
        return []
    return sorted(f for f in os.listdir(person_dir) if f.endswith("_meta.json"))


def read_sample_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_person_from_full(full_frame, meta):
    h, w = full_frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(meta["x1"], meta["y1"], meta["x2"], meta["y2"], w, h)
    return full_frame[y1:y2, x1:x2].copy()


def point_inside_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_intersection_over_face(face_box, other_box):
    """Fraction of face box area that overlaps other_box (0..1)."""
    fx1, fy1, fx2, fy2 = face_box
    ox1, oy1, ox2, oy2 = other_box
    ix1 = max(fx1, ox1)
    iy1 = max(fy1, oy1)
    ix2 = min(fx2, ox2)
    iy2 = min(fy2, oy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)

def write_ready(folder):
    with open(os.path.join(folder, READY_MARKER), "w", encoding="utf-8") as f:
        f.write("ready")


def has_ready(folder):
    return os.path.isfile(os.path.join(folder, READY_MARKER))


def mark_processed(folder):
    with open(os.path.join(folder, PROCESSED_MARKER), "w", encoding="utf-8") as f:
        f.write("done")


def is_processed(folder):
    return os.path.isfile(os.path.join(folder, PROCESSED_MARKER))


def clear_processed(folder):
    """Remove processed marker so a folder can be handed off again."""
    marker = os.path.join(folder, PROCESSED_MARKER)
    if os.path.isfile(marker):
        os.remove(marker)


def folder_inactive(folder, wait_sec=5):
    """True when no file was modified recently. Ready batches skip the wait."""
    if has_ready(folder):
        return True
    now = time.time()
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) < wait_sec:
            return False
    return True


def parse_track_id_from_folder(folder_name):
    match = re.match(r"^person_(\d+)", folder_name)
    return int(match.group(1)) if match else None


def get_pending_folders(queue_dir, *, require_ready=True, skip_processed=True):
    if not os.path.isdir(queue_dir):
        return []

    pending = []
    for folder_name in os.listdir(queue_dir):
        if not folder_name.startswith("person_"):
            continue
        folder_path = os.path.join(queue_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if skip_processed and is_processed(folder_path):
            continue
        ready_path = os.path.join(folder_path, READY_MARKER)
        if require_ready and not os.path.isfile(ready_path):
            continue
        ready_mtime = os.path.getmtime(ready_path) if os.path.isfile(ready_path) else 0
        pending.append((ready_mtime, folder_name, folder_path, ready_path))

    pending.sort(key=lambda item: item[0])
    return pending


def iter_staging_samples(staging_dir):
    """Yield (frame_num, meta, full_path, person_path) sorted by largest person area."""
    meta_files = list_meta_files(staging_dir)
    meta_files.sort(
        key=lambda m: read_sample_meta(os.path.join(staging_dir, m)).get("area", 0),
        reverse=True,
    )
    for meta_file in meta_files:
        frame_num = frame_num_from_meta_filename(meta_file)
        if not frame_num:
            continue
        meta_path = os.path.join(staging_dir, meta_file)
        full_path = os.path.join(staging_dir, f"frame_{frame_num}_full.jpg")
        person_path = os.path.join(staging_dir, f"frame_{frame_num}_person.jpg")
        if not os.path.isfile(full_path):
            continue
        try:
            meta = read_sample_meta(meta_path)
        except (json.JSONDecodeError, OSError):
            continue
        yield frame_num, meta, full_path, person_path


def count_pending_batches(queue_dir):
    """Count person_queue folders that have .ready but not .processed."""
    return len(get_pending_folders(queue_dir, require_ready=True, skip_processed=True))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _rmtree_contents(path):
    """Delete path if it exists (file or directory). Returns True if removed."""
    if not path or not os.path.exists(path):
        return False
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        return True
    except OSError:
        return False


def clear_pipeline_workdir(config=None):
    """
    Wipe ephemeral pipeline folders so ByteTrack IDs restarting at 1
    do not collide with old recognition_folder/person_N (.processed) dirs.
    """
    if config is None:
        config = load_config()
    person_cfg = config.get("person") or {}
    ofiq_cfg = config.get("ofiq") or {}
    ranker_cfg = config.get("face_ranker") or {}
    roi_cfg = config.get("roi") or {}

    targets = [
        person_cfg.get("queue_dir", "person_queue"),
        ofiq_cfg.get("recognition_dir", "recognition_folder"),
        ofiq_cfg.get("low_ofiq_dir", "low_ofiq_faces"),
        ranker_cfg.get("output_dir", "ranked_faces"),
        ranker_cfg.get("rejected_dir", "rejected_faces"),
        roi_cfg.get("outside_roi_dir", "outside_roi_faces"),
    ]
    # Deduplicate while preserving order
    seen = set()
    removed = []
    for path in targets:
        if not path or path in seen:
            continue
        seen.add(path)
        if _rmtree_contents(path):
            removed.append(path)
            ensure_dir(path)

    # Recognition employee cooldown is fine to keep; track IDs are not used there.
    if removed:
        print(f"Cleared pipeline folders for fresh track IDs: {', '.join(removed)}")
    else:
        print("Pipeline folders already empty (nothing to clear)")
    return removed


def roi_config_path(config):
    roi_cfg = config.get("roi") or {}
    return roi_cfg.get("file", "roi.json")


def is_roi_active(config=None):
    """ROI is on when enabled in config or in roi.json."""
    if config is None:
        config = load_config()
    roi_cfg = config.get("roi") or {}
    if roi_cfg.get("enabled"):
        return True
    roi_path = roi_config_path(config)
    if not os.path.isfile(roi_path):
        return False
    try:
        with open(roi_path, encoding="utf-8") as f:
            return bool(json.load(f).get("enabled"))
    except (json.JSONDecodeError, OSError):
        return False


def load_roi(config=None, path=None):
    """Load ROI dict from JSON. Returns None if missing or inactive."""
    if config is None:
        config = load_config()
    if not is_roi_active(config):
        return None
    roi_path = path or roi_config_path(config)
    if not os.path.isfile(roi_path):
        return None
    try:
        with open(roi_path, encoding="utf-8") as f:
            roi = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    for key in ("x1", "y1", "x2", "y2"):
        if key not in roi:
            return None
    roi["enabled"] = True
    return roi


def save_roi(roi, path=None, config=None):
    if config is None:
        config = load_config()
    roi_path = path or roi_config_path(config)
    ensure_dir(os.path.dirname(roi_path) or ".")
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)
    print(f"Saved ROI to {roi_path}")


def roi_pixel_bounds(roi, frame_w, frame_h):
    if not roi:
        return None
    x1 = int(min(roi["x1"], roi["x2"]) * frame_w)
    y1 = int(min(roi["y1"], roi["y2"]) * frame_h)
    x2 = int(max(roi["x1"], roi["x2"]) * frame_w)
    y2 = int(max(roi["y1"], roi["y2"]) * frame_h)
    return clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)


def face_in_roi(face_box, frame_w, frame_h, roi, *, mode="center", min_overlap=0.5):
    """Return True if ROI is off or the face satisfies the ROI rule."""
    return bbox_in_roi(
        face_box,
        frame_w,
        frame_h,
        roi,
        mode=mode,
        min_overlap=min_overlap,
        anchor_y=0.5,
    )


def bbox_in_roi(bbox, frame_w, frame_h, roi, *, mode="center", min_overlap=0.5, anchor_y=0.5):
    """Check person/face bbox against ROI. anchor_y: 0=top, 0.5=center, 1=bottom of box."""
    if not roi:
        return True
    bounds = roi_pixel_bounds(roi, frame_w, frame_h)
    if bounds is None:
        return True
    if mode == "overlap":
        return bbox_intersection_over_face(bbox, bounds) >= min_overlap
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = y1 + anchor_y * (y2 - y1)
    return point_inside_bbox(cx, cy, bounds)
