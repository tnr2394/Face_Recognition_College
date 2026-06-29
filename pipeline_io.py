"""Shared folder contracts and I/O helpers for the multi-service pipeline."""

import json
import os
import re
import time
from pathlib import Path

import yaml

CONFIG_PATH = "pipeline_config.yaml"
READY_MARKER = ".ready"
PROCESSED_MARKER = ".processed"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def mark_processed(folder):
    with open(os.path.join(folder, PROCESSED_MARKER), "w", encoding="utf-8") as f:
        f.write("done")


def is_processed(folder):
    return os.path.isfile(os.path.join(folder, PROCESSED_MARKER))


def folder_inactive(folder, wait_sec=5):
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
