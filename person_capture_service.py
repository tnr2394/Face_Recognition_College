"""Stage 1: Person tracking only — writes flushed batches to person_queue/."""

import json
import os
import queue
import re
import shutil
import threading
import time
from urllib.parse import urlparse, urlunparse

import cv2
import torch
from ultralytics import YOLO

from pipeline_io import (
    bbox_iou,
    clamp_bbox,
    clear_pipeline_workdir,
    ensure_dir,
    load_config,
    write_ready,
)

VALID_ROTATIONS = (0, 90, 180, 270)
_GST_FLIP = {
    90: "clockwise",
    180: "rotate-180",
    270: "counterclockwise",
}


def normalize_rotation(degrees):
    try:
        value = int(degrees) % 360
    except (TypeError, ValueError):
        return 0
    return value if value in VALID_ROTATIONS else 0


def is_rtsp_source(source):
    return isinstance(source, str) and source.strip().lower().startswith("rtsp://")


def _strip_rotation_port(parsed):
    port = parsed.port
    if port is None or port not in VALID_ROTATIONS:
        return parsed, 0
    netloc = parsed.hostname or ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return parsed._replace(netloc=netloc), port


def _strip_rotation_path_suffix(parsed):
    path = parsed.path or ""
    match = re.search(r":(\d+)$", path)
    if not match:
        return parsed, 0
    candidate = int(match.group(1))
    if candidate not in VALID_ROTATIONS:
        return parsed, 0
    new_path = path[: match.start()] or "/"
    return parsed._replace(path=new_path), candidate


def parse_video_source(source):
    if isinstance(source, int):
        return source, 0, False
    if isinstance(source, str) and source.isdigit():
        return int(source), 0, False
    if not is_rtsp_source(source):
        return source, 0, False
    parsed = urlparse(source.strip())
    parsed, rotation = _strip_rotation_port(parsed)
    if rotation == 0:
        parsed, rotation = _strip_rotation_path_suffix(parsed)
    return urlunparse(parsed), rotation, True


def apply_frame_rotation(frame, degrees):
    rotation = normalize_rotation(degrees)
    if frame is None or rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def build_rotated_gstreamer_pipeline(rtsp_url, rotation):
    flip = _GST_FLIP.get(normalize_rotation(rotation))
    flip_stage = f"videoflip method={flip} ! " if flip else ""
    return (
        f'rtspsrc location="{rtsp_url}" latency=200 drop-on-latency=true ! '
        "rtpjitterbuffer ! rtph264depay ! avdec_h264 ! "
        f"videoconvert ! {flip_stage}video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def gstreamer_pipeline_opens(pipeline):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return False
    cap.release()
    return True


def open_rtsp_capture(rtsp_url):
    # TCP + socket read timeout — helps detect dead sessions (still needs app-level watchdog)
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def read_latest_frame(cap, drain=3):
    frame = None
    for _ in range(max(1, drain)):
        ret, img = cap.read()
        if ret and img is not None:
            frame = img
    return frame


def is_valid_video_source(source):
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        if source.isdigit():
            return True  # webcam index, e.g. "0"
        if is_rtsp_source(source):
            return True
        return os.path.isfile(source)
    return False


class LatestFrameGrabber:
    """
    Background RTSP reader that keeps the newest frame.

    OpenCV/FFmpeg RTSP clients often stall after hours (half-open socket) while
    MediaMTX still serves new clients (VLC). We track freshness and consecutive
    failures so the capture loop can force-reconnect for 24/7 operation.
    """

    def __init__(self, cap, fail_limit=60):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._frame_seq = 0
        self._last_ok_time = 0.0
        self._fail_streak = 0
        self._fail_limit = max(10, int(fail_limit))
        self._dead = False
        self._dead_reason = ""
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtsp-grabber")
        self._thread.start()

    def _mark_dead(self, reason):
        with self._lock:
            self._dead = True
            self._dead_reason = reason

    def _loop(self):
        while not self._stopped:
            if self._dead:
                time.sleep(0.05)
                continue
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._mark_dead("capture closed")
                    continue
                ret, img = self._cap.read()
            except Exception as e:
                self._fail_streak += 1
                if self._fail_streak >= self._fail_limit:
                    self._mark_dead(f"read exception: {e}")
                time.sleep(0.05)
                continue

            if not ret or img is None:
                self._fail_streak += 1
                if self._fail_streak >= self._fail_limit:
                    self._mark_dead(f"no frames ({self._fail_streak} failures)")
                time.sleep(0.02)
                continue

            with self._lock:
                self._frame = img
                self._frame_seq += 1
                self._last_ok_time = time.time()
                self._fail_streak = 0

    def read_fresh(self, last_seq=-1):
        """
        Return (frame_copy, seq, age_sec) only when a newer frame arrived.
        If stream is dead, raises ConnectionError.
        """
        with self._lock:
            if self._dead:
                raise ConnectionError(
                    f"RTSP grabber stalled: {self._dead_reason or 'unknown'}"
                )
            if self._frame is None or self._frame_seq == last_seq:
                age = (
                    (time.time() - self._last_ok_time)
                    if self._last_ok_time > 0
                    else float("inf")
                )
                return None, last_seq, age
            age = time.time() - self._last_ok_time
            return self._frame.copy(), self._frame_seq, age

    def age_sec(self):
        with self._lock:
            if self._last_ok_time <= 0:
                return float("inf")
            return time.time() - self._last_ok_time

    def is_dead(self):
        with self._lock:
            return self._dead

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)


class PersonCaptureService:
    """Track persons and flush sample batches to person_queue/."""

    def __init__(self, config=None):
        self.config = config or load_config()
        p = self.config["person"]
        self.queue_dir = p["queue_dir"]
        staging_subdir = p.get("staging_subdir", ".staging")
        self.staging_root = os.path.join(self.queue_dir, staging_subdir)
        self.min_box_area_ratio = p["min_box_area_ratio"]
        self.max_samples = p["max_samples"]
        self.first_flush = float(p.get("first_flush_sec", 5))
        self.flush_interval = float(p.get("flush_interval_sec", 10))
        self.flush_stable = float(p.get("flush_stable_sec", 3))
        self.min_samples = p["min_samples_before_flush"]
        self.track_resit_cooldown = float(p.get("track_resit_cooldown_sec", 60))
        self.jpeg_quality = int(p.get("jpeg_quality", 85))
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self.vid_stride = p["vid_stride"]
        self.rtsp_reconnect_delay = p["rtsp_reconnect_delay"]
        # Force reconnect if no *new* RTSP frame for this many seconds (OpenCV stall)
        self.rtsp_stall_timeout = float(p.get("rtsp_stall_timeout_sec", 12))
        self.tracker_config = p["tracker"]
        self.person_conf = p["conf"]
        self.max_bbox_y1_ratio = p.get("max_bbox_y1_ratio", 0.72)
        self.receding_peak_ratio = p.get("receding_peak_area_ratio", 0.82)
        self.receding_shrink_ratio = p.get("receding_shrink_ratio", 0.97)
        self.receding_shrink_frames = p.get("receding_shrink_frames", 2)
        self.receding_require_shrink_streak = p.get("receding_require_shrink_streak", False)
        self.receding_min_track_samples = p.get("receding_min_track_samples", 2)
        self.clear_pipeline_on_start = bool(p.get("clear_pipeline_on_start", True))
        self.sample_stride = max(1, int(p.get("sample_stride", 2)))
        self.id_break_min_iou = float(p.get("id_break_min_iou", 0.12))
        self.id_break_max_center_frac = float(p.get("id_break_max_center_frac", 0.22))
        writer_qsize = int(p.get("writer_queue_size", 48))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_model = YOLO(p["model"])
        self.active_track_ids = set()
        self.track_meta = {}
        self._meta_lock = threading.Lock()
        self._staging_lock = threading.Lock()
        self._write_q = queue.Queue(maxsize=max(8, writer_qsize))
        self._flush_q = queue.Queue()
        self._writer_stop = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_cv = threading.Condition(self._pending_lock)
        # pending keyed by (track_id, gen) so flush never blocks new samples
        self._pending_writes = {}
        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="sample-writer")
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True, name="batch-flusher")
        self._writer.start()
        self._flusher.start()

        ensure_dir(self.queue_dir)
        ensure_dir(self.staging_root)
        profile = self.config.get("camera_profile", "standard")
        print(
            f"Person capture on {self.device}, queue: {self.queue_dir}, "
            f"staging: {self.staging_root}, profile: {profile}, "
            f"sample_stride={self.sample_stride}, async_flush=on"
        )

    def _staging_dir(self, track_id, gen=None):
        if gen is None:
            with self._meta_lock:
                gen = self.track_meta.get(track_id, {}).get("gen", 0)
        return os.path.join(self.staging_root, f"person_{track_id}_g{gen}")

    def _pending_key(self, track_id, gen):
        return (int(track_id), int(gen))

    def _inc_pending(self, track_id, gen):
        key = self._pending_key(track_id, gen)
        with self._pending_cv:
            self._pending_writes[key] = self._pending_writes.get(key, 0) + 1

    def _dec_pending(self, track_id, gen):
        key = self._pending_key(track_id, gen)
        with self._pending_cv:
            left = self._pending_writes.get(key, 0) - 1
            if left <= 0:
                self._pending_writes.pop(key, None)
            else:
                self._pending_writes[key] = left
            self._pending_cv.notify_all()

    def _wait_gen_writes(self, track_id, gen, timeout=15.0):
        key = self._pending_key(track_id, gen)
        deadline = time.time() + timeout
        with self._pending_cv:
            while self._pending_writes.get(key, 0) > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(f"Warning: timed out waiting writes track={track_id} gen={gen}")
                    break
                self._pending_cv.wait(timeout=min(0.25, remaining))

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _is_id_break(self, last_bbox, new_bbox, frame_w, frame_h):
        """Detect ByteTrack ID swap / teleport onto another person."""
        if last_bbox is None:
            return False
        iou = bbox_iou(last_bbox, new_bbox)
        if iou >= self.id_break_min_iou:
            return False
        lx, ly = self._bbox_center(last_bbox)
        nx, ny = self._bbox_center(new_bbox)
        diag = max(1.0, (frame_w ** 2 + frame_h ** 2) ** 0.5)
        dist = ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5
        return dist >= (self.id_break_max_center_frac * diag)

    def _prepare_roi(self, video_source, draw_roi=False):
        if not draw_roi:
            return
        from draw_roi import draw_roi_interactive, grab_first_frame

        frame = grab_first_frame(video_source)
        roi = draw_roi_interactive(frame)
        if roi is None:
            print("ROI draw cancelled.")
            return
        roi_path = (self.config.get("roi") or {}).get("file", "roi.json")
        from pipeline_io import save_roi

        save_roi(roi, path=roi_path, config=self.config)

    def _tracker_kwargs(self, show=True, stream=True):
        return dict(
            stream=stream,
            show=show,
            vid_stride=self.vid_stride,
            persist=True,
            device=self.device,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.person_conf,
            iou=0.5,
            verbose=False,
        )

    @staticmethod
    def _crop_person(full_frame, meta):
        h, w = full_frame.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(meta["x1"], meta["y1"], meta["x2"], meta["y2"], w, h)
        return full_frame[y1:y2, x1:x2].copy()

    def _writer_loop(self):
        while not self._writer_stop.is_set() or not self._write_q.empty():
            try:
                job = self._write_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self._write_q.task_done()
                break
            track_id = gen = None
            try:
                kind = job[0]
                track_id = job[1]
                gen = job[2]
                if kind == "sample":
                    _, _, _, person_dir, frame_num, meta, frame = job
                    ensure_dir(person_dir)
                    cv2.imwrite(
                        os.path.join(person_dir, f"frame_{frame_num}_full.jpg"),
                        frame,
                        self.jpeg_params,
                    )
                    crop = self._crop_person(frame, meta)
                    if crop.size > 0:
                        cv2.imwrite(
                            os.path.join(person_dir, f"frame_{frame_num}_person.jpg"),
                            crop,
                            self.jpeg_params,
                        )
                    with open(
                        os.path.join(person_dir, f"frame_{frame_num}_meta.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(meta, f)
                elif kind == "remove":
                    _, _, _, person_dir, frame_num = job
                    for name in (
                        f"frame_{frame_num}_meta.json",
                        f"frame_{frame_num}_full.jpg",
                        f"frame_{frame_num}_person.jpg",
                    ):
                        path = os.path.join(person_dir, name)
                        if os.path.isfile(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                elif kind == "discard_dir":
                    _, _, _, person_dir = job
                    shutil.rmtree(person_dir, ignore_errors=True)
            except Exception as e:
                print(f"Sample writer error: {e}")
            finally:
                if track_id is not None and gen is not None:
                    self._dec_pending(track_id, gen)
                self._write_q.task_done()

    def _enqueue_write(self, job):
        """job: (kind, track_id, gen, ...)"""
        track_id, gen = job[1], job[2]
        self._inc_pending(track_id, gen)
        try:
            self._write_q.put(job, timeout=0.05)
        except queue.Full:
            self._dec_pending(track_id, gen)
            # Drop under load — never block the live capture loop

    def _flush_loop(self):
        while not self._writer_stop.is_set() or not self._flush_q.empty():
            try:
                job = self._flush_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self._flush_q.task_done()
                break
            try:
                action = job[0]
                if action == "publish":
                    _, track_id, gen, staging_dir, batch, reason, sample_count = job
                    self._wait_gen_writes(track_id, gen)
                    if sample_count < 1 or not os.path.isdir(staging_dir):
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        continue
                    ts = int(time.time())
                    output_name = f"person_{track_id}_b{batch}_{ts}"
                    output_dir = os.path.join(self.queue_dir, output_name)
                    with self._staging_lock:
                        if os.path.isdir(output_dir):
                            shutil.rmtree(output_dir, ignore_errors=True)
                        try:
                            os.rename(staging_dir, output_dir)
                        except OSError:
                            shutil.copytree(staging_dir, output_dir)
                            shutil.rmtree(staging_dir, ignore_errors=True)
                        write_ready(output_dir)
                    label = {
                        "lost": "left frame",
                        "interval": "interval",
                        "final": "video end",
                        "id_break": "id-break split",
                    }.get(reason, reason)
                    print(
                        f"Track {track_id} flush ({label}) -> {output_name} "
                        f"({sample_count} samples)"
                    )
                elif action == "discard":
                    _, track_id, gen, staging_dir = job
                    self._wait_gen_writes(track_id, gen, timeout=5.0)
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    print(f"Track {track_id} discarded contaminated buffer gen={gen}")
            except Exception as e:
                print(f"Flush worker error: {e}")
            finally:
                self._flush_q.task_done()

    def _is_receding(self, track_id, box_area, peak_area):
        """True when person bbox is well below this track's peak (walking away)."""
        if peak_area <= 0 or box_area >= peak_area * self.receding_peak_ratio:
            with self._meta_lock:
                meta = self.track_meta.get(track_id, {})
                meta["last_area"] = box_area
                self.track_meta[track_id] = meta
            return False

        below_peak = True
        with self._meta_lock:
            meta = self.track_meta.get(track_id, {})
            samples_seen = meta.get("samples_seen", 0)
            if samples_seen < self.receding_min_track_samples:
                meta["last_area"] = box_area
                self.track_meta[track_id] = meta
                return False

            last_area = meta.get("last_area")
            shrink_streak = meta.get("shrink_streak", 0)
            if last_area and box_area < last_area * self.receding_shrink_ratio:
                shrink_streak += 1
            else:
                shrink_streak = 0
            meta["last_area"] = box_area
            meta["shrink_streak"] = shrink_streak
            self.track_meta[track_id] = meta

        if self.receding_require_shrink_streak:
            return below_peak and shrink_streak >= self.receding_shrink_frames
        return below_peak

    def _store_sample(self, track_id, gen, person_dir, frame_num, bbox, box_area, frame, *, receding=False):
        meta = {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
            "area": int(box_area),
            "receding": bool(receding),
        }
        with self._meta_lock:
            tmeta = self.track_meta.setdefault(track_id, {})
            samples = tmeta.setdefault("samples", {})

            if frame_num in samples:
                samples[frame_num] = meta["area"]
                self._enqueue_write(
                    ("sample", track_id, gen, person_dir, frame_num, meta, frame)
                )
                return True

            if len(samples) < self.max_samples:
                samples[frame_num] = meta["area"]
                self._enqueue_write(
                    ("sample", track_id, gen, person_dir, frame_num, meta, frame)
                )
                return True

            smallest_frame = min(samples, key=lambda fn: samples[fn])
            if meta["area"] <= samples[smallest_frame]:
                return False
            samples.pop(smallest_frame, None)
            self._enqueue_write(("remove", track_id, gen, person_dir, smallest_frame))
            samples[frame_num] = meta["area"]
            self._enqueue_write(
                ("sample", track_id, gen, person_dir, frame_num, meta, frame)
            )
            return True

    def _sample_count(self, track_id):
        with self._meta_lock:
            return len(self.track_meta.get(track_id, {}).get("samples", {}))

    def _schedule_flush(self, track_id, reason="interval"):
        """Non-blocking: bump generation immediately, publish old buffer in background."""
        with self._meta_lock:
            meta = self.track_meta.get(track_id)
            if meta is None:
                return
            gen = int(meta.get("gen", 0))
            batch = int(meta.get("batch", 0))
            sample_count = len(meta.get("samples", {}))
            staging_dir = self._staging_dir(track_id, gen=gen)

            min_required = self.min_samples if reason in ("interval", "id_break") else 1
            if sample_count < min_required:
                if reason in ("lost", "final", "id_break"):
                    # Drop empty/contaminated buffer without publishing
                    self._flush_q.put(("discard", track_id, gen, staging_dir))
                    if reason in ("lost", "final"):
                        self.track_meta.pop(track_id, None)
                    else:
                        meta["gen"] = gen + 1
                        meta["samples"] = {}
                        meta["peak_buffer_area"] = 0
                        meta["last_bbox"] = None
                        meta["frames_since_sample"] = 0
                return

            # Publish current gen; immediately open a new gen so capture never waits
            if reason == "interval":
                meta["batch"] = batch + 1
                meta["last_flush"] = time.time()
                meta["samples"] = {}
                meta["gen"] = gen + 1
                meta["frames_since_sample"] = 0
                if batch == 0:
                    meta["resit_until"] = time.time() + self.track_resit_cooldown
                self.track_meta[track_id] = meta
            elif reason == "id_break":
                # Contaminated mix — discard, do not send to ranker
                self._flush_q.put(("discard", track_id, gen, staging_dir))
                meta["gen"] = gen + 1
                meta["samples"] = {}
                meta["peak_buffer_area"] = 0
                meta["peak_area_time"] = time.time()
                meta["last_bbox"] = None
                meta["frames_since_sample"] = 0
                self.track_meta[track_id] = meta
                return
            else:
                self.track_meta.pop(track_id, None)

        self._flush_q.put(
            ("publish", track_id, gen, staging_dir, batch, reason, sample_count)
        )

    def _flush_track(self, track_id, reason="interval"):
        self._schedule_flush(track_id, reason=reason)

    def _interval_due(self, meta, now):
        batch = meta.get("batch", 0)
        last_flush = meta.get("last_flush", now)
        interval = self.first_flush if batch == 0 else self.flush_interval
        return now - last_flush >= interval

    def _check_interval_flushes(self, active_ids):
        now = time.time()
        for track_id in list(active_ids):
            with self._meta_lock:
                meta = self.track_meta.get(track_id)
                if meta is None:
                    continue
                # Sitting cooldown: after first flush, pause further interval flushes
                if meta.get("batch", 0) >= 1 and now < float(meta.get("resit_until", 0)):
                    continue
                sample_count = len(meta.get("samples", {}))
                buffer_full = sample_count >= self.max_samples
                due = self._interval_due(meta, now) or (
                    buffer_full and now - meta.get("peak_area_time", now) >= self.flush_stable
                )
                if not due:
                    continue
                peak_area_time = meta.get("peak_area_time", 0)

            if sample_count < self.min_samples:
                with self._meta_lock:
                    if track_id in self.track_meta:
                        self.track_meta[track_id]["last_flush"] = now
                continue
            if now - peak_area_time < self.flush_stable and not buffer_full:
                continue
            # For buffer-full flush still require peak stable briefly
            if buffer_full and now - peak_area_time < self.flush_stable:
                continue
            self._flush_track(track_id, reason="interval")

    def _process_frame(self, results, frame_num):
        current_ids = set()
        now = time.time()

        if results.boxes is None or results.boxes.id is None:
            lost = self.active_track_ids - current_ids
            for tid in lost:
                self._flush_track(int(tid), reason="lost")
            self.active_track_ids = current_ids
            self._check_interval_flushes(current_ids)
            return

        # One frame reference for disk; draw overlay on a separate copy only if showing
        clean_frame = results.orig_img
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        fh, fw = clean_frame.shape[:2]
        frame_area = fh * fw
        draw = None

        for i, track_id in enumerate(track_ids):
            tid = int(track_id)
            current_ids.add(tid)
            x1, y1, x2, y2 = clamp_bbox(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], fw, fh)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < frame_area * self.min_box_area_ratio:
                continue
            if y1 > fh * self.max_bbox_y1_ratio:
                continue

            person_bbox = (x1, y1, x2, y2)

            orphan_cleanup = False
            id_break = False
            with self._meta_lock:
                if tid not in self.track_meta:
                    self.track_meta[tid] = {
                        "last_flush": now,
                        "batch": 0,
                        "gen": 0,
                        "peak_buffer_area": 0,
                        "peak_area_time": now,
                        "last_area": None,
                        "last_bbox": None,
                        "shrink_streak": 0,
                        "samples_seen": 0,
                        "frames_since_sample": 0,
                        "samples": {},
                        "resit_until": 0,
                    }
                    orphan_cleanup = True
                else:
                    last_bbox = self.track_meta[tid].get("last_bbox")
                    if last_bbox is not None and self._is_id_break(
                        last_bbox, person_bbox, fw, fh
                    ):
                        id_break = True

                peak_area = self.track_meta[tid].get("peak_buffer_area", 0)
                self.track_meta[tid]["samples_seen"] = (
                    self.track_meta[tid].get("samples_seen", 0) + 1
                )
                in_resit = (
                    self.track_meta[tid].get("batch", 0) >= 1
                    and now < float(self.track_meta[tid].get("resit_until", 0))
                )
                staging_gen = self.track_meta[tid].get("gen", 0)

            if orphan_cleanup and os.path.isdir(self.staging_root):
                for name in list(os.listdir(self.staging_root)):
                    if name.startswith(f"person_{tid}_g") or name == f"person_{tid}":
                        shutil.rmtree(
                            os.path.join(self.staging_root, name),
                            ignore_errors=True,
                        )

            if id_break:
                # Drop mixed buffer; never publish swapped-ID samples together
                print(f"Track {tid} identity break (bbox jump) — discarding buffer")
                self._schedule_flush(tid, reason="id_break")
                with self._meta_lock:
                    if tid in self.track_meta:
                        staging_gen = self.track_meta[tid].get("gen", 0)
                        peak_area = 0

            if in_resit:
                if draw is None:
                    draw = clean_frame.copy()
                cv2.rectangle(draw, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(
                    draw, f"ID:{tid} cool", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1,
                )
                with self._meta_lock:
                    if tid in self.track_meta:
                        self.track_meta[tid]["last_bbox"] = person_bbox
                continue

            with self._meta_lock:
                meta = self.track_meta.get(tid, {})
                meta["frames_since_sample"] = int(meta.get("frames_since_sample", 0)) + 1
                should_sample = meta["frames_since_sample"] >= self.sample_stride
                if should_sample:
                    meta["frames_since_sample"] = 0
                staging_gen = meta.get("gen", staging_gen)
                self.track_meta[tid] = meta

            saved = False
            if should_sample:
                receding = self._is_receding(tid, box_area, max(peak_area, box_area))
                staging_dir = self._staging_dir(tid, gen=staging_gen)
                os.makedirs(staging_dir, exist_ok=True)
                # Copy once for the writer thread only
                saved = self._store_sample(
                    tid,
                    staging_gen,
                    staging_dir,
                    frame_num,
                    person_bbox,
                    box_area,
                    clean_frame.copy(),
                    receding=receding,
                )
                with self._meta_lock:
                    meta = self.track_meta.get(tid)
                    if meta:
                        max_area = (
                            max(meta.get("samples", {}).values())
                            if meta.get("samples")
                            else 0
                        )
                        if max_area > meta.get("peak_buffer_area", 0):
                            meta["peak_buffer_area"] = max_area
                            meta["peak_area_time"] = now
                        meta["last_bbox"] = person_bbox
            else:
                with self._meta_lock:
                    if tid in self.track_meta:
                        self.track_meta[tid]["last_bbox"] = person_bbox

            if draw is None:
                draw = clean_frame.copy()
            color = (0, 200, 0) if saved or not should_sample else (0, 180, 255)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                draw, f"ID:{tid}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        if draw is not None:
            results.orig_img = draw
        lost = self.active_track_ids - current_ids
        for tid in lost:
            self._flush_track(int(tid), reason="lost")
        self.active_track_ids = current_ids
        self._check_interval_flushes(current_ids)

    def _run_yolo_stream(self, source, show=True):
        """File / GStreamer sources — Ultralytics owns the reader."""
        frame_num = 0
        for results in self.person_model.track(source=source, **self._tracker_kwargs(show=show)):
            frame_num += 1
            self._process_frame(results, frame_num)

    def _run_latest_frame_loop(self, rtsp_url, rotation=0, show=True):
        """
        Always process the newest RTSP frame. If inference/disk lags, old frames
        are dropped. If the OpenCV client stalls (common after hours with MediaMTX),
        raise so the outer loop force-reconnects — VLC working does not mean our
        existing TCP session is healthy.
        """
        cap = open_rtsp_capture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP: {rtsp_url}")
        grabber = LatestFrameGrabber(cap)
        frame_num = 0
        last_seq = -1
        last_progress = time.time()
        print(
            f"RTSP latest-frame grabber on (rotation={rotation}, "
            f"stall_timeout={self.rtsp_stall_timeout}s)"
        )
        try:
            while True:
                try:
                    raw, seq, age = grabber.read_fresh(last_seq)
                except ConnectionError as e:
                    raise ConnectionError(str(e)) from e

                if raw is None:
                    # No newer frame yet — do not reprocess a stale frame
                    if age >= self.rtsp_stall_timeout:
                        raise ConnectionError(
                            f"RTSP stall: no new frame for {age:.1f}s "
                            f"(MediaMTX may still work for new clients)"
                        )
                    if grabber.is_dead() or not cap.isOpened():
                        raise ConnectionError("RTSP capture closed")
                    time.sleep(0.01)
                    continue

                last_seq = seq
                last_progress = time.time()
                frame = apply_frame_rotation(raw, rotation) if rotation else raw
                results_list = self.person_model.track(
                    frame, **self._tracker_kwargs(show=False, stream=False)
                )
                if not results_list:
                    if time.time() - last_progress >= self.rtsp_stall_timeout:
                        raise ConnectionError("RTSP stall during tracking")
                    continue
                result = results_list[0]
                result.orig_img = frame
                frame_num += 1
                self._process_frame(result, frame_num)
                if show:
                    vis = result.orig_img
                    cv2.imshow("Person Capture", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
        finally:
            grabber.stop()
            try:
                cap.release()
            except Exception:
                pass

    def _run_rotated_capture_loop(self, rtsp_url, rotation, show=True):
        self._run_latest_frame_loop(rtsp_url, rotation=rotation, show=show)

    def _run_rtsp_stream(self, rtsp_url, rotation, show=True):
        while True:
            try:
                # Always latest-frame path for RTSP — never let Ultralytics buffer lag
                self._run_latest_frame_loop(rtsp_url, rotation=rotation, show=show)
                print(f"RTSP ended, reconnecting in {self.rtsp_reconnect_delay}s...")
                time.sleep(self.rtsp_reconnect_delay)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"RTSP error: {e}. Reconnecting in {self.rtsp_reconnect_delay}s...")
                time.sleep(self.rtsp_reconnect_delay)

    def _finalize_active_tracks(self):
        names = []
        if os.path.isdir(self.staging_root):
            names.extend(os.listdir(self.staging_root))
        seen = set()
        for name in names:
            # person_12_g0 or legacy person_12
            match = re.match(r"^person_(\d+)(?:_g\d+)?$", name)
            if not match:
                continue
            track_id = int(match.group(1))
            if track_id in seen:
                continue
            seen.add(track_id)
            self._flush_track(track_id, reason="final")
        for track_id in list(self.active_track_ids):
            if int(track_id) not in seen:
                self._flush_track(int(track_id), reason="final")

    def _shutdown_writer(self):
        # Drain flushes first so pending publishes finish
        self._flush_q.join()
        self._write_q.join()
        self._writer_stop.set()
        try:
            self._flush_q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._write_q.put_nowait(None)
        except queue.Full:
            pass
        self._flusher.join(timeout=8.0)
        self._writer.join(timeout=5.0)

    def run(self, video_source, show=True, draw_roi=False):
        if self.clear_pipeline_on_start:
            # ByteTrack IDs restart at 1 after PM2 restart; wipe old person_N folders
            # so recognition_folder/.processed does not block new tracks.
            clear_pipeline_workdir(self.config)
            ensure_dir(self.queue_dir)
            ensure_dir(self.staging_root)
        self._prepare_roi(video_source, draw_roi=draw_roi)
        play_source, rotation, is_rtsp = parse_video_source(video_source)
        try:
            if is_rtsp:
                print(f"RTSP: {play_source}, rotation={rotation}")
                self._run_rtsp_stream(play_source, rotation, show=show)
            else:
                source = play_source if play_source is not None else video_source
                print(f"Processing: {source}")
                self._run_yolo_stream(source, show=show)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self._finalize_active_tracks()
            self._shutdown_writer()
            if show:
                cv2.destroyAllWindows()
            print("Person capture finished.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: person capture service")
    parser.add_argument(
        "video_source",
        nargs="?",
        default=None,
        help="RTSP URL, video file, webcam index (or set CAPTURE_SOURCE env)",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--draw-roi",
        action="store_true",
        help="Draw ROI on first frame before capture (saves roi.json)",
    )
    args = parser.parse_args()

    # Prefer env so Windows cmd does not strip '&' from RTSP query strings
    video_source = (
        os.environ.get("CAPTURE_SOURCE", "").strip()
        or args.video_source
        or r"test_videos\Knowns.mp4"
    )

    if not is_valid_video_source(video_source):
        print(f"Invalid source: {video_source}")
        return

    print(f"Capture source: {video_source}")
    PersonCaptureService().run(
        video_source, show=not args.no_show, draw_roi=args.draw_roi
    )


if __name__ == "__main__":
    main()
