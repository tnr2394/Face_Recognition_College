"""Stage 1: Person tracking only — writes flushed batches to person_queue/."""

import json
import os
import re
import shutil
import threading
import time
from urllib.parse import urlparse, urlunparse

import cv2
import torch
from ultralytics import YOLO

from pipeline_io import (
    clamp_bbox,
    ensure_dir,
    frame_num_from_meta_filename,
    list_meta_files,
    load_config,
    read_sample_meta,
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
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
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
        if is_rtsp_source(source):
            return True
        return os.path.isfile(source)
    return False


class PersonCaptureService:
    """Track persons and flush sample batches to person_queue/."""

    def __init__(self, config=None):
        self.config = config or load_config()
        p = self.config["person"]
        self.queue_dir = p["queue_dir"]
        self.min_box_area_ratio = p["min_box_area_ratio"]
        self.max_samples = p["max_samples"]
        self.flush_interval = p["flush_interval_sec"]
        self.flush_stable = p["flush_stable_sec"]
        self.min_samples = p["min_samples_before_flush"]
        self.vid_stride = p["vid_stride"]
        self.rtsp_reconnect_delay = p["rtsp_reconnect_delay"]
        self.tracker_config = p["tracker"]
        self.person_conf = p["conf"]
        self.max_bbox_y1_ratio = p.get("max_bbox_y1_ratio", 0.72)
        self.receding_peak_ratio = p.get("receding_peak_area_ratio", 0.82)
        self.receding_shrink_ratio = p.get("receding_shrink_ratio", 0.97)
        self.receding_shrink_frames = p.get("receding_shrink_frames", 2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_model = YOLO(p["model"])
        self.active_track_ids = set()
        self.track_meta = {}
        self._meta_lock = threading.Lock()
        self._staging_lock = threading.Lock()
        ensure_dir(self.queue_dir)
        print(f"Person capture on {self.device}, queue: {self.queue_dir}")

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

    def _remove_sample(self, person_dir, frame_num):
        for name in (
            f"frame_{frame_num}_meta.json",
            f"frame_{frame_num}_full.jpg",
            f"frame_{frame_num}_person.jpg",
        ):
            path = os.path.join(person_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    def _is_receding(self, track_id, box_area, peak_area):
        """True when bbox is shrinking away from camera (walking backward)."""
        if peak_area <= 0:
            return False
        below_peak = box_area < peak_area * self.receding_peak_ratio
        with self._meta_lock:
            meta = self.track_meta.get(track_id, {})
            last_area = meta.get("last_area")
            shrink_streak = meta.get("shrink_streak", 0)
            if last_area and box_area < last_area * self.receding_shrink_ratio:
                shrink_streak += 1
            else:
                shrink_streak = 0
            meta["last_area"] = box_area
            meta["shrink_streak"] = shrink_streak
            self.track_meta[track_id] = meta
        shrinking = shrink_streak >= self.receding_shrink_frames
        return below_peak and shrinking

    def _save_sample(self, person_dir, frame_num, bbox, box_area, frame, *, receding=False):
        meta = {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "area": box_area,
            "receding": receding,
        }
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num}_full.jpg"), frame)
        crop = self._crop_person(frame, meta)
        if crop.size > 0:
            cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num}_person.jpg"), crop)
        with open(os.path.join(person_dir, f"frame_{frame_num}_meta.json"), "w") as f:
            json.dump(meta, f)

    def _store_sample(self, person_dir, frame_num, bbox, box_area, frame, *, receding=False):
        metas = list_meta_files(person_dir)
        if len(metas) < self.max_samples:
            self._save_sample(person_dir, frame_num, bbox, box_area, frame, receding=receding)
            return True
        smallest = min(
            metas,
            key=lambda m: read_sample_meta(os.path.join(person_dir, m)).get("area", 0),
        )
        smallest_area = read_sample_meta(os.path.join(person_dir, smallest)).get("area", 0)
        if box_area <= smallest_area:
            return False
        old_frame = frame_num_from_meta_filename(smallest)
        if old_frame:
            self._remove_sample(person_dir, old_frame)
        self._save_sample(person_dir, frame_num, bbox, box_area, frame, receding=receding)
        return True

    def _max_area_in_dir(self, person_dir):
        areas = []
        for meta_file in list_meta_files(person_dir):
            try:
                areas.append(read_sample_meta(os.path.join(person_dir, meta_file))["area"])
            except (KeyError, json.JSONDecodeError, OSError):
                pass
        return max(areas) if areas else 0

    def _flush_track(self, track_id, reason="interval"):
        staging_dir = os.path.join(self.queue_dir, f"_staging_person_{track_id}")
        person_dir = staging_dir
        sample_count = len(list_meta_files(person_dir))

        min_required = self.min_samples if reason == "interval" else 1
        if sample_count < min_required:
            if reason in ("lost", "final"):
                with self._meta_lock:
                    self.track_meta.pop(track_id, None)
                if os.path.isdir(staging_dir):
                    shutil.rmtree(staging_dir, ignore_errors=True)
            return

        with self._meta_lock:
            meta = self.track_meta.get(track_id, {"batch": 0})
            batch = meta.get("batch", 0)
            if reason == "interval":
                meta["batch"] = batch + 1
                meta["last_flush"] = time.time()
                self.track_meta[track_id] = meta
            else:
                self.track_meta.pop(track_id, None)

        ts = int(time.time())
        output_name = f"person_{track_id}_b{batch}_{ts}"
        output_dir = os.path.join(self.queue_dir, output_name)

        with self._staging_lock:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            if os.path.isdir(staging_dir):
                os.rename(staging_dir, output_dir)
            else:
                return
            os.makedirs(staging_dir, exist_ok=True)
            write_ready(output_dir)

        label = {"lost": "left frame", "interval": "interval", "final": "video end"}.get(
            reason, reason
        )
        print(f"Track {track_id} flush ({label}) -> {output_name} ({sample_count} samples)")

    def _check_interval_flushes(self, active_ids):
        now = time.time()
        for track_id in list(active_ids):
            with self._meta_lock:
                meta = self.track_meta.get(track_id)
                if meta is None:
                    continue
                if now - meta.get("last_flush", now) < self.flush_interval:
                    continue
            staging_dir = os.path.join(self.queue_dir, f"_staging_person_{track_id}")
            if len(list_meta_files(staging_dir)) < self.min_samples:
                with self._meta_lock:
                    if track_id in self.track_meta:
                        self.track_meta[track_id]["last_flush"] = now
                continue
            with self._meta_lock:
                peak_area_time = self.track_meta.get(track_id, {}).get("peak_area_time", 0)
            if now - peak_area_time < self.flush_stable:
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

        clean_frame = results.orig_img.copy()
        frame = clean_frame.copy()
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        fh, fw = frame.shape[:2]
        frame_area = fh * fw

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

            with self._meta_lock:
                if tid not in self.track_meta:
                    self.track_meta[tid] = {
                        "last_flush": now,
                        "batch": 0,
                        "peak_buffer_area": 0,
                        "peak_area_time": now,
                        "last_area": None,
                        "shrink_streak": 0,
                    }
                peak_area = self.track_meta[tid].get("peak_buffer_area", 0)

            receding = self._is_receding(tid, box_area, max(peak_area, box_area))

            staging_dir = os.path.join(self.queue_dir, f"_staging_person_{tid}")
            os.makedirs(staging_dir, exist_ok=True)
            saved = self._store_sample(
                staging_dir,
                frame_num,
                person_bbox,
                box_area,
                clean_frame,
                receding=receding,
            )

            max_area = self._max_area_in_dir(staging_dir)
            with self._meta_lock:
                meta = self.track_meta.get(tid)
                if meta and max_area > meta.get("peak_buffer_area", 0):
                    meta["peak_buffer_area"] = max_area
                    meta["peak_area_time"] = now

            if saved:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame, f"ID:{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,
                )

        results.orig_img = frame
        lost = self.active_track_ids - current_ids
        for tid in lost:
            self._flush_track(int(tid), reason="lost")
        self.active_track_ids = current_ids
        self._check_interval_flushes(current_ids)

    def _run_yolo_stream(self, source, show=True):
        frame_num = 0
        for results in self.person_model.track(source=source, **self._tracker_kwargs(show=show)):
            frame_num += 1
            self._process_frame(results, frame_num)

    def _run_rotated_capture_loop(self, rtsp_url, rotation, show=True):
        cap = open_rtsp_capture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP: {rtsp_url}")
        frame_num = 0
        try:
            while True:
                raw = read_latest_frame(cap, drain=3)
                if raw is None:
                    if not cap.isOpened():
                        raise ConnectionError("RTSP capture closed")
                    continue
                frame = apply_frame_rotation(raw, rotation)
                results_list = self.person_model.track(
                    frame, **self._tracker_kwargs(show=False, stream=False)
                )
                if not results_list:
                    continue
                result = results_list[0]
                result.orig_img = frame
                frame_num += 1
                self._process_frame(result, frame_num)
                if show:
                    cv2.imshow("Person Capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
        finally:
            cap.release()

    def _run_rtsp_stream(self, rtsp_url, rotation, show=True):
        while True:
            try:
                if rotation:
                    pipeline = build_rotated_gstreamer_pipeline(rtsp_url, rotation)
                    if gstreamer_pipeline_opens(pipeline):
                        print(f"RTSP rotation {rotation} via GStreamer")
                        self._run_yolo_stream(pipeline, show=show)
                    else:
                        print(f"RTSP rotation {rotation} via software rotate")
                        self._run_rotated_capture_loop(rtsp_url, rotation, show=show)
                else:
                    self._run_yolo_stream(rtsp_url, show=show)
                print(f"RTSP ended, reconnecting in {self.rtsp_reconnect_delay}s...")
                time.sleep(self.rtsp_reconnect_delay)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"RTSP error: {e}. Reconnecting in {self.rtsp_reconnect_delay}s...")
                time.sleep(self.rtsp_reconnect_delay)

    def _finalize_active_tracks(self):
        for name in os.listdir(self.queue_dir):
            if name.startswith("_staging_person_"):
                try:
                    track_id = int(name.replace("_staging_person_", ""))
                except ValueError:
                    continue
                self._flush_track(track_id, reason="final")
        for track_id in list(self.active_track_ids):
            self._flush_track(int(track_id), reason="final")

    def run(self, video_source, show=True, draw_roi=False):
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
            if show:
                cv2.destroyAllWindows()
            print("Person capture finished.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: person capture service")
    parser.add_argument(
        "video_source",
        nargs="?",
        default=r"test_videos\Knowns.mp4",
        help="RTSP URL, video file, webcam index",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--draw-roi",
        action="store_true",
        help="Draw ROI on first frame before capture (saves roi.json)",
    )
    args = parser.parse_args()

    if not is_valid_video_source(args.video_source):
        print(f"Invalid source: {args.video_source}")
        return

    PersonCaptureService().run(
        args.video_source, show=not args.no_show, draw_roi=args.draw_roi
    )


if __name__ == "__main__":
    main()
