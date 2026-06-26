import cv2
import numpy as np
import os
import re
import shutil
import json
from urllib.parse import urlparse, urlunparse
from ultralytics import YOLO
import torch
import onnxruntime as ort
import threading
import time

from cooldown_state import RECOGNITION_COOLDOWN_SEC

# --- tracking / quality config ---
PERSON_MODEL_PATH = "yolov8n.pt"
FACE_MODEL_PATH = "models/yolov8n-face.pt"
PERSON_CONF = 0.4              # keep >= new_track_thresh in rao_tracker.yaml
FACE_CONF = 0.5
MIN_BOX_AREA_RATIO = 0.002   # skip tiny boxes vs frame size (angle-agnostic)
MAX_SAMPLES_PER_TRACK = 15   # per track: keep the largest crops seen
FLUSH_STABLE_SEC = 4         # wait after person stops getting larger before interval flush
VID_STRIDE = 1                 # every frame — helps tracker continuity
OFIQ_SCORE_THRESHOLD = 16      # reject known bad samples (~14–15); tune via flush near-miss logs
MIN_FACE_HEIGHT_PX = 36        # min face bbox height in person crop (pixels)
MIN_FACE_AREA_RATIO = 0.008    # face area / person crop area
MIN_FACE_BLUR_VAR = 20.0       # Laplacian variance — RTSP compression keeps this low
MAX_FACE_Y_CENTER_RATIO = 0.62 # face must sit in upper portion of person crop
EXPORT_TOP_N = 5
TRACK_FLUSH_INTERVAL_SEC = 18
MIN_SAMPLES_BEFORE_FLUSH = 3
TRACKER_CONFIG = "rao_tracker.yaml"

VALID_ROTATIONS = (0, 90, 180, 270)
RTSP_RECONNECT_DELAY = 5

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
    match = re.search(r':(\d+)$', path)
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

    rtsp_url = urlunparse(parsed)
    return rtsp_url, rotation, True


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


def resolve_face_model_path():
    if os.path.isfile(FACE_MODEL_PATH):
        return FACE_MODEL_PATH
    fallback = "yolov8n-face.pt"
    if os.path.isfile(fallback):
        return fallback
    return FACE_MODEL_PATH


def face_blur_variance(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality_reject_reason(face_info):
    """Return rejection reason string, or None if face passes pre-OFIQ gates."""
    if face_info is None:
        return "no_face"
    if face_info["height"] < MIN_FACE_HEIGHT_PX:
        return f"small_face({face_info['height']}px)"
    if face_info["area_ratio"] < MIN_FACE_AREA_RATIO:
        return f"small_ratio({face_info['area_ratio']:.3f})"
    if face_info["blur_var"] < MIN_FACE_BLUR_VAR:
        return f"blurry({face_info['blur_var']:.1f})"
    if face_info["y_center_ratio"] > MAX_FACE_Y_CENTER_RATIO:
        return f"low_in_crop(y={face_info['y_center_ratio']:.2f})"
    if face_info["conf"] < FACE_CONF:
        return f"low_conf({face_info['conf']:.2f})"
    return None


class OFIQScorer:
    """Scores face quality using a pre-trained OFIQ ONNX model."""

    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda" and ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)
        self.recognition_folder = "recognition_folder"

    def _preprocess(self, face_img):
        img = cv2.resize(face_img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[np.newaxis, ...]

    def get_score(self, face_img):
        if face_img is None or face_img.size == 0:
            return 0.0
        try:
            output = self.session.run(None, {self.input_name: self._preprocess(face_img)})
            return float(output[0].item())
        except Exception as e:
            print(f"Error scoring image: {e}")
            return 0.0


class LiveFaceProcessor:
    """Track persons, save body crops, score faces with OFIQ, export for recognition."""

    def __init__(
        self,
        person_model_path=PERSON_MODEL_PATH,
        face_model_path=None,
        ofiq_model_path="OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx",
        save_dir="live_results",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        face_path = face_model_path or resolve_face_model_path()
        self.person_model = YOLO(person_model_path)
        self.face_model = YOLO(face_path)
        self.scorer = OFIQScorer(ofiq_model_path, device=self.device)
        self.save_dir = save_dir
        self.active_track_ids = set()
        self.track_meta = {}
        self._session_export_until = {}
        self.stream_url = None
        self._meta_lock = threading.Lock()

        print(f"Person tracker: {person_model_path}")
        print(f"ByteTrack config: {TRACKER_CONFIG}")
        print(f"Face model (post-process): {face_path}")
        print(
            f"Quality gates: OFIQ>{OFIQ_SCORE_THRESHOLD}, face_h>={MIN_FACE_HEIGHT_PX}px, "
            f"blur>={MIN_FACE_BLUR_VAR}, area>={MIN_FACE_AREA_RATIO:.0%}"
        )

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def _tracker_kwargs(self, show=True, stream=True):
        return dict(
            stream=stream,
            show=show,
            vid_stride=VID_STRIDE,
            persist=True,
            device=self.device,
            tracker=TRACKER_CONFIG,
            classes=[0],
            conf=PERSON_CONF,
            iou=0.5,
            verbose=False,
        )

    @staticmethod
    def _clamp_bbox(x1, y1, x2, y2, frame_w, frame_h):
        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(x1 + 1, min(int(x2), frame_w))
        y2 = max(y1 + 1, min(int(y2), frame_h))
        return x1, y1, x2, y2

    @staticmethod
    def _frame_num_from_meta_filename(filename):
        match = re.match(r"frame_(\d+)_meta\.json", filename)
        return match.group(1) if match else None

    @staticmethod
    def _list_sample_meta_files(person_dir):
        if not os.path.isdir(person_dir):
            return []
        return sorted(f for f in os.listdir(person_dir) if f.endswith("_meta.json"))

    @staticmethod
    def _read_sample_meta(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _max_sample_area_in_dir(person_dir):
        areas = []
        for meta_file in LiveFaceProcessor._list_sample_meta_files(person_dir):
            try:
                areas.append(LiveFaceProcessor._read_sample_meta(os.path.join(person_dir, meta_file))["area"])
            except (KeyError, json.JSONDecodeError, OSError):
                pass
        return max(areas) if areas else 0

    def _remove_sample(self, person_dir, frame_num):
        for name in (
            f"frame_{frame_num}_meta.json",
            f"frame_{frame_num}_full.jpg",
            f"frame_{frame_num}_person.jpg",
        ):
            path = os.path.join(person_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    def _save_person_sample(self, person_dir, frame_num, bbox, box_area, frame):
        meta_path = os.path.join(person_dir, f"frame_{frame_num}_meta.json")
        full_path = os.path.join(person_dir, f"frame_{frame_num}_full.jpg")
        person_path = os.path.join(person_dir, f"frame_{frame_num}_person.jpg")
        meta = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "area": box_area}
        cv2.imwrite(full_path, frame)
        person_crop = self._crop_person_from_full(frame, meta)
        if person_crop.size > 0:
            cv2.imwrite(person_path, person_crop)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _store_person_sample(self, person_dir, frame_num, bbox, box_area, frame):
        """Keep full frame + bbox meta; retain the largest-area samples when buffer is full."""
        metas = self._list_sample_meta_files(person_dir)
        if len(metas) < MAX_SAMPLES_PER_TRACK:
            self._save_person_sample(person_dir, frame_num, bbox, box_area, frame)
            return True

        smallest_meta = min(
            metas,
            key=lambda m: self._read_sample_meta(os.path.join(person_dir, m)).get("area", 0),
        )
        smallest_area = self._read_sample_meta(os.path.join(person_dir, smallest_meta)).get("area", 0)
        if box_area <= smallest_area:
            return False

        old_frame = self._frame_num_from_meta_filename(smallest_meta)
        if old_frame:
            self._remove_sample(person_dir, old_frame)
        self._save_person_sample(person_dir, frame_num, bbox, box_area, frame)
        return True

    def _crop_person_from_full(self, full_frame, meta):
        h, w = full_frame.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(
            meta["x1"], meta["y1"], meta["x2"], meta["y2"], w, h
        )
        return full_frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _clear_directory(path):
        if not os.path.isdir(path):
            return
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                os.remove(fp)

    def _find_best_face_in_person(self, person_img):
        results = self.face_model.predict(person_img, conf=FACE_CONF, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        person_h, person_w = person_img.shape[:2]
        person_area = person_w * person_h
        best = None
        best_area = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(person_w, x2), min(person_h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw <= 0 or fh <= 0:
                continue

            area = fw * fh
            crop = person_img[y1:y2, x1:x2].copy()
            info = {
                "crop": crop,
                "conf": float(box.conf[0]),
                "height": fh,
                "area_ratio": area / person_area,
                "y_center_ratio": ((y1 + y2) / 2) / person_h,
                "blur_var": face_blur_variance(crop),
            }
            if area > best_area:
                best_area = area
                best = info
        return best

    def _flush_track(self, track_id, reason="interval"):
        person_dir = os.path.join(self.save_dir, f"person_{track_id}")
        sample_count = len(self._list_sample_meta_files(person_dir))

        if sample_count < MIN_SAMPLES_BEFORE_FLUSH:
            if reason == "lost":
                with self._meta_lock:
                    self.track_meta.pop(track_id, None)
                self._session_export_until.pop(track_id, None)
                self._clear_directory(person_dir)
            return

        if reason == "interval" and time.time() < self._session_export_until.get(track_id, 0):
            with self._meta_lock:
                if track_id in self.track_meta:
                    self.track_meta[track_id]["last_flush"] = time.time()
            return

        with self._meta_lock:
            meta = self.track_meta.get(track_id, {"batch": 0, "last_flush": time.time()})
            batch = meta.get("batch", 0)
            if reason == "interval":
                meta["batch"] = batch + 1
                meta["last_flush"] = time.time()
                self.track_meta[track_id] = meta
            else:
                self.track_meta.pop(track_id, None)
                self._session_export_until.pop(track_id, None)

        ts = int(time.time())
        staging_dir = f"{person_dir}_staging_b{batch}_{ts}"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        if os.path.isdir(person_dir):
            os.rename(person_dir, staging_dir)
        os.makedirs(person_dir, exist_ok=True)

        output_folder = f"person_{track_id}_b{batch}_{ts}"
        label = "left frame" if reason == "lost" else "interval"
        print(f"Track {track_id} flush ({label}) -> {output_folder} ({sample_count} samples)")

        thread = threading.Thread(
            target=self._score_and_export_staging,
            args=(staging_dir, output_folder, track_id),
            daemon=True,
        )
        thread.start()

    def _score_and_export_staging(self, staging_dir, output_folder, track_id):
        try:
            meta_files = self._list_sample_meta_files(staging_dir)
            meta_files.sort(
                key=lambda m: self._read_sample_meta(os.path.join(staging_dir, m)).get("area", 0),
                reverse=True,
            )
            if not meta_files:
                return

            recognition_dir = os.path.join(self.scorer.recognition_folder, output_folder)
            os.makedirs(recognition_dir, exist_ok=True)
            scored_candidates = []
            skipped_no_face = 0
            skipped_quality = 0
            skipped_ofiq = 0
            near_misses = []

            for meta_file in meta_files:
                frame_num = self._frame_num_from_meta_filename(meta_file)
                if not frame_num:
                    continue

                meta_path = os.path.join(staging_dir, meta_file)
                full_path = os.path.join(staging_dir, f"frame_{frame_num}_full.jpg")
                full_frame = cv2.imread(full_path)
                if full_frame is None:
                    continue

                try:
                    meta = self._read_sample_meta(meta_path)
                except (json.JSONDecodeError, OSError):
                    continue

                person_crop = self._crop_person_from_full(full_frame, meta)
                if person_crop.size == 0:
                    continue

                face_info = self._find_best_face_in_person(person_crop)
                reject = face_quality_reject_reason(face_info)
                if reject:
                    if reject == "no_face":
                        skipped_no_face += 1
                    else:
                        skipped_quality += 1
                        near_misses.append((0.0, frame_num, reject, face_info))
                    continue

                score = self.scorer.get_score(face_info["crop"])
                if score <= OFIQ_SCORE_THRESHOLD:
                    skipped_ofiq += 1
                    near_misses.append((score, frame_num, "low_ofiq", face_info))
                    continue

                person_path = os.path.join(staging_dir, f"frame_{frame_num}_person.jpg")
                scored_candidates.append((score, frame_num, full_path, person_path))
                time.sleep(0.01)

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            to_export = scored_candidates[:EXPORT_TOP_N]

            exported = 0
            for score, frame_num, full_path, person_path in to_export:
                person_scored = f"frame_{frame_num}_person_score_{score:.2f}.jpg"
                full_scored = f"frame_{frame_num}_full_score_{score:.2f}.jpg"
                if os.path.isfile(person_path):
                    shutil.copy2(person_path, os.path.join(recognition_dir, person_scored))
                elif os.path.isfile(full_path):
                    shutil.copy2(full_path, os.path.join(recognition_dir, person_scored))
                if os.path.isfile(full_path):
                    shutil.copy2(full_path, os.path.join(recognition_dir, full_scored))
                exported += 1

            if exported > 0:
                ready_file = os.path.join(recognition_dir, ".ready")
                with open(ready_file, "w") as f:
                    f.write("ready")
                self._session_export_until[track_id] = time.time() + RECOGNITION_COOLDOWN_SEC
                print(
                    f"Track {track_id}: exported {exported}/{len(scored_candidates)} "
                    f"frame(s) to {recognition_dir}"
                )
            else:
                shutil.rmtree(recognition_dir, ignore_errors=True)
                print(
                    f"Track {track_id}: no export for {output_folder} "
                    f"(no_face={skipped_no_face}, quality={skipped_quality}, "
                    f"low_ofiq={skipped_ofiq}, passed={len(scored_candidates)})"
                )
                near_misses.sort(key=lambda x: x[0], reverse=True)
                for score, frame_num, reason, info in near_misses[:3]:
                    print(
                        f"  near-miss frame {frame_num}: {reason} "
                        f"OFIQ={score:.2f} h={info['height']}px "
                        f"blur={info['blur_var']:.1f} area={info['area_ratio']:.3f}"
                    )
        except Exception as e:
            print(f"Error scoring staging dir {staging_dir}: {e}")
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _check_interval_flushes(self, active_ids):
        now = time.time()
        for track_id in list(active_ids):
            with self._meta_lock:
                meta = self.track_meta.get(track_id)
                if meta is None:
                    continue
                if now - meta.get("last_flush", now) < TRACK_FLUSH_INTERVAL_SEC:
                    continue

            person_dir = os.path.join(self.save_dir, f"person_{track_id}")
            if len(self._list_sample_meta_files(person_dir)) < MIN_SAMPLES_BEFORE_FLUSH:
                with self._meta_lock:
                    if track_id in self.track_meta:
                        self.track_meta[track_id]["last_flush"] = now
                continue

            with self._meta_lock:
                meta = self.track_meta.get(track_id, {})
                peak_area_time = meta.get("peak_area_time", 0)

            if now - peak_area_time < FLUSH_STABLE_SEC:
                continue

            self._flush_track(track_id, reason="interval")

    def _process_tracking_frame(self, results, frame_num):
        current_frame_track_ids = set()
        now = time.time()

        if results.boxes is not None and results.boxes.id is not None:
            frame = results.orig_img.copy()
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            for i, track_id in enumerate(track_ids):
                tid = int(track_id)
                current_frame_track_ids.add(tid)
                x1, y1, x2, y2 = self._clamp_bbox(
                    boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], frame_w, frame_h
                )
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < frame_area * MIN_BOX_AREA_RATIO:
                    continue

                with self._meta_lock:
                    if tid not in self.track_meta:
                        self.track_meta[tid] = {
                            "last_flush": now,
                            "batch": 0,
                            "peak_buffer_area": 0,
                            "peak_area_time": now,
                        }

                person_dir = os.path.join(self.save_dir, f"person_{tid}")
                os.makedirs(person_dir, exist_ok=True)

                saved = self._store_person_sample(
                    person_dir, frame_num, (x1, y1, x2, y2), box_area, frame
                )

                max_area = self._max_sample_area_in_dir(person_dir)
                with self._meta_lock:
                    meta = self.track_meta.get(tid)
                    if meta and max_area > meta.get("peak_buffer_area", 0):
                        meta["peak_buffer_area"] = max_area
                        meta["peak_area_time"] = now

                if not saved:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    2,
                )

            results.orig_img = frame

        lost_track_ids = self.active_track_ids - current_frame_track_ids
        for track_id in lost_track_ids:
            self._flush_track(int(track_id), reason="lost")

        self.active_track_ids = current_frame_track_ids
        self._check_interval_flushes(current_frame_track_ids)

    def _run_yolo_stream(self, source, show=True):
        frame_num = 0
        results_generator = self.person_model.track(source=source, **self._tracker_kwargs(show=show))
        for results in results_generator:
            frame_num += 1
            self._process_tracking_frame(results, frame_num)

    def _run_rotated_capture_loop(self, rtsp_url, rotation, show=True):
        cap = open_rtsp_capture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

        frame_num = 0
        try:
            while True:
                raw = read_latest_frame(cap, drain=3)
                if raw is None:
                    if not cap.isOpened():
                        raise ConnectionError("RTSP capture closed")
                    continue

                frame = apply_frame_rotation(raw, rotation)
                results_list = self.person_model.track(frame, **self._tracker_kwargs(show=False, stream=False))
                if not results_list:
                    continue

                result = results_list[0]
                result.orig_img = frame
                frame_num += 1
                self._process_tracking_frame(result, frame_num)

                if show:
                    cv2.imshow("Live Face Processor", frame)
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
                        print(f"RTSP rotation {rotation}° via GStreamer")
                        self._run_yolo_stream(pipeline, show=show)
                    else:
                        print(f"RTSP rotation {rotation}° via software rotate (GStreamer unavailable)")
                        self._run_rotated_capture_loop(rtsp_url, rotation, show=show)
                else:
                    print(f"RTSP stream (no rotation): {rtsp_url}")
                    self._run_yolo_stream(rtsp_url, show=show)

                print(f"RTSP stream ended, reconnecting in {RTSP_RECONNECT_DELAY}s...")
                time.sleep(RTSP_RECONNECT_DELAY)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"RTSP error: {e}. Reconnecting in {RTSP_RECONNECT_DELAY}s...")
                time.sleep(RTSP_RECONNECT_DELAY)

    def _finalize_active_tracks(self):
        for track_id in list(self.active_track_ids):
            self._flush_track(int(track_id), reason="lost")

    def process_video(self, video_source, show=True):
        play_source, rotation, is_rtsp = parse_video_source(video_source)
        self.stream_url = play_source if is_rtsp else str(video_source)

        try:
            if is_rtsp:
                print(f"Live RTSP mode — url={play_source}, rotation={rotation}° (Ctrl+C to stop)")
                self._run_rtsp_stream(play_source, rotation, show=show)
            else:
                source = play_source if play_source is not None else video_source
                print(f"Processing source: {source}")
                self._run_yolo_stream(source, show=show)
        except KeyboardInterrupt:
            print("\nProcessing stopped.")
        finally:
            if is_rtsp:
                print("RTSP stopped. Flushing remaining active tracks...")
            else:
                print("Video finished. Processing remaining tracks...")
            self._finalize_active_tracks()
            if show:
                cv2.destroyAllWindows()
            print("All processing complete.")


if __name__ == "__main__":
    video_source = r"C:\Users\raoit\Work\Face_Recognition_College_OFIQ_RANKER\test_videos\Knowns.mp4"
    # video_source = 0
    # video_source = "rtsp://office:office123@192.170.1.50/stream1:180"

    if not is_valid_video_source(video_source):
        print(f"Error: Invalid video source: {video_source}")
    else:
        play_source, rotation, is_rtsp = parse_video_source(video_source)
        if is_rtsp:
            print(f"RTSP URL: {play_source} | rotation: {rotation}°")
        processor = LiveFaceProcessor()
        processor.process_video(video_source)
        if not is_rtsp:
            print(f"\nLive processing finished. Results are in '{os.path.abspath(processor.save_dir)}'.")
