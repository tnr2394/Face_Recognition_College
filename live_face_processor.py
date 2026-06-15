import cv2
import numpy as np
import os
import re
import shutil
from urllib.parse import urlparse, urlunparse
from ultralytics import YOLO
from collections import defaultdict
import torch
import onnxruntime as ort
import threading
import time

# Live Face Processor Added

OFIQ_SCORE_THRESHOLD = 15  # Min OFIQ score to move into recognition_folder (try 20–25 for stricter)
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
    """If netloc port is 0/90/180/270, treat it as rotation (not an RTSP port)."""
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
    """If path ends with :0/:90/:180/:270, treat it as rotation (e.g. /stream1:270)."""
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
    """
    Parse a video source.

    RTSP rotation can be encoded as:
      1. Port on host:     rtsp://camera-host:180/path
      2. Suffix on path:   rtsp://user:pass@host/stream1:270

    Values 0, 90, 180, 270 mean rotation degrees (0 = no rotate).
    Real RTSP ports like 554 are left unchanged.
    """
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

class OFIQScorer:
    """Scores face quality using a pre-trained OFIQ ONNX model."""
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        providers = ['CUDAExecutionProvider'] if device == 'cuda' and ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)
        self.recognition_folder = 'recognition_folder'

    def _preprocess(self, face_img):
        """Preprocesses a single face image (numpy array) for the ONNX model."""
        img = cv2.resize(face_img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        return img

    def get_score(self, face_img):
        """Calculates the quality score for a single face image."""
        if face_img is None or face_img.size == 0:
            return 0.0
        try:
            preprocessed_img = self._preprocess(face_img)
            output = self.session.run(None, {self.input_name: preprocessed_img})
            score = output[0].item()
            return score
        except Exception as e:
            print(f"Error scoring image: {e}")
            return 0.0

class LiveFaceProcessor:
    """Processes a video in real-time to track, save, and score faces."""

    def __init__(self, yolo_model_path='models/yolov8n-face.pt', ofiq_model_path='OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx', save_dir='live_results'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.yolo_model = YOLO(yolo_model_path)
        self.scorer = OFIQScorer(ofiq_model_path, device=self.device)
        self.save_dir = save_dir
        self.tracked_faces = defaultdict(list)
        self.active_track_ids = set()
        self.stream_url = None
        print(f"OFIQ recognition threshold: > {OFIQ_SCORE_THRESHOLD}")

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def _process_lost_tracks(self, lost_track_ids):
        """Process folders of tracks that are no longer active."""
        for track_id in lost_track_ids:
            print(f"Person with ID {track_id} has left. Processing their folder.")
            person_dir = os.path.join(self.save_dir, f"person_{track_id}")
            # Run processing in a separate thread to avoid blocking the main loop
            thread = threading.Thread(target=self._score_and_rename_faces, args=(person_dir,))
            thread.start()

    def _score_and_rename_faces(self, person_dir):
        """Scores all faces in a directory and renames them with the score."""
        if not os.path.isdir(person_dir):
            return

        # Get only cropped face images (not full frames) for scoring
        face_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png')) and '_full' not in f]
        
        if not face_files:
            return

        recognition_person_dir = None
        for filename in face_files:
            img_path = os.path.join(person_dir, filename)
            face_img = cv2.imread(img_path)
            if face_img is not None:
                score = self.scorer.get_score(face_img)
                new_filename = f"{os.path.splitext(filename)[0]}_score_{score:.2f}.jpg"
                
                # Check if corresponding full frame exists
                frame_num = filename.replace('frame_', '').replace('.jpg', '')
                full_frame_filename = f"frame_{frame_num}_full.jpg"
                full_frame_path = os.path.join(person_dir, full_frame_filename)
                
                try:
                    new_path = os.path.join(person_dir, new_filename)
                    if score > OFIQ_SCORE_THRESHOLD:
                        person_folder_name = os.path.basename(person_dir)
                        recognition_person_dir = os.path.join(self.scorer.recognition_folder, person_folder_name)
                        os.makedirs(recognition_person_dir, exist_ok=True)

                        rec_face_path = os.path.join(recognition_person_dir, new_filename)
                        shutil.move(img_path, rec_face_path)

                        if os.path.exists(full_frame_path):
                            full_frame_new_filename = f"frame_{frame_num}_full_score_{score:.2f}.jpg"
                            rec_full_frame_path = os.path.join(recognition_person_dir, full_frame_new_filename)
                            shutil.move(full_frame_path, rec_full_frame_path)
                            print(f"Moved {filename} and {full_frame_filename} to recognition folder with score {score:.2f}")
                        else:
                            print(f"Moved {filename} to recognition folder as {new_filename}")
                    else:
                        os.rename(img_path, new_path)
                        if os.path.exists(full_frame_path):
                            full_frame_new_filename = f"frame_{frame_num}_full_score_{score:.2f}.jpg"
                            full_frame_new_path = os.path.join(person_dir, full_frame_new_filename)
                            os.rename(full_frame_path, full_frame_new_path)
                        print(f"Below threshold ({score:.2f} <= {OFIQ_SCORE_THRESHOLD}), kept in live_results: {filename}")
                except OSError as e:
                    print(f"Error processing file {img_path}: {e}")
            time.sleep(0.01) # Small delay to yield CPU

        if recognition_person_dir:
            try:
                ready_file = os.path.join(recognition_person_dir, '.ready')
                with open(ready_file, 'w') as f:
                    f.write('ready')
            except Exception as e:
                print(f"Error creating .ready file in {recognition_person_dir}: {e}")

    def _tracker_stream_kwargs(self, show=True):
        return dict(
            stream=True,
            show=show,
            vid_stride=4,
            persist=True,
            device=self.device,
            tracker='rao_tracker.yaml',
        )

    def _process_tracking_frame(self, results, frame_num):
        current_frame_track_ids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            current_frame_track_ids.update(track_ids)

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]
                face_img = results.orig_img[y1:y2, x1:x2]
                cv2.rectangle(results.orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    results.orig_img, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )

                person_dir = os.path.join(self.save_dir, f"person_{track_id}")
                os.makedirs(person_dir, exist_ok=True)

                face_filename = f"frame_{frame_num}.jpg"
                cv2.imwrite(os.path.join(person_dir, face_filename), face_img)

                frame_filename = f"frame_{frame_num}_full.jpg"
                cv2.imwrite(os.path.join(person_dir, frame_filename), results.orig_img)

        lost_track_ids = self.active_track_ids - current_frame_track_ids
        if lost_track_ids:
            self._process_lost_tracks(lost_track_ids)

        self.active_track_ids = current_frame_track_ids

    def _run_yolo_stream(self, source, show=True):
        frame_num = 0
        results_generator = self.yolo_model.track(source=source, **self._tracker_stream_kwargs(show))
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
                results_list = self.yolo_model.track(
                    frame,
                    persist=True,
                    show=False,
                    device=self.device,
                    tracker='rao_tracker.yaml',
                    verbose=False,
                )
                if not results_list:
                    continue

                result = results_list[0]
                result.orig_img = frame
                frame_num += 1
                self._process_tracking_frame(result, frame_num)

                if show:
                    cv2.imshow("Live Face Processor", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
        finally:
            cap.release()

    def _run_rtsp_stream(self, rtsp_url, rotation, show=True):
        """Keep processing RTSP forever; reconnect when the stream drops."""
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
        if self.active_track_ids:
            self._process_lost_tracks(self.active_track_ids)

    def process_video(self, video_source, show=True):
        """Process a video file, webcam, or live RTSP stream."""
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

if __name__ == '__main__':
    # Video file, webcam (0), or RTSP with rotation:
    #   rtsp://office:office123@192.170.1.50/stream1:270   (rotation on path)
    #   rtsp://camera-host:180/stream1                     (rotation on host port)
    video_source = r'C:\Users\raoit\Work\Face_Recognition_College_OFIQ_RANKER\test_videos\Knowns.mp4'
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
