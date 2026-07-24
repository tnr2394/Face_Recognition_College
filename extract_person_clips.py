"""
Build ~5-minute person-only clips from a folder of videos.

Writes IMMEDIATELY when a person is detected (output file appears right away).
Empty frames are skipped. One clip can span many source videos.

Edit settings, then run:
    python extract_person_clips.py
"""

from pathlib import Path
import shutil

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Settings (edit these)
# ---------------------------------------------------------------------------
INPUT_FOLDER = r"D:\Tapo camera videos"
OUTPUT_FOLDER = r"D:\Person clips"
PROCESSED_FOLDER = r"D:\Tapo camera videos\processed"

PROCESS_FPS = 5  # how often to run person detection
SOURCE_FPS_FALLBACK = 25.0

CLIP_MINUTES = 5
CLIP_TOLERANCE_SECONDS = 15  # finished clips are ~5 min ± 15s
CONF = 0.25
MODEL = "yolov8n.pt"
DEVICE = 0  # or "cpu"

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"}

CLIP_TARGET = CLIP_MINUTES * 60
CLIP_MIN = CLIP_TARGET - CLIP_TOLERANCE_SECONDS
# ---------------------------------------------------------------------------


class LivePersonClipWriter:
    """Opens an output mp4 on first person frame and keeps appending."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.clip_no = 1
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.frames = 0
        self.fps = SOURCE_FPS_FALLBACK
        self.size: tuple[int, int] | None = None
        self.total_clips = 0

    def _target_frames(self) -> int:
        return int(CLIP_TARGET * self.fps)

    def _open(self, frame, fps: float) -> None:
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.fps = fps if fps > 1e-3 else SOURCE_FPS_FALLBACK
        self.path = self.out_dir / f"person_clip_{self.clip_no:04d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        self.frames = 0
        tqdm.write(f"  opened {self.path.name} (writing person frames live)")

    def write(self, frame, fps: float) -> None:
        if self.writer is None:
            self._open(frame, fps)

        assert self.size is not None and self.writer is not None
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        self.writer.write(frame)
        self.frames += 1

        # Rotate to a new file as soon as this clip reaches ~5 minutes
        if self.frames >= self._target_frames():
            self._finish(ok=True)

    def _finish(self, ok: bool) -> None:
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        secs = self.frames / self.fps if self.fps > 0 else 0
        if ok and self.path is not None:
            final = self.path.with_name(
                f"person_clip_{self.clip_no:04d}_{secs:.0f}s.mp4"
            )
            if final != self.path:
                self.path.replace(final)
            tqdm.write(f"  finished {final.name} ({secs:.0f}s, {self.frames} frames)")
            self.total_clips += 1
            self.clip_no += 1
        elif self.path is not None and self.path.exists():
            # Too short — remove incomplete clip
            self.path.unlink(missing_ok=True)
            tqdm.write(f"  removed short incomplete clip ({secs:.0f}s)")
        self.path = None
        self.frames = 0

    def close_end(self) -> None:
        """End of all videos: keep clip only if length is in range."""
        if self.writer is None:
            return
        secs = self.frames / self.fps if self.fps > 0 else 0
        self._finish(ok=(secs >= CLIP_MIN))


def get_source_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 1e-3 or fps > 240:
        return SOURCE_FPS_FALLBACK
    return fps


def detection_stride(source_fps: float) -> int:
    return max(1, int(round(source_fps / PROCESS_FPS)))


def has_person(model: YOLO, frame) -> bool:
    results = model.predict(
        frame, conf=CONF, classes=[0], verbose=False, device=DEVICE
    )
    boxes = results[0].boxes
    return boxes is not None and len(boxes) > 0


def move_to_processed(video_path: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / video_path.name
    if dest.exists():
        dest = processed_dir / f"{video_path.stem}_done{video_path.suffix}"
    shutil.move(str(video_path), str(dest))
    print(f"  moved source -> {dest}", flush=True)


def scan_and_write(model: YOLO, video_path: Path, clipper: LivePersonClipWriter) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  SKIP (cannot open): {video_path.name}")
        return False

    fps = get_source_fps(cap)
    stride = detection_stride(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(
        f"  source_fps={fps:.2f}, process_fps={PROCESS_FPS}, stride={stride}, "
        f"frames={total_frames}",
        flush=True,
    )

    frame_idx = 0
    hits = 0
    written = 0
    in_person = False

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="  scanning",
        unit="frame",
        leave=True,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run detector every stride frames; reuse result for frames in between
        if frame_idx % stride == 0:
            in_person = has_person(model, frame)
            if in_person:
                hits += 1

        if in_person:
            clipper.write(frame, fps)
            written += 1

        frame_idx += 1
        pbar.set_postfix(hits=hits, written=written, clip=clipper.clip_no)
        pbar.update(1)

    pbar.close()
    cap.release()
    print(
        f"  done video: hits={hits}, frames written to clips={written}, "
        f"current clip frames={clipper.frames}",
        flush=True,
    )
    return True


def main() -> None:
    in_dir = Path(INPUT_FOLDER)
    out_dir = Path(OUTPUT_FOLDER)
    processed_dir = Path(PROCESSED_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"No videos found in {in_dir}")
        return

    print(f"Loading model {MODEL} ...")
    model = YOLO(MODEL)
    clipper = LivePersonClipWriter(out_dir)

    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video.name}")
        ok = scan_and_write(model, video, clipper)
        if not ok:
            print("  left in input folder (could not open)", flush=True)
            continue
        move_to_processed(video, processed_dir)

    clipper.close_end()
    print(f"Done. Finished clips={clipper.total_clips} in {out_dir}")
    print(f"Processed sources are in {processed_dir}")


if __name__ == "__main__":
    main()
