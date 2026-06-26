"""
Compare person vs face tracking ID stability on a video file.

Edit the config variables below, then run:
    python verify_person_tracking.py

Person mode uses YOLO COCO class 0 + ByteTrack (proposed live pipeline).
Face mode uses the face YOLO model (current live_face_processor behavior).
"""

import os
import sys
from collections import defaultdict

import cv2
import torch
from ultralytics import YOLO

# --- config: edit these ---
VIDEO_PATH = r"C:\Users\raoit\Work\Face_Recognition_College_OFIQ_RANKER\test_videos\Knowns.mp4"
MODE = "both"  # "person", "face", or "both"
SAVE_PATH = None  # e.g. "tracking_compare.mp4" or None to skip saving
CONF = 0.5
VID_STRIDE = 1
SHOW = True
# --- end config ---

TRACKER_CONFIG = "rao_tracker.yaml"
PERSON_MODEL = "yolov8n.pt"
FACE_MODEL = "models/yolov8n-face.pt"

PERSON_COLOR = (0, 200, 0)
FACE_COLOR = (0, 0, 255)


def track_video(model, video_path, *, classes=None, label, color, conf, vid_stride, show, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track_kwargs = dict(
        source=video_path,
        stream=True,
        persist=True,
        device=device,
        tracker=TRACKER_CONFIG,
        conf=conf,
        vid_stride=vid_stride,
        verbose=False,
    )
    if classes is not None:
        track_kwargs["classes"] = classes

    frame_counts = defaultdict(int)
    frame_num = 0
    writer = None

    print(f"\n--- {label} tracking ---")
    print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'yolo'} | device: {device}")

    for results in model.track(**track_kwargs):
        frame_num += 1
        frame = results.orig_img

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                tid = int(track_id)
                frame_counts[tid] += 1

                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{tid}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        header = f"{label} | frame {frame_num} | active IDs: {len(frame_counts)}"
        cv2.putText(frame, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if save_path:
            if writer is None:
                h, w = frame.shape[:2]
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
                )
            writer.write(frame)

        if show:
            cv2.imshow(f"Tracking verify — {label}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early (q pressed).")
                break

    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return summarize(label, frame_num, frame_counts)


def summarize(label, total_frames, frame_counts):
    unique_ids = len(frame_counts)
    durations = list(frame_counts.values())
    mean_dur = sum(durations) / unique_ids if unique_ids else 0
    short_tracks = sum(1 for d in durations if d <= 3)

    stats = {
        "label": label,
        "total_frames": total_frames,
        "unique_ids": unique_ids,
        "mean_track_frames": mean_dur,
        "short_tracks_le_3": short_tracks,
        "frame_counts": dict(frame_counts),
    }

    print(f"  Frames processed:     {total_frames}")
    print(f"  Unique track IDs:     {unique_ids}")
    print(f"  Mean track length:    {mean_dur:.1f} frames")
    print(f"  Short tracks (<=3f):  {short_tracks}  (high = ID churn)")
    if frame_counts:
        top = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Longest tracks:       {top}")

    return stats


def print_comparison(person_stats, face_stats):
    print("\n" + "=" * 60)
    print("COMPARISON (lower unique IDs + fewer short tracks = more stable)")
    print("=" * 60)
    rows = [
        ("Unique IDs", person_stats["unique_ids"], face_stats["unique_ids"]),
        ("Mean track length (frames)", f"{person_stats['mean_track_frames']:.1f}", f"{face_stats['mean_track_frames']:.1f}"),
        ("Short tracks (<=3 frames)", person_stats["short_tracks_le_3"], face_stats["short_tracks_le_3"]),
    ]
    print(f"{'Metric':<28} {'Person':>12} {'Face':>12}")
    print("-" * 60)
    for name, p, f in rows:
        print(f"{name:<28} {p:>12} {f:>12}")

    if person_stats["unique_ids"] < face_stats["unique_ids"]:
        print("\nPerson tracking produced FEWER unique IDs — likely more stable for this clip.")
    elif person_stats["unique_ids"] > face_stats["unique_ids"]:
        print("\nFace tracking produced fewer unique IDs on this clip (unusual — check footage).")
    else:
        print("\nSame number of unique IDs — check mean track length and short-track counts.")


def resolve_face_model():
    if os.path.isfile(FACE_MODEL):
        return FACE_MODEL
    fallback = "yolov8n-face.pt"
    if os.path.isfile(fallback):
        return fallback
    return FACE_MODEL


def resolve_save_path(mode, save_path):
    if not save_path:
        return None
    if mode == "person":
        return save_path
    if mode == "face":
        return save_path
    if save_path.endswith(".mp4"):
        return save_path.replace(".mp4", "_person.mp4")
    return f"{save_path}_person.mp4"


def resolve_face_save_path(save_path):
    if not save_path:
        return None
    if save_path.endswith(".mp4"):
        return save_path.replace(".mp4", "_face.mp4")
    return f"{save_path}_face.mp4"


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"Error: video not found: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(TRACKER_CONFIG):
        print(f"Warning: {TRACKER_CONFIG} not found; using Ultralytics default tracker.")

    person_stats = face_stats = None

    if MODE in ("person", "both"):
        person_model = YOLO(PERSON_MODEL)
        save = resolve_save_path("person" if MODE == "person" else "both", SAVE_PATH)
        person_stats = track_video(
            person_model,
            VIDEO_PATH,
            classes=[0],
            label="PERSON",
            color=PERSON_COLOR,
            conf=CONF,
            vid_stride=VID_STRIDE,
            show=SHOW,
            save_path=save,
        )

    if MODE in ("face", "both"):
        face_path = resolve_face_model()
        if not os.path.isfile(face_path):
            print(f"Warning: face model not found at {face_path}; download or place weights first.")
        face_model = YOLO(face_path)
        save = SAVE_PATH if MODE == "face" else resolve_face_save_path(SAVE_PATH)
        face_stats = track_video(
            face_model,
            VIDEO_PATH,
            classes=None,
            label="FACE",
            color=FACE_COLOR,
            conf=CONF,
            vid_stride=VID_STRIDE,
            show=SHOW,
            save_path=save,
        )

    if person_stats and face_stats:
        print_comparison(person_stats, face_stats)

    print("\nDone. Watch the overlay: IDs should stay constant per person while they are on screen.")
    print("Green = PERSON (proposed) | Red = FACE (current)")


if __name__ == "__main__":
    main()
