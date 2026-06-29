"""Draw a rectangular ROI on the first video frame and save to roi.json."""

import argparse
import os

import cv2

from pipeline_io import load_config, save_roi
from person_capture_service import (
    apply_frame_rotation,
    is_valid_video_source,
    open_rtsp_capture,
    parse_video_source,
    read_latest_frame,
)


def grab_first_frame(video_source, timeout_reads=60):
    play_source, rotation, is_rtsp = parse_video_source(video_source)

    if is_rtsp:
        cap = open_rtsp_capture(play_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP: {play_source}")
        try:
            for _ in range(timeout_reads):
                raw = read_latest_frame(cap, drain=1)
                if raw is not None:
                    return apply_frame_rotation(raw, rotation)
            raise RuntimeError("No frame received from RTSP")
        finally:
            cap.release()

    source = play_source if play_source is not None else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    try:
        for _ in range(timeout_reads):
            ret, frame = cap.read()
            if ret and frame is not None:
                return apply_frame_rotation(frame, rotation)
        raise RuntimeError("No frame received from video source")
    finally:
        cap.release()


def _normalize_drag_rect(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_roi_interactive(frame, window_name="Draw ROI"):
    """
    Drag a rectangle with the mouse.
    Enter = save, r = reset, q = quit without saving.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame")

    base = frame.copy()
    rect = {"pts": None}
    dragging = {"active": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rect["pts"] = (x, y, x, y)
            dragging["active"] = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging["active"] and rect["pts"]:
            x1, y1, _, _ = rect["pts"]
            rect["pts"] = (x1, y1, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging["active"] = False
            if rect["pts"]:
                x1, y1, _, _ = rect["pts"]
                rect["pts"] = (x1, y1, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Drag a rectangle. Enter = save, r = reset, q = quit.")

    while True:
        display = base.copy()
        if rect["pts"]:
            x1, y1, x2, y2 = _normalize_drag_rect(rect["pts"][:2], rect["pts"][2:])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                display,
                "ROI",
                (x1 + 4, max(y1 + 18, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(16) & 0xFF
        if key in (13, 10):
            break
        if key == ord("r"):
            rect["pts"] = None
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    if not rect["pts"]:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _normalize_drag_rect(rect["pts"][:2], rect["pts"][2:])
    if x2 - x1 < 8 or y2 - y1 < 8:
        print("ROI too small — draw a larger rectangle.")
        return None

    return {
        "enabled": True,
        "x1": round(x1 / w, 6),
        "y1": round(y1 / h, 6),
        "x2": round(x2 / w, 6),
        "y2": round(y2 / h, 6),
        "source_width": w,
        "source_height": h,
    }


def main():
    parser = argparse.ArgumentParser(description="Draw pipeline ROI and save to JSON")
    parser.add_argument(
        "video_source",
        nargs="?",
        default=r"test_videos\Knowns.mp4",
        help="RTSP URL, video file, or webcam index",
    )
    parser.add_argument("--roi-file", help="Override ROI JSON path from pipeline_config.yaml")
    args = parser.parse_args()

    if not is_valid_video_source(args.video_source):
        print(f"Invalid source: {args.video_source}")
        return 1

    config = load_config()
    roi_path = args.roi_file or config.get("roi", {}).get("file", "roi.json")

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    frame = grab_first_frame(args.video_source)
    roi = draw_roi_interactive(frame)
    if roi is None:
        print("ROI not saved.")
        return 1

    save_roi(roi, path=roi_path, config=config)
    print(
        f"ROI normalized: x1={roi['x1']}, y1={roi['y1']}, "
        f"x2={roi['x2']}, y2={roi['y2']}"
    )
    print("Set roi.enabled: true in pipeline_config.yaml to enforce it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
