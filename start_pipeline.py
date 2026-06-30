"""Launch pipeline services. Default: person capture + face ranker only (no API recognition)."""

import os
import signal
import subprocess
import sys
import time

from pipeline_io import count_pending_batches, load_config


DETECT_SERVICES = [
    ("person_capture", "person_capture_service.py"),
    ("face_ranker", "face_ranker_service.py"),
]

FULL_SERVICES = DETECT_SERVICES + [
    ("recognition", "recognition_watcher.py"),
]

_processes = {}
_shutdown_requested = False


def _shutdown(signum=None, frame=None):
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutting down pipeline services...")
    for proc in _processes.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for proc in _processes.values():
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


def _drain_face_queue(queue_dir, ranker_proc, timeout_sec=180):
    """Wait for face ranker to process all pending person_queue batches."""
    print(f"Video finished — draining person_queue (up to {timeout_sec}s)...")
    deadline = time.time() + timeout_sec
    last_pending = -1
    while time.time() < deadline and not _shutdown_requested:
        if ranker_proc.poll() is not None:
            print("Face ranker exited before queue drain completed.")
            break
        pending = count_pending_batches(queue_dir)
        if pending == 0:
            print("person_queue fully ranked.")
            return
        if pending != last_pending:
            print(f"  {pending} batch(es) still pending in person_queue...")
            last_pending = pending
        time.sleep(2)
    remaining = count_pending_batches(queue_dir)
    if remaining:
        print(f"Drain timeout: {remaining} batch(es) still pending — run face_ranker_service.py to finish.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the face detection pipeline")
    parser.add_argument(
        "video_source",
        nargs="?",
        default=r"C:\Users\raoit\Work\Face_Recognition_College_OFIQ_RANKER\test_videos\Knowns.mp4",
        help="Video source for person capture (RTSP URL, file path, or webcam index)",
    )
    parser.add_argument("--no-show", action="store_true", help="Hide capture preview window")
    parser.add_argument(
        "--with-recognition",
        action="store_true",
        help="Also start recognition_watcher.py (API + Discord)",
    )
    parser.add_argument(
        "--draw-roi",
        action="store_true",
        help="Draw ROI on first frame before capture (saves roi.json)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    config = load_config()
    queue_dir = config["person"]["queue_dir"]

    services = FULL_SERVICES if args.with_recognition else DETECT_SERVICES
    child_env = os.environ.copy()
    if args.with_recognition:
        child_env["PIPELINE_RECOGNITION_MODE"] = "1"
        if args.video_source:
            child_env["PIPELINE_VIDEO_SOURCE"] = str(args.video_source)
        recognition_dir = config.get("ofiq", {}).get("recognition_dir", "recognition_folder")
        low_ofiq_dir = config.get("ofiq", {}).get("low_ofiq_dir", "low_ofiq_faces")
        outside_roi_dir = config.get("roi", {}).get("outside_roi_dir", "outside_roi_faces")
        rejected_dir = config.get("face_ranker", {}).get("rejected_dir", "rejected_faces")
        os.makedirs(recognition_dir, exist_ok=True)
        os.makedirs(low_ofiq_dir, exist_ok=True)
        os.makedirs(outside_roi_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)
        print("Mode: full pipeline (capture + rank -> recognition_folder + API)")
    else:
        print("Mode: detect only (capture + rank -> ranked_faces/)")

    for name, script in services:
        full_cmd = [sys.executable, script]
        if name == "person_capture":
            if args.video_source:
                full_cmd.append(args.video_source)
            if args.no_show:
                full_cmd.append("--no-show")
            if args.draw_roi:
                full_cmd.append("--draw-roi")
        print(f"Starting {name}: {' '.join(full_cmd)}")
        proc = subprocess.Popen(full_cmd, env=child_env)
        _processes[name] = proc
        time.sleep(1)

    print("Pipeline running. Press Ctrl+C to stop all services.")
    if args.with_recognition:
        recognition_dir = config.get("ofiq", {}).get("recognition_dir", "recognition_folder")
        print(f"Check {recognition_dir}/ for face exports; watcher sends *_face_score_* to API.")
    else:
        print("Check ranked_faces/ for best face exports per person (top 2 globally per track).")

    capture_is_file = isinstance(args.video_source, str) and os.path.isfile(args.video_source)
    capture_done = False

    while not _shutdown_requested:
        for name, proc in list(_processes.items()):
            code = proc.poll()
            if code is None:
                continue
            if name == "person_capture" and capture_is_file and not capture_done:
                capture_done = True
                ranker = _processes.get("face_ranker")
                drain_sec = 300 if args.with_recognition else 180
                if ranker is not None and ranker.poll() is None:
                    _drain_face_queue(queue_dir, ranker, timeout_sec=drain_sec)
                print("Person capture finished.")
                continue
            if name == "recognition" and capture_is_file:
                print(f"Recognition watcher exited (code {code}); ranker may still be draining.")
                _processes.pop(name, None)
                continue
            if name != "person_capture" or not capture_is_file:
                print(f"Service {name} exited with code {code}")
                _shutdown()
        time.sleep(2)

    _shutdown()


if __name__ == "__main__":
    main()
