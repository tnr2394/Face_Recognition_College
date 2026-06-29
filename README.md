# Face Recognition Pipeline (v2)

Multi-service pipeline: **person tracking → face ranking → recognition API**.

## Quick start

```bash
# Detect + rank only (writes ranked_faces/)
python start_pipeline.py --no-show

# Full pipeline (writes recognition_folder/ + API)
python start_pipeline.py --with-recognition --no-show
```

## Required files

| File | Role |
|------|------|
| `start_pipeline.py` | Launches all services |
| `person_capture_service.py` | Stage 1: YOLO + ByteTrack → `person_queue/` |
| `face_ranker_service.py` | Stage 2: face quality + OFIQ → `ranked_faces/` or `recognition_folder/` |
| `recognition_watcher.py` | Stage 3: watches `recognition_folder/`, calls API |
| `face_quality.py` | Landmark gates, multi-face pick, roll alignment |
| `ofiq_scorer.py` | ONNX OFIQ quality score |
| `pipeline_io.py` | Folder contracts (`.ready`, `.processed`) |
| `pipeline_config.yaml` | All thresholds and paths |
| `cooldown_state.py` | Recognition cooldown shared state |
| `rao_tracker.yaml` | ByteTrack tracker tuning |
| `calibrate_pipeline.py` | Offline threshold tuning on a queue batch |
| `yolov8n.pt` | YOLO person model |

## Local assets (not in git)

- `models/yolov8n-face-landmarks.pt` — face detector
- `OFIQ-MODELS/...` — OFIQ ONNX model (see `pipeline_config.yaml`)

## Config

Edit `pipeline_config.yaml`. Key flags:

- `face_ranker.rank_only: true` — detect-only (default)
- `--with-recognition` — exports minimal `*_face_score_*.jpg` to `recognition_folder/`

## Runtime folders (auto-created, gitignored)

- `person_queue/` — capture handoff batches
- `ranked_faces/` — debug exports (detect-only mode)
- `recognition_folder/` — API handoff (recognition mode)

## ROI (region of interest)

ROI is applied **after face ranking** — all people are captured and faces are scored, then only ranked faces inside the ROI are exported.

1. Draw ROI:

```bash
python draw_roi.py rtsp://your-camera-url
```

2. Enable in `pipeline_config.yaml`:

```yaml
roi:
  enabled: true
  file: roi.json
  face_mode: overlap
  face_min_overlap: 0.15
```

Ranker logs include `outside_roi` for faces ranked but not exported.
