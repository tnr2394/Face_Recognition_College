# Fresh Ubuntu + GPU setup (tracking + face server)

Use this on a **blank Ubuntu machine with NVIDIA GPU**. No enrolled faces are required — the server creates empty tables; you enroll people later from the UI.

## Repositories

| Role | GitHub | Branch |
|------|--------|--------|
| Tracking / ranker / watcher | https://github.com/tnr2394/Face_Recognition_College | `feature/pipeline-v2` |
| Face server + UI | https://github.com/RaoInfotech/face-recognition-with-training | `master` |

Clone them **side by side**:

```bash
mkdir -p ~/Work && cd ~/Work
git clone -b feature/pipeline-v2 https://github.com/tnr2394/Face_Recognition_College.git
mkdir -p face-rec-college-py
git clone -b master https://github.com/RaoInfotech/face-recognition-with-training.git face-rec-college-py/face_identity_ui
# Expected layout:
#   ~/Work/Face_Recognition_College/
#   ~/Work/face-rec-college-py/face_identity_ui/
```

> Large model files are **not** in git. Copy from your working PC:
> - `yolov8n.pt`
> - `models/yolov8n-face-landmarks.pt` (or `yolov8n-face-landmarks.pt` in repo root)
> - `OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx`

## One-command install

```bash
cd ~/Work/Face_Recognition_College
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

What the script does:

1. Installs system packages (Python, Node, PM2, FFmpeg, PostgreSQL, **pgvector**)
2. Creates Python venv + installs **GPU** PyTorch, **onnxruntime-gpu**, TensorFlow CUDA, Ultralytics, Flask, etc.
3. Creates Postgres user/databases (`employees`, `vectors`) and enables `vector` extension
4. Writes a sample `ecosystem` env hint file
5. Prints GPU verification commands

Then copy models, set `CAPTURE_SOURCE`, and start PM2 (see below).

## Manual PostgreSQL (if you skip the script)

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
# If package name fails, try: postgresql-15-pgvector or postgresql-14-pgvector

sudo systemctl enable --now postgresql

# Set password for postgres user (example: postgres)
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# Create DBs (server also auto-creates tables on first start)
sudo -u postgres psql <<'SQL'
CREATE DATABASE employees;
CREATE DATABASE vectors;
\c vectors
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

Match `face_identity_ui/config.json`:

```json
"database": {
  "main_database": "employees",
  "vector_database": "vectors",
  "host": "localhost",
  "user": "postgres",
  "password": "postgres",
  "vector_size": 512
},
"api": { "host": "0.0.0.0", "port": 4445 }
```

Tables (`employees`, `employee_images`, `recognition_events`, `face_embeddings`) are created automatically when the face server starts.

## Configure RTSP + PM2

```bash
export CAPTURE_SOURCE="rtsp://localhost:8554/live/cam1:270"   # :270 = rotate 270°
export FACE_SERVER_URL="http://localhost:4445"
export FACE_SERVER_DIR="$HOME/Work/face-rec-college-py/face_identity_ui"
export PIPELINE_VIDEO_SOURCE="cam1"
export FACE_REC_PYTHON="$HOME/Work/Face_Recognition_College/.venv/bin/python"

cd ~/Work/Face_Recognition_College
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Verify GPU

```bash
source ~/Work/Face_Recognition_College/.venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
nvidia-smi
```

Expect: torch CUDA True, `CUDAExecutionProvider`, TF GPU devices listed.  
Face server on Linux enables GPU by default (CPU on Windows unless `FACE_SERVER_FORCE_GPU=1`).

## After start — enroll faces

1. Open `http://<machine-ip>:4445` (UI from face server)
2. Enroll employees / face images (empty DB is fine at first)
3. Walk in front of the camera — capture → ranker → watcher → recognition logs

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Interpreter python NOT AVAILABLE` | Set `FACE_REC_PYTHON` to the venv python absolute path |
| OFIQ on CPU only | `pip uninstall onnxruntime` then `pip install onnxruntime-gpu` |
| TF no GPU | Install `tensorflow[and-cuda]` matching your driver/CUDA |
| RTSP stall / reconnect loops | Check MediaMTX; raise `rtsp_stall_timeout_sec` in `pipeline_config.yaml` |
| Missing face model | Copy `yolov8n-face-landmarks.pt` into `models/` |
