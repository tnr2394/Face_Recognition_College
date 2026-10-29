#!/usr/bin/env bash
# Fresh Ubuntu installer for Face Recognition pipeline + face server deps.
# Run from Face_Recognition_College repo root:
#   chmod +x install_ubuntu.sh && ./install_ubuntu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
# Default deploy root on Ubuntu production
WWW_ROOT="/var/www/html"
FACE_UI_DEFAULT="${WWW_ROOT}/face-rec-college-py/face_identity_ui"
# If this repo is already under /var/www/html, prefer sibling path from ROOT
if [[ "${ROOT}" == "${WWW_ROOT}"/* ]]; then
  FACE_UI_DEFAULT="$(cd "${ROOT}/.." && pwd)/face-rec-college-py/face_identity_ui"
fi

echo "==> Face Recognition College — Ubuntu GPU install"
echo "    Repo: ${ROOT}"
echo "    Face UI (expected): ${FACE_UI_DEFAULT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Ubuntu/Linux. On Windows use the existing PM2 setup."
  exit 1
fi

echo "==> [1/6] System packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-venv python3-pip python3-dev \
  build-essential git curl wget \
  ffmpeg libgl1 libglib2.0-0 \
  postgresql postgresql-contrib \
  nodejs npm

# pgvector — try common package names
if ! dpkg -l | grep -q pgvector; then
  sudo apt-get install -y postgresql-16-pgvector \
    || sudo apt-get install -y postgresql-15-pgvector \
    || sudo apt-get install -y postgresql-14-pgvector \
    || echo "WARNING: pgvector package not found — install manually for your Postgres major version"
fi

if ! command -v pm2 >/dev/null 2>&1; then
  sudo npm install -g pm2
fi

echo "==> [2/6] NVIDIA check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA drivers before expecting GPU."
fi

echo "==> [3/6] Python venv + GPU packages"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel setuptools

# PyTorch CUDA 12.4 wheels (change if your CUDA differs — see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip uninstall -y onnxruntime 2>/dev/null || true
pip install onnxruntime-gpu
pip install "tensorflow[and-cuda]" || pip install tensorflow

pip install \
  ultralytics opencv-python-headless numpy pyyaml requests \
  flask flask-cors pillow mtcnn psycopg2-binary keras-facenet scikit-learn

echo "==> [4/6] Verify GPU Python stacks"
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch.cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch check failed:", e)
try:
    import onnxruntime as ort
    print("onnxruntime providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime check failed:", e)
try:
    import tensorflow as tf
    print("tf GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("tensorflow check failed:", e)
PY

echo "==> [5/6] PostgreSQL databases (empty — enroll faces later)"
sudo systemctl enable --now postgresql

# Ensure postgres role can login with password 'postgres' (change later in config.json)
sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL' || true
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'postgres') THEN
    CREATE ROLE postgres LOGIN PASSWORD 'postgres' SUPERUSER;
  ELSE
    ALTER ROLE postgres WITH LOGIN PASSWORD 'postgres';
  END IF;
END$$;
SQL

# peercmd for local password auth — keep default peer for unix socket; server uses password on TCP
sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'CREATE DATABASE employees' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'employees')\gexec
SELECT 'CREATE DATABASE vectors' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'vectors')\gexec
\c vectors
CREATE EXTENSION IF NOT EXISTS vector;
SQL

echo "Postgres ready: databases employees + vectors (pgvector)."
echo "Face server will create tables on first start (employees, face_embeddings, recognition_events, ...)."

echo "==> [6/6] Env hint file"
cat > "${ROOT}/.env.pm2.example" <<EOF
# Copy these into your shell before: pm2 start ecosystem.config.js
export FACE_REC_PYTHON="${VENV}/bin/python"
export FACE_SERVER_DIR="${FACE_UI_DEFAULT}"
export FACE_SERVER_URL="http://localhost:4445"
export PIPELINE_VIDEO_SOURCE="cam1"
export CAPTURE_SOURCE="rtsp://localhost:8554/live/cam1:270"
EOF

echo
echo "DONE."
echo "Next steps:"
echo "  1) Copy model files into this repo (yolov8n.pt, face landmarks, OFIQ ONNX)"
echo "  2) Clone face server if missing:"
echo "       git clone -b master https://github.com/RaoInfotech/face-recognition-with-training.git ${FACE_UI_DEFAULT}"
echo "  3) source ${ROOT}/.env.pm2.example   # edit CAPTURE_SOURCE"
echo "  4) cd ${ROOT} && pm2 start ecosystem.config.js && pm2 save"
echo "  5) Open http://localhost:4445 and enroll faces"
echo
echo "Full guide: ${ROOT}/SETUP_UBUNTU.md"
