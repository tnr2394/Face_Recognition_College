/**
 * PM2 ecosystem — face API server + per-camera OFIQ capture/rank/watch.
 *
 * Cameras: cameras.json (edit RTSP/rotation in Settings UI — do not hardcode here).
 * Isolated dirs: work/<camera_id>/... so IN and OUT do not wipe each other.
 *
 *   pm2 start ecosystem.config.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT_DIR = __dirname;
const LOG_DIR = path.join(ROOT_DIR, 'logs', 'pm2');
const RUN_PYTHON_CMD = path.join(ROOT_DIR, 'scripts', 'run_python.cmd');
const CAMERAS_JSON = path.join(ROOT_DIR, 'cameras.json');

function loadPm2EnvFile() {
  const filePath = path.join(ROOT_DIR, '.env.pm2');
  const out = {};
  if (!fs.existsSync(filePath)) {
    return out;
  }
  for (const raw of fs.readFileSync(filePath, 'utf8').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }
    const m = line.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!m) {
      continue;
    }
    let value = m[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[m[1]] = value;
  }
  return out;
}

function loadCameras() {
  if (!fs.existsSync(CAMERAS_JSON)) {
    console.warn(`[ecosystem] Missing ${CAMERAS_JSON}`);
    return [];
  }
  try {
    const data = JSON.parse(fs.readFileSync(CAMERAS_JSON, 'utf8'));
    return Array.isArray(data.cameras) ? data.cameras : [];
  } catch (err) {
    console.error(`[ecosystem] Failed to parse cameras.json: ${err.message}`);
    return [];
  }
}

const FILE_ENV = loadPm2EnvFile();

const FACE_SERVER_DIR =
  FILE_ENV.FACE_SERVER_DIR ||
  process.env.FACE_SERVER_DIR ||
  path.join(path.dirname(ROOT_DIR), 'face-rec-college-py', 'face_identity_ui');

const FACE_SERVER_URL =
  FILE_ENV.FACE_SERVER_URL ||
  process.env.FACE_SERVER_URL ||
  'http://localhost:4445';

if (process.env.FACE_REC_PYTHON == null && FILE_ENV.FACE_REC_PYTHON) {
  process.env.FACE_REC_PYTHON = FILE_ENV.FACE_REC_PYTHON;
}

function quoteCmdArg(value) {
  const s = String(value);
  return `"${s.replace(/"/g, '""')}"`;
}

function resolvePythonInterpreter() {
  if (process.env.FACE_REC_PYTHON && fs.existsSync(process.env.FACE_REC_PYTHON)) {
    return process.env.FACE_REC_PYTHON;
  }
  const known = [];
  if (process.platform === 'win32') {
    const local = process.env.LOCALAPPDATA || '';
    const home = process.env.USERPROFILE || '';
    known.push(
      path.join(local, 'Programs', 'Python', 'Python312', 'python.exe'),
      path.join(local, 'Programs', 'Python', 'Python311', 'python.exe'),
      path.join(local, 'Programs', 'Python', 'Python310', 'python.exe'),
      'C:\\Python312\\python.exe',
      path.join(home, 'anaconda3', 'python.exe'),
      path.join(home, 'miniconda3', 'python.exe')
    );
  }
  for (const p of known) {
    if (p && fs.existsSync(p)) {
      return p;
    }
  }
  const candidates =
    process.platform === 'win32'
      ? ['python', 'python3', 'py']
      : ['python3', 'python'];
  for (const cmd of candidates) {
    try {
      const flag =
        cmd === 'py'
          ? '-3 -c "import sys; print(sys.executable)"'
          : '-c "import sys; print(sys.executable)"';
      const out = execSync(`${cmd} ${flag}`, {
        encoding: 'utf8',
        windowsHide: true,
      }).trim();
      if (out && fs.existsSync(out)) {
        return out;
      }
    } catch (_err) {
      // try next
    }
  }
  return null;
}

const PYTHON = resolvePythonInterpreter();
const USE_CMD_WRAPPER = process.platform === 'win32' && !PYTHON;
const CAMERAS = loadCameras();

console.log(`[ecosystem] Using Python interpreter: ${PYTHON || 'python3'}`);
console.log(`[ecosystem] Face server dir: ${FACE_SERVER_DIR}`);
console.log(`[ecosystem] FACE_SERVER_URL: ${FACE_SERVER_URL}`);
console.log(
  `[ecosystem] Cameras: ${CAMERAS.map((c) => c.id).join(', ') || '(none)'}`
);

if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

function pythonApp(name, scriptFile, extra = {}) {
  const cwd = extra.cwd || ROOT_DIR;
  const base = {
    name,
    cwd,
    windowsHide: true,
    env: {
      PYTHONUNBUFFERED: '1',
      FACE_SERVER_URL,
      ...(extra.env || {}),
    },
    autorestart: true,
    max_restarts: 15,
    min_uptime: '10s',
    restart_delay: 5000,
    watch: false,
    merge_logs: true,
    time: true,
    out_file: path.join(LOG_DIR, `${name}-out.log`),
    error_file: path.join(LOG_DIR, `${name}-error.log`),
  };

  if (USE_CMD_WRAPPER) {
    const absScript = path.isAbsolute(scriptFile)
      ? scriptFile
      : path.join(cwd, scriptFile);
    const argList = Array.isArray(extra.args)
      ? extra.args
      : extra.args
        ? [String(extra.args).trim()]
        : [];
    const pyArgs = [RUN_PYTHON_CMD, absScript, ...argList]
      .map(quoteCmdArg)
      .join(' ');
    return {
      ...base,
      script: 'cmd',
      args: `/c ${pyArgs}`,
      interpreter: 'none',
    };
  }

  return {
    ...base,
    script: scriptFile,
    interpreter: PYTHON || (process.platform === 'win32' ? 'python' : 'python3'),
    args: extra.args,
  };
}

function cameraWorkEnv(cameraId) {
  const work = path.join(ROOT_DIR, 'work', cameraId);
  return {
    CAMERA_ID: cameraId,
    PIPELINE_VIDEO_SOURCE: cameraId,
    PERSON_QUEUE_DIR: path.join(work, 'person_queue'),
    RECOGNITION_DIR: path.join(work, 'recognition_folder'),
    LOW_OFIQ_DIR: path.join(work, 'low_ofiq_faces'),
    REJECTED_DIR: path.join(work, 'rejected_faces'),
    OUTSIDE_ROI_DIR: path.join(work, 'outside_roi_faces'),
    RANKED_DIR: path.join(work, 'ranked_faces'),
    ROI_FILE: path.join(work, 'roi.json'),
    PIPELINE_RECOGNITION_MODE: '1',
    FACE_SERVER_URL,
  };
}

const apps = [
  pythonApp('face-server', 'face_rec_server_enhanced.py', {
    cwd: FACE_SERVER_DIR,
    env: {
      FACE_SERVER_FORCE_CPU: '1',
      CUDA_VISIBLE_DEVICES: '-1',
    },
  }),
];

for (const cam of CAMERAS) {
  const id = String(cam.id || '').trim();
  if (!id) {
    continue;
  }
  const env = cameraWorkEnv(id);
  apps.push(
    pythonApp(`face-ranker-${id}`, 'face_ranker_service.py', { env }),
    pythonApp(`recognition-watcher-${id}`, 'recognition_watcher.py', {
      env: { ...env, FACE_SERVER_URL },
    }),
    pythonApp(`person-capture-${id}`, 'person_capture_service.py', { env })
  );
}

module.exports = { apps };
