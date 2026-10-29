/**
 * PM2 ecosystem — face API server + OFIQ capture/rank/watch pipeline.
 * cwd for pipeline apps = this repo; face-server cwd = face_identity_ui.
 *
 * Windows:
 *   .\pm2_pipeline.ps1 start
 *   .\pm2_pipeline.ps1 status | logs | stop | restart | delete | save
 *
 * Linux / Jetson:
 *   pm2 start ecosystem.config.js
 *
 * Optional env:
 *   FACE_REC_PYTHON=C:\Path\to\python.exe
 *   PIPELINE_VIDEO_SOURCE=cam1
 *   FACE_SERVER_URL=http://localhost:4445
 *   CAPTURE_SOURCE=rtsp://localhost:8554/live/cam1:270
 *   FACE_SERVER_DIR=/var/www/html/face-rec-college-py/face_identity_ui
 *
 * Ubuntu layout:
 *   /var/www/html/Face_Recognition_College/
 *   /var/www/html/face-rec-college-py/face_identity_ui/
 *
 * Add more apps under module.exports.apps anytime.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT_DIR = __dirname;
const LOG_DIR = path.join(ROOT_DIR, 'logs', 'pm2');
const RUN_PYTHON_CMD = path.join(ROOT_DIR, 'scripts', 'run_python.cmd');

// Sibling of this repo: .../face-rec-college-py/face_identity_ui
// On Ubuntu production that is usually /var/www/html/...
const FACE_SERVER_DIR =
  process.env.FACE_SERVER_DIR ||
  path.join(path.dirname(ROOT_DIR), 'face-rec-college-py', 'face_identity_ui');

const FACE_SERVER_URL =
  process.env.FACE_SERVER_URL || 'http://localhost:4445';

/** Video source for person_capture_service.py — override with CAPTURE_SOURCE env. */
// Password special chars must be URL-encoded (@ -> %40, # -> %23, etc.)
const CAPTURE_SOURCE =
  process.env.CAPTURE_SOURCE ||
  "C:\\Users\\raoit\\Work\\Face_Recognition_College_OFIQ_RANKER\\test_videos\\Knowns.mp4";

function quoteCmdArg(value) {
  // Keep & ? = and spaces inside one argv for cmd.exe /c
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
      'C:\\Python311\\python.exe',
      'C:\\Python310\\python.exe',
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
/** On Windows Store Python, use cmd wrapper so PM2 does not require existsSync(python.exe). */
const USE_CMD_WRAPPER = process.platform === 'win32' && !PYTHON;

if (USE_CMD_WRAPPER) {
  console.log(
    `[ecosystem] Using scripts\\run_python.cmd (Windows Store Python / PATH stub)`
  );
} else {
  console.log(`[ecosystem] Using Python interpreter: ${PYTHON || 'python3'}`);
}
console.log(`[ecosystem] Face server dir: ${FACE_SERVER_DIR}`);
console.log(`[ecosystem] FACE_SERVER_URL: ${FACE_SERVER_URL}`);

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
    // Windows: PM2 must spawn cmd.exe (spawn EINVAL on .cmd directly).
    // Quote every argv so RTSP `&` / `@` are not eaten by cmd.
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

module.exports = {
  apps: [
    // API + dashboard — start first; watcher calls this on localhost:4445
    pythonApp('face-server', 'face_rec_server_enhanced.py', {
      cwd: FACE_SERVER_DIR,
    }),
    // Stage 2: person_queue -> recognition_folder
    pythonApp('face-ranker', 'face_ranker_service.py', {
      env: {
        PIPELINE_RECOGNITION_MODE: '1', // '0' = detect-only (ranked_faces)
      },
    }),
    // Stage 3: recognition_folder -> API + Discord + /recognitions log
    pythonApp('recognition-watcher', 'recognition_watcher.py', {
      env: {
        FACE_SERVER_URL,
        PIPELINE_VIDEO_SOURCE:
          process.env.PIPELINE_VIDEO_SOURCE || 'cam1',
      },
    }),
    // Stage 1: RTSP / video -> person_queue
    // Pass URL via CAPTURE_SOURCE env (cmd.exe eats '&' in argv, which drops subtype=0)
    pythonApp('person-capture', 'person_capture_service.py', {
      env: {
        CAPTURE_SOURCE,
        PIPELINE_VIDEO_SOURCE:
          process.env.PIPELINE_VIDEO_SOURCE || 'cam1',
      },
    }),
  ],
};
