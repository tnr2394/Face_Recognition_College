import os
import time
import re
import shutil
import requests
import base64
import cv2
from datetime import datetime
import json
import glob
from collections import Counter

from cooldown_state import (
    COOLDOWN_IN_TRAINING,
    parse_track_id_from_folder,
    set_cooldown,
)
from pipeline_io import load_config

FACE_SERVER_URL = "http://192.170.1.114:4445"
TRAINING_MODE_CACHE_SECONDS = 30
_training_mode_cache = {'value': None, 'checked_at': 0}
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1143427649243459584/NhuVUtPoNnMBlBlBXKTKFppZiFyBOTWYbzwxSXZb5MJm-NrSK30rW4DcV3Rwx19d6rRT"
DISCORD_WEBHOOK_UNRECOGNIZED = DISCORD_WEBHOOK

GOOGLE_SHEET_API_URL = "https://script.google.com/macros/s/AKfycbye9j6E3fmdZ_b636aYcDxyVHtH7_lNgYwJHRieec1XCdL9nZlqgAWmyC2ib92qjgSz3g/exec"

RECOGNITION_FOLDER = "recognition_folder"
LOW_OFIQ_FOLDER = "low_ofiq_faces"
MAX_API_IMAGE_DIM = 1024  # cap longest side for API payload; preserve aspect ratio
INACTIVITY_WAIT = 5  # seconds to wait for folder inactivity
CHECK_INTERVAL = 3   # seconds between folder checks
RECOGNITION_BATCH_SIZE = 15  # Process up to this many faces per person (same as face_recognition.py)

def is_training_mode_enabled():
    """Read training mode from the face server (cached briefly)."""
    now = time.time()
    if (
        _training_mode_cache['value'] is not None
        and now - _training_mode_cache['checked_at'] < TRAINING_MODE_CACHE_SECONDS
    ):
        return _training_mode_cache['value']

    try:
        response = requests.get(f"{FACE_SERVER_URL}/settings/training", timeout=5)
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                enabled = bool(payload.get('enabled', False))
            else:
                config_response = requests.get(f"{FACE_SERVER_URL}/config", timeout=5)
                config_payload = config_response.json() if config_response.status_code == 200 else {}
                enabled = bool((config_payload or {}).get('training', {}).get('enabled', False))
            _training_mode_cache['value'] = enabled
            _training_mode_cache['checked_at'] = now
            return enabled
    except Exception as e:
        print(f"Could not fetch training mode from server: {e}")

    if _training_mode_cache['value'] is not None:
        return _training_mode_cache['value']
    return False

def encode_face_image_b64(image_path):
    """Encode image for API — preserve aspect ratio, optional max dimension cap."""
    img = cv2.imread(image_path)
    if img is None:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    h, w = img.shape[:2]
    scale = min(1.0, MAX_API_IMAGE_DIM / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    ok, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    return base64.b64encode(buffer).decode('utf-8')

def recognize_face(image_path, track_id):
    """Send face image to local server: /extract in training mode, /search otherwise."""
    try:
        training_mode = is_training_mode_enabled()
        endpoint = "extract" if training_mode else "search"
        print(f"Recognizing face ({endpoint}): {image_path}")
        url = f"{FACE_SERVER_URL}/{endpoint}"

        base64_image = encode_face_image_b64(image_path)

        data = {'image': base64_image}
        if not training_mode:
            data['id'] = 1

        response = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            json=data,
            timeout=30,
        )

        print(f"API Response Status: {response.status_code}")
        if response.status_code != 200:
            print(f"API Error Details: {response.text}")
            return {
                "status": False,
                "image_path": image_path,
                "track_id": track_id,
                "error": f"API Error {response.status_code}: {response.text[:200]}",
            }

        result = response.json()
        print(f"API Response: {result}")

        if training_mode:
            if result.get('error') or 'id' not in result:
                return {
                    "status": False,
                    "image_path": image_path,
                    "track_id": track_id,
                    "error": result.get('error', 'Training extract failed'),
                }
            person_name = result.get('name', 'Unknown')
        else:
            if result.get('status') == 'No match found' or 'id' not in result:
                return {
                    "status": False,
                    "image_path": image_path,
                    "track_id": track_id,
                }
            person_name = result['name']

        name_parts = person_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        return {
            "status": True,
            "id": result["id"],
            "name": person_name,
            "first_name": first_name,
            "last_name": last_name,
            "distance_image": result.get("distance", 0),
            "image_path": image_path,
            "track_id": track_id,
        }
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        return {
            "status": False,
            "image_path": image_path,
            "track_id": track_id,
            "error": str(e),
        }

def send_to_google_sheets(data):
    try:
        resp = requests.post(GOOGLE_SHEET_API_URL, json=data)
        if resp.status_code == 200:
            print("Logged to Google Sheets")
        else:
            print(f"Google Sheets API error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Error sending to Google Sheets: {e}")

def send_to_discord(result, webhook_url=None, tracking_id=None):
    """Send recognition result and image to Discord webhook as a rich embed with image preview above the embed."""
    if webhook_url is None:
        webhook_url = DISCORD_WEBHOOK
    
    # Use full frame for display if available, otherwise use the processed image
    display_path = result.get('display_image_path', result['image_path'])
    orig_path = result.get('orig_image_path', display_path)
    
    img = cv2.imread(display_path)
    if img is not None:
        height, width = img.shape[:2]
        box_size = f"{width} x {height}"
    else:
        box_size = "N/A"
    ts = os.path.getmtime(orig_path)
    # For Discord: append tracking_id to timestamp
    captured_at = datetime.fromtimestamp(ts).strftime('%d %b %H:%M')
    if tracking_id is not None:
        captured_at = f"{captured_at} ({tracking_id})"
    score = parse_score(orig_path)

    if result['status']:
        title = f"{result.get('name', 'Unknown')} Recognized"
        distance = result.get('distance_image', 'N/A')
        color = 3066993  # green
        fields = [
            {"name": "Distance:", "value": str(distance), "inline": False},
            {"name": "Box Size:", "value": box_size, "inline": False},
            {"name": "Score:", "value": str(score), "inline": False},
            {"name": "Captured At", "value": captured_at, "inline": False},
        ]
    else:
        title = "Not recognized"
        color = 15158332  # red
        fields = [
            {"name": "Box Size:", "value": box_size, "inline": False},
            {"name": "Score:", "value": str(score), "inline": False},
            {"name": "Captured At", "value": captured_at, "inline": False},
        ]
    embed = {
        "title": title,
        "fields": fields,
        "image": {"url": f"attachment://{os.path.basename(display_path)}"},
        "color": color
    }
    data = {"embeds": [embed]}
    with open(display_path, 'rb') as f:
        files = {"file": (os.path.basename(display_path), f, "image/jpeg")}
        try:
            resp = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(data)},
                files=files
            )
            if resp.status_code in (200, 204):
                print(f"Sent to Discord: {display_path}")
            else:
                print(f"Discord webhook error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Error sending to Discord: {e}")
    # --- Send to Google Sheets ---
    # For Google Sheets: do not include tracking_id in timestamp
    # sheet_data = {
    #     "name": result.get("name", "Unknown"),
    #     "status": "recognized" if result["status"] else "unrecognized",
    #     "score": score,
    #     "image_filename": os.path.basename(display_path)
    # }
    # send_to_google_sheets(sheet_data)

def parse_score(filename):
    match = re.search(r'_score_([0-9]+(?:\.[0-9]+)?)', filename)
    return float(match.group(1)) if match else 0.0

def list_face_crop_files(person_dir):
    """Scored face crops for API; fall back to person crop, then full frame."""
    face_scored = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith(('.jpg', '.png'))
        and '_face_score_' in f
        and not f.startswith('.')
    ]
    if face_scored:
        return face_scored
    person_scored = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith(('.jpg', '.png'))
        and '_person_score_' in f
        and not f.startswith('.')
    ]
    if person_scored:
        return person_scored
    return [
        f for f in os.listdir(person_dir)
        if f.lower().endswith(('.jpg', '.png'))
        and '_full_score_' in f
        and not f.startswith('.')
    ]

def get_ofiq_threshold():
    return load_config().get("ofiq", {}).get("threshold", 16)

def get_low_ofiq_folder():
    return load_config().get("ofiq", {}).get("low_ofiq_dir", LOW_OFIQ_FOLDER)

def folder_max_ofiq_score(person_dir):
    scores = [parse_score(f) for f in list_face_crop_files(person_dir)]
    return max(scores) if scores else 0.0

def relocate_to_low_ofiq(person_dir, person_folder):
    """Move a recognition handoff folder to low_ofiq_faces/ (no API / Discord)."""
    dest_root = os.path.join(get_low_ofiq_folder(), person_folder)
    os.makedirs(get_low_ofiq_folder(), exist_ok=True)
    ready_file = os.path.join(person_dir, ".ready")
    if os.path.isfile(ready_file):
        os.remove(ready_file)
    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
    shutil.move(person_dir, dest_root)
    print(f"Moved {person_folder} -> {dest_root} (OFIQ below threshold)")
    return dest_root

def get_top_cropped_faces(person_dir, top_n=RECOGNITION_BATCH_SIZE):
    """Return highest-OFIQ image paths for API batching."""
    images = list_face_crop_files(person_dir)
    scored = [(img, parse_score(img)) for img in images]
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] Sorted samples by OFIQ score: {scored[:top_n]}")
    return [os.path.join(person_dir, img) for img, _ in scored[:top_n]]

def get_full_frame_for_crop(person_dir, crop_path):
    """Match a face/person crop to its full-frame pair for Discord display."""
    if not crop_path:
        return crop_path

    base = os.path.basename(crop_path)
    if '_full_score_' in base:
        return crop_path

    frame_num = None
    match = re.match(r'.*frame_(\d+)_(?:face|person)_score_.*', base)
    if match:
        frame_num = match.group(1)

    if frame_num:
        for pattern in (
            os.path.join(person_dir, f"frame_{frame_num}_full_score_*.jpg"),
            os.path.join(person_dir, f"frame_{frame_num}_full.jpg"),
        ):
            candidates = glob.glob(pattern)
            if candidates:
                return candidates[0]

        manifest_path = os.path.join(person_dir, '_rank_manifest.json')
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, encoding='utf-8') as f:
                    for entry in json.load(f).get('entries', []):
                        if str(entry.get('frame_num')) == frame_num:
                            full_path = entry.get('full_path')
                            if full_path and os.path.isfile(full_path):
                                return full_path
            except (json.JSONDecodeError, OSError):
                pass

    full_frame = os.path.join(person_dir, 'full_frame.jpg')
    if os.path.isfile(full_frame):
        return full_frame

    return crop_path

def get_metadata_from_txt(image_path, folder_path):
    """Extract metadata from corresponding txt file (same as face_recognition.py)."""
    try:
        if not image_path:
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            txt_path = os.path.join(folder_path, txt_files[0]) if txt_files else None
        else:
            image_filename = os.path.basename(image_path)
            txt_filename = image_filename.replace('.jpg', '.txt')
            txt_path = os.path.join(folder_path, txt_filename)

        if not txt_path or not os.path.exists(txt_path):
            return {}

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        metadata = {}
        for line in lines:
            line = line.strip()
            if line.startswith('VideoName:'):
                metadata['VideoName'] = line.split(':', 1)[1].strip()
            elif ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()

        if 'Box' in metadata:
            box_str = metadata['Box']
            match = re.findall(r'\d+', box_str)
            if len(match) >= 4:
                x1, y1, x2, y2 = map(int, match[:4])
                metadata['box_width'] = x2 - x1
                metadata['box_height'] = y2 - y1

        return metadata
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return {}

def determine_final_result(recognition_results, folder_path, track_id):
    """Determine final recognition result using majority voting (same as face_recognition.py)."""
    try:
        all_face_files = list_face_crop_files(folder_path)
        total_collected = len(all_face_files)

        print(
            f"Person {track_id}: collected {total_collected} samples, "
            f"processed {len(recognition_results)} via API"
        )

        successful_recognitions = [r for r in recognition_results if r.get('status')]

        if successful_recognitions:
            person_ids = [r['id'] for r in successful_recognitions]
            most_common_id = Counter(person_ids).most_common(1)[0][0]
            final_person_data = next(r for r in successful_recognitions if r['id'] == most_common_id)
            metadata = get_metadata_from_txt(final_person_data['image_path'], folder_path)

            return {
                "status": True,
                "recognition_data": final_person_data,
                "metadata": metadata,
                "track_id": track_id,
                "total_samples": total_collected,
                "processed_samples": len(recognition_results),
                "successful_samples": len(successful_recognitions),
            }

        first_result = recognition_results[0] if recognition_results else {}
        metadata = get_metadata_from_txt(first_result.get('image_path', ''), folder_path)
        errors = [r['error'] for r in recognition_results if 'error' in r]

        print(f"Recognition failed for person {track_id}: {len(recognition_results)} attempts, 0 successful")
        if errors:
            print(f"API errors encountered: {errors[:3]}")

        final_result = {
            "status": False,
            "metadata": metadata,
            "track_id": track_id,
            "total_samples": total_collected,
            "processed_samples": len(recognition_results),
            "successful_samples": 0,
        }
        if errors:
            final_result["errors"] = errors[:5]
        return final_result
    except Exception as e:
        print(f"Error determining final result: {e}")
        return {"status": False, "track_id": track_id, "error": str(e)}

def ensure_full_frame_copy(person_dir, final_result):
    """Copy a representative full frame to full_frame.jpg for dashboard logging."""
    full_frame_dst = os.path.join(person_dir, 'full_frame.jpg')
    if os.path.isfile(full_frame_dst):
        return

    image_path = ''
    if final_result.get('recognition_data'):
        image_path = final_result['recognition_data'].get('image_path', '')
    if not image_path:
        crops = get_top_cropped_faces(person_dir, 1)
        image_path = crops[0] if crops else ''

    if image_path:
        full_frame_src = get_full_frame_for_crop(person_dir, image_path)
        if full_frame_src and os.path.isfile(full_frame_src):
            if full_frame_src != image_path or not os.path.isfile(full_frame_dst):
                shutil.copy2(full_frame_src, full_frame_dst)
            return

    manifest_path = os.path.join(person_dir, '_rank_manifest.json')
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding='utf-8') as f:
                entries = json.load(f).get('entries', [])
            for entry in sorted(entries, key=lambda e: tuple(e.get('rank', [])), reverse=True):
                full_path = entry.get('full_path')
                if full_path and os.path.isfile(full_path):
                    shutil.copy2(full_path, full_frame_dst)
                    return
        except (json.JSONDecodeError, OSError):
            pass

    frame_files = [f for f in os.listdir(person_dir) if '_full_score_' in f and f.endswith('.jpg')]
    if frame_files:
        shutil.copy2(os.path.join(person_dir, frame_files[0]), full_frame_dst)

def log_recognition_event(folder_path, track_id, folder_name, final_result):
    """POST recognition result to face server for recognition_events table (same as face_recognition.py)."""
    try:
        face_image_path = None
        if final_result.get('status') and final_result.get('recognition_data'):
            rec_data = final_result['recognition_data']
            if rec_data.get('image_path'):
                candidate = os.path.join(folder_path, os.path.basename(rec_data['image_path']))
                if not os.path.isfile(candidate):
                    candidate = rec_data['image_path']
                if os.path.isfile(candidate):
                    face_image_path = candidate

        if not face_image_path:
            crops = get_top_cropped_faces(folder_path, 1)
            if crops:
                face_image_path = crops[0]

        frame_image_path = os.path.join(folder_path, 'full_frame.jpg')
        if not os.path.isfile(frame_image_path):
            frame_files = [
                f for f in os.listdir(folder_path)
                if '_full_score_' in f and f.endswith('.jpg')
            ]
            frame_image_path = os.path.join(folder_path, frame_files[0]) if frame_files else None

        metadata = final_result.get('metadata') or {}
        payload = {
            'status': bool(final_result.get('status')),
            'track_id': str(track_id),
            'track_folder': folder_name,
            'video_source': metadata.get('VideoName'),
            'total_samples': final_result.get('total_samples'),
            'successful_samples': final_result.get('successful_samples'),
            'processed_samples': final_result.get('processed_samples'),
            'metadata': metadata,
        }

        if face_image_path and os.path.isfile(face_image_path):
            with open(face_image_path, 'rb') as img_f:
                payload['face_image_b64'] = base64.b64encode(img_f.read()).decode('utf-8')
        if frame_image_path and os.path.isfile(frame_image_path):
            with open(frame_image_path, 'rb') as img_f:
                payload['frame_image_b64'] = base64.b64encode(img_f.read()).decode('utf-8')

        if final_result.get('status') and final_result.get('recognition_data'):
            rec_data = final_result['recognition_data']
            payload['employee_id'] = rec_data.get('id')
            payload['person_name'] = rec_data.get('name')
            payload['distance'] = rec_data.get('distance_image')

        response = requests.post(
            f"{FACE_SERVER_URL}/recognitions",
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=10,
        )
        if response.status_code in (200, 201):
            print(f"Logged recognition event for person {track_id} (recognition_events)")
        else:
            print(f"Warning: failed to log recognition event ({response.status_code}): {response.text[:200]}")
    except Exception as e:
        print(f"Warning: could not log recognition event for person {track_id}: {e}")

def build_discord_result(person_dir, final_result):
    """Build a Discord payload from the voted final recognition result."""
    if final_result.get('status') and final_result.get('recognition_data'):
        result = dict(final_result['recognition_data'])
        result['status'] = True
    else:
        crops = get_top_cropped_faces(person_dir, 1)
        crop_path = crops[0] if crops else ''
        result = {'status': False, 'image_path': crop_path}

    crop_path = result.get('image_path', '')
    if crop_path:
        result['orig_image_path'] = crop_path
        display_path = get_full_frame_for_crop(person_dir, crop_path)
        if display_path == crop_path and os.path.isfile(crop_path):
            full_frame_dst = os.path.join(person_dir, 'full_frame.jpg')
            if not os.path.isfile(full_frame_dst):
                manifest_path = os.path.join(person_dir, '_rank_manifest.json')
                if os.path.isfile(manifest_path):
                    try:
                        with open(manifest_path, encoding='utf-8') as f:
                            entries = json.load(f).get('entries', [])
                        for entry in sorted(
                            entries, key=lambda e: tuple(e.get('rank', [])), reverse=True
                        ):
                            full_path = entry.get('full_path')
                            if full_path and os.path.isfile(full_path):
                                shutil.copy2(full_path, full_frame_dst)
                                display_path = full_frame_dst
                                break
                    except (json.JSONDecodeError, OSError):
                        pass
        result['display_image_path'] = display_path
    return result

def process_person_folder(person_dir, person_folder, tracking_id):
    """Run batch recognition, save result JSON, log to DB, and notify Discord."""
    face_files = get_top_cropped_faces(person_dir, RECOGNITION_BATCH_SIZE)
    if not face_files:
        print(f"Person {tracking_id} has no face crops — skipping empty recognition folder")
        ready_file = os.path.join(person_dir, ".ready")
        if os.path.isfile(ready_file):
            os.remove(ready_file)
        mark_processed(person_dir)
        return

    threshold = get_ofiq_threshold()
    max_score = folder_max_ofiq_score(person_dir)
    if max_score <= threshold:
        print(
            f"Person {tracking_id} max OFIQ {max_score:.2f} <= {threshold} — "
            f"no API/Discord, moving to {get_low_ofiq_folder()}/"
        )
        relocate_to_low_ofiq(person_dir, person_folder)
        mark_processed(os.path.join(get_low_ofiq_folder(), person_folder))
        ready_file = os.path.join(RECOGNITION_FOLDER, person_folder, ".ready")
        if os.path.isfile(ready_file):
            os.remove(ready_file)
        return

    training_mode = is_training_mode_enabled()
    mode_label = "training (/extract)" if training_mode else "search (/search)"
    print(f"Processing person {tracking_id} with {len(face_files)} face samples [{mode_label}]")

    recognition_results = []
    for face_path in face_files:
        recognition_results.append(recognize_face(face_path, tracking_id))

    final_result = determine_final_result(recognition_results, person_dir, tracking_id)
    ensure_full_frame_copy(person_dir, final_result)

    result_file = os.path.join(person_dir, 'recognition_result.json')
    with open(result_file, 'w') as f:
        json.dump(final_result, f, indent=2, default=str)

    log_recognition_event(person_dir, tracking_id, person_folder, final_result)

    discord_result = build_discord_result(person_dir, final_result)
    if final_result.get('status'):
        if not (is_training_mode_enabled() and not COOLDOWN_IN_TRAINING):
            employee_id = final_result.get('recognition_data', {}).get('id')
            set_cooldown(employee_id=employee_id)
        send_to_discord(discord_result, webhook_url=DISCORD_WEBHOOK, tracking_id=tracking_id)
    else:
        send_to_discord(discord_result, webhook_url=DISCORD_WEBHOOK_UNRECOGNIZED, tracking_id=tracking_id)

def folder_inactive(person_dir, wait=INACTIVITY_WAIT):
    """Return True if no file in the folder has been modified in the last 'wait' seconds."""
    now = time.time()
    for fname in os.listdir(person_dir):
        fpath = os.path.join(person_dir, fname)
        if os.path.isfile(fpath):
            if now - os.path.getmtime(fpath) < wait:
                return False
    return True

def mark_processed(person_dir):
    with open(os.path.join(person_dir, '.processed'), 'w') as f:
        f.write('done')

def is_processed(person_dir):
    return os.path.exists(os.path.join(person_dir, '.processed'))

def get_pending_person_folders():
    """Return ready person folders sorted oldest-first by .ready file time."""
    if not os.path.isdir(RECOGNITION_FOLDER):
        return []

    pending = []
    for person_folder in os.listdir(RECOGNITION_FOLDER):
        if not person_folder.startswith('person_'):
            continue
        person_dir = os.path.join(RECOGNITION_FOLDER, person_folder)
        ready_file = os.path.join(person_dir, '.ready')
        if not os.path.isdir(person_dir) or is_processed(person_dir) or not os.path.exists(ready_file):
            continue
        pending.append((os.path.getmtime(ready_file), person_folder, person_dir, ready_file))

    pending.sort(key=lambda item: item[0])
    return pending

def main():
    os.makedirs(RECOGNITION_FOLDER, exist_ok=True)
    training_mode = is_training_mode_enabled()
    print(f"Face server: {FACE_SERVER_URL}")
    print(f"Mode: {'training (/extract)' if training_mode else 'search (/search)'}")
    print(f"Watching '{RECOGNITION_FOLDER}' for new person folders...")
    while True:
        try:
            for _, person_folder, person_dir, ready_file in get_pending_person_folders():
                tracking_id = parse_track_id_from_folder(person_folder)
                if not folder_inactive(person_dir):
                    continue  # Wait for inactivity
                print(f"Processing {person_dir}...")
                process_person_folder(person_dir, person_folder, tracking_id)
                mark_processed(person_dir)
                # Remove the .ready file after processing
                try:
                    os.remove(ready_file)
                except Exception as e:
                    print(f"Could not delete .ready file {ready_file}: {e}")
        except Exception as e:
            print(f"Error in main loop: {e}")
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main() 