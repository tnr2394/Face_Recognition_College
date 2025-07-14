import os
import time
import re
import requests
import base64
import cv2
from datetime import datetime
import json
import glob
import random
# Placeholders for API and Discord webhook
URL = "https://apis.lensapp.raoinfo.tech/api"
# HEADERS = {"Authorization": "Bearer <TOKEN>", "Content-Type": "application/json"}
HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0aW1lIjoiMjAyNS0wNC0wMlQxMjozNDo1Ny44ODhaIiwiZXhwaXJlc0luIjoiMjAyNi0wNC0wMlQxMjozNDo1Ny44ODhaIiwiY29tcGFueV9pZCI6MSwiaWF0IjoxNzQzNTk3Mjk3fQ.vWnkVHt4vX9UOiZEY0qMYQWjNBiW_ABipgEoNzc_F-U'
        }
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1394232120267902976/uDzwK0ASy5EEEFtM4w3vmnVvshZQXjszPchmuKg5T8Ia9fsDHyBoa6VErwsPz9CWzIMT"
DISCORD_WEBHOOK_UNRECOGNIZED = DISCORD_WEBHOOK

GOOGLE_SHEET_API_URL = "https://script.google.com/macros/s/AKfycbye9j6E3fmdZ_b636aYcDxyVHtH7_lNgYwJHRieec1XCdL9nZlqgAWmyC2ib92qjgSz3g/exec"

RECOGNITION_FOLDER = "recognition_folder"
INACTIVITY_WAIT = 5  # seconds to wait for folder inactivity
CHECK_INTERVAL = 3   # seconds between folder checks
TOP_N = 10

COUNTS_FILE = 'recognition_counts.json'

def load_counts():
    if not os.path.exists(COUNTS_FILE):
        return {'recognized': 0, 'unrecognized': 0}
    with open(COUNTS_FILE, 'r') as f:
        return json.load(f)

def save_counts(counts):
    with open(COUNTS_FILE, 'w') as f:
        json.dump(counts, f)

def resize_image_to_160(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return image_path  # fallback to original if read fails
    resized = cv2.resize(img, (160, 160))
    base, ext = os.path.splitext(image_path)
    resized_path = base + '_resized' + ext
    cv2.imwrite(resized_path, resized)
    return resized_path

def recognize_face(image_path):
    try:
        print("Recognizing face...", image_path)
        url = f"{URL}/attendance/add"
        headers = HEADERS
        base64_image = ""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        data = { 'image': base64_image, 'id': 1 }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            distance_image = result.get('distance_image')
            print("API Response Data:", data, distance_image)
            if data:
                fname, lname = data.get('first_name', ''), data.get('last_name', '')
                return { "status": True, "id": data.get("id"), "name": f"{fname} {lname}", "distance_image": distance_image, 'image_path': image_path}
            else:
                return { "status": False, 'image_path': image_path }
        else:
            print("Error in API call:", response.status_code, response.text)
            return { "status": False, 'image_path': image_path}
    except Exception as e:
        print("Error in recognize_face:", e)
        return { "status": False, 'image_path': image_path}

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

    # --- Overlay recognized/unrecognized counts ---
    counts = load_counts()
    label = f"Recognized: {counts['recognized']}  Unrecognized: {counts['unrecognized']}"
    if img is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        margin = 10
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = width - text_w - margin
        y = text_h + margin
        cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0,0,0), -1)
        cv2.putText(img, label, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        # Save to a temp file for Discord
        overlay_path = display_path + '_overlay.jpg'
        cv2.imwrite(overlay_path, img)
        display_path = overlay_path

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
    # Clean up temp overlay file
    if img is not None and os.path.exists(display_path) and display_path.endswith('_overlay.jpg'):
        try:
            os.remove(display_path)
        except Exception as e:
            print(f"Could not delete temp overlay file {display_path}: {e}")
    # --- Send to Google Sheets ---
    # For Google Sheets: do not include tracking_id in timestamp
    sheet_data = {
        "name": result.get("name", "Unknown"),
        "status": "recognized" if result["status"] else "unrecognized",
        "score": score,
        "image_filename": os.path.basename(display_path)
    }
    send_to_google_sheets(sheet_data)

def parse_score(filename):
    match = re.search(r'_score_([0-9]+(?:\.[0-9]+)?)', filename)
    return float(match.group(1)) if match else 0.0

def get_top_images(person_dir, top_n=3):
    # Get full frame images (for Discord display) that have scores
    full_frame_images = [f for f in os.listdir(person_dir)
                        if f.lower().endswith(('.jpg', '.png')) and '_full_score_' in f and '_resized' not in f]
    
    # If no full frame images with scores, fall back to regular images
    if not full_frame_images:
        images = [f for f in os.listdir(person_dir)
                  if f.lower().endswith(('.jpg', '.png')) and '_resized' not in f and '_full' not in f]
        scored = [(img, parse_score(img)) for img in images]
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Sorted images by score: {scored}")  # Debug line
        return [os.path.join(person_dir, img) for img, _ in scored[:top_n]]
    
    # Sort full frame images by score
    scored = [(img, parse_score(img)) for img in full_frame_images]
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] Sorted full frame images by score: {scored}")  # Debug line
    return [os.path.join(person_dir, img) for img, _ in scored[:top_n]]

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

def main():
    print(f"Watching '{RECOGNITION_FOLDER}' for new person folders...")
    while True:
        try:
            for person_folder in os.listdir(RECOGNITION_FOLDER):
                if not person_folder.startswith('person_'):
                    continue
                person_dir = os.path.join(RECOGNITION_FOLDER, person_folder)
                tracking_id = None
                try:
                    tracking_id = int(person_folder.replace('person_', ''))
                except Exception:
                    pass
                ready_file = os.path.join(person_dir, '.ready')
                if not os.path.isdir(person_dir) or is_processed(person_dir) or not os.path.exists(ready_file):
                    continue
                if not folder_inactive(person_dir):
                    continue  # Wait for inactivity
                print(f"Processing {person_dir}...")
                top_images = get_top_images(person_dir, TOP_N)
                recognized = False
                unrecognized_result = None
                unrecognized_resized_path = None
                unrecognized_tracking_id = None
                for idx, img_path in enumerate(top_images):
                    # Check if this is a full frame image
                    is_full_frame = '_full_score_' in img_path
                    
                    if is_full_frame:
                        # For full frame images, find the corresponding cropped face for recognition
                        match = re.match(r'.*frame_(\d+)_full_score_.*', os.path.basename(img_path))
                        cropped_face_path = None
                        if match:
                            frame_num = match.group(1)
                            # Find any file like frame_{frame_num}_score_*.jpg (not _full_)
                            pattern = os.path.join(person_dir, f"frame_{frame_num}_score_*.jpg")
                            candidates = [f for f in glob.glob(pattern) if '_full' not in f]
                            if candidates:
                                cropped_face_path = candidates[0]
                        if cropped_face_path and os.path.exists(cropped_face_path):
                            # Use cropped face for recognition, full frame for display
                            resized_cropped_path = resize_image_to_160(cropped_face_path)
                            result = recognize_face(resized_cropped_path)
                            result['orig_image_path'] = cropped_face_path
                            result['image_path'] = resized_cropped_path
                            result['display_image_path'] = img_path  # Full frame for Discord display
                        else:
                            # Fallback: use full frame for both recognition and display
                            print(f"[DEBUG] No cropped face found for {img_path}")
                            resized_path = resize_image_to_160(img_path)
                            result = recognize_face(resized_path)
                            result['orig_image_path'] = img_path
                            result['image_path'] = resized_path
                            result['display_image_path'] = img_path
                    else:
                        # For regular cropped face images
                        resized_path = resize_image_to_160(img_path)
                        result = recognize_face(resized_path)
                        result['orig_image_path'] = img_path
                        result['image_path'] = resized_path
                        result['display_image_path'] = img_path
                    
                    if result['status']:
                        # Update recognized count
                        counts = load_counts()
                        counts['recognized'] += 1
                        save_counts(counts)
                        send_to_discord(result, webhook_url=DISCORD_WEBHOOK, tracking_id=tracking_id)
                        recognized = True
                        # Clean up resized image
                        resized_path = result['image_path']
                        if resized_path != result['orig_image_path']:
                            try:
                                os.remove(resized_path)
                            except Exception as e:
                                print(f"Could not delete temp file {resized_path}: {e}")
                        break  # Stop processing further images if recognized
                    else:
                        # Save the first unrecognized result for Discord
                        if idx == 0:
                            unrecognized_result = result
                            unrecognized_resized_path = result['image_path']
                            unrecognized_tracking_id = tracking_id
                        else:
                            # Clean up resized image for other unrecognized
                            resized_path = result['image_path']
                            if resized_path != result['orig_image_path']:
                                try:
                                    os.remove(resized_path)
                                except Exception as e:
                                    print(f"Could not delete temp file {resized_path}: {e}")
                
                if not recognized and unrecognized_result is not None:
                    # Update unrecognized count
                    counts = load_counts()
                    counts['unrecognized'] += 1
                    save_counts(counts)
                    send_to_discord(unrecognized_result, webhook_url=DISCORD_WEBHOOK_UNRECOGNIZED, tracking_id=unrecognized_tracking_id)
                    # Clean up resized image
                    if unrecognized_resized_path and unrecognized_resized_path != unrecognized_result['orig_image_path']:
                        try:
                            os.remove(unrecognized_resized_path)
                        except Exception as e:
                            print(f"Could not delete temp file {unrecognized_resized_path}: {e}")
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