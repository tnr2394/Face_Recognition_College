import cv2, numpy as np, os, torch, time, shutil
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
from recognition_on_images import queue_process

class FaceDetectionRanker:
    def __init__(self, model_path='yolov8n-face-landmarks.pt', save_dir='ranked_faces'):
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists('sended_persons'): shutil.rmtree('sended_persons')
        os.makedirs('sended_persons', exist_ok=True)
        self.all_faces_dir = os.path.join(save_dir, "all_faces"); os.makedirs(self.all_faces_dir, exist_ok=True)
        self.top_faces_dir = os.path.join(save_dir, "top_faces"); os.makedirs(self.top_faces_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "debug"), exist_ok=True)
        
        self.iou_threshold, self.distance_threshold = 0.2, 150
        self.max_frames_missing, self.min_face_size = 45, 50
        self.min_track_length, self.max_faces_per_id = 5, 5
        self.pose_change_tolerance = 0.6
        self.edge_margin = 20
        self.quality_threshold = {'blur': 60, 'completeness': 70, 'eyes': 60, 'frontality': 65, 'mouth': 70, 'combined': 60}
        self.score_weights = {'blur': 0.15, 'lighting': 0.1, 'size': 0.1, 'completeness': 0.2, 'frontality': 0.2, 'eyes': 0.15, 'mouth': 0.1}
        
        self.active_tracks, self.completed_tracks = {}, {}
        self.next_id, self.frame_count = 0, 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def detect_faces(self, frame):
        return self.model.predict(frame, conf=0.78, device=self.device, iou=0.35)[0]

    def calculate_blur_score(self, img):
        if img.size == 0: return 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_lighting_score(self, img):
        if img.size == 0: return 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        return (1 - abs(mean_brightness - 127) / 255) * 70 + (std_brightness / 128) * 30

    def calculate_box_size_score(self, w, h, fw, fh):
        size_pct = (w * h) / (fw * fh)
        if size_pct < 0.005: return size_pct * 20000
        elif size_pct > 0.5: return (1 - size_pct) * 200
        else: return 100 * size_pct * 2

    def calculate_face_descriptor(self, img):
        if img.size == 0: return np.zeros(10)
        try:
            face_small = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(face_small, cv2.COLOR_BGR2GRAY)
            hist = cv2.normalize(cv2.calcHist([gray], [0], None, [8], [0, 256]), None).flatten()
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.count_nonzero(edges) / (64*64)
            return np.append(hist, edge_count)
        except: return np.zeros(10)

    def compare_face_descriptors(self, d1, d2):
        try:
            score = np.corrcoef(d1, d2)[0, 1]
            return 0 if np.isnan(score) else max(0, score)
        except: return 0

    def check_face_completeness(self, face_img):
        """More aggressive half-face detection"""
        try:
            if face_img.size == 0: return 0
            face = cv2.resize(face_img, (64, 64))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Symmetry check
            left = gray[:, :32]
            right = np.fliplr(gray[:, 32:])
            symmetry = np.corrcoef(left.flatten(), right.flatten())[0, 1]
            if np.isnan(symmetry): symmetry = 0
            
            # Edge density analysis
            edges = cv2.Canny(gray, 100, 200)
            left_density = np.count_nonzero(edges[:, :32]) / (32*64)
            right_density = np.count_nonzero(edges[:, 32:]) / (32*64)
            
            # Calculate balance 
            balance = min(left_density, right_density) / max(left_density, right_density) if left_density + right_density > 0 else 0
            
            # Center of mass
            y, x = np.mgrid[0:64, 0:64]
            center_x = np.sum(x * gray) / np.sum(gray) if np.sum(gray) > 0 else 32
            center_score = 1 - abs(center_x - 32) / 32
            
            # Combine metrics
            score = (symmetry * 0.35 + balance * 0.3 + center_score * 0.35) * 100
            
            # Apply penalty if extremely asymmetric
            if balance < 0.3 or symmetry < 0.3: score *= 0.7
            
            return max(0, min(100, score))
        except: return 0

    def calculate_frontality_score(self, img, landmarks):
        try:
            if img.size == 0: return 0
            if landmarks is None or len(landmarks) < 4:
                # Fallback to symmetry-based estimation
                face = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                left_half = gray[:, :32]
                right_half = np.fliplr(gray[:, 32:])
                symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if np.isnan(symmetry): symmetry = 0
                h, w = img.shape[:2]
                aspect_score = 1.0 - min(1.0, abs((w/h if h > 0 else 0) - 0.8) / 0.4)
                return (symmetry * 0.7 + aspect_score * 0.3) * 100
            
            # With landmarks, calculate based on eye positions
            left_eye = (landmarks[0], landmarks[1])
            right_eye = (landmarks[2], landmarks[3])
            
            # Eye alignment horizontally vs vertically
            eye_dx = abs(right_eye[0] - left_eye[0])
            eye_dy = abs(right_eye[1] - left_eye[1])
            horizontal_alignment = max(0, 1.0 - (eye_dy / eye_dx if eye_dx > 0 else 1.0))
            
            # Eye center offset from face center
            h, w = img.shape[:2]
            eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
            center_alignment = max(0, 1.0 - abs(eye_midpoint_x - (w/2)) / (w/2))
            
            # Eye distance relative to face width
            eye_distance = abs(right_eye[0] - left_eye[0])
            distance_ratio = eye_distance / w
            distance_score = 1.0 - min(1.0, abs(distance_ratio - 0.43) / 0.2)
            
            # Combined score
            return min(100, max(0, (horizontal_alignment * 0.4 + center_alignment * 0.4 + distance_score * 0.2) * 100))
        except: return 50

    def detect_open_eyes(self, img, landmarks):
        try:
            if img.size == 0: return 0
            
            # Calculate regions
            if landmarks is None or len(landmarks) < 4:
                h, w = img.shape[:2]
                left_eye_region = img[int(h*0.2):int(h*0.45), int(w*0.15):int(w*0.45)]
                right_eye_region = img[int(h*0.2):int(h*0.45), int(w*0.55):int(w*0.85)]
                
                if left_eye_region.size == 0 or right_eye_region.size == 0: return 50
            else:
                h, w = img.shape[:2]
                eye_width = int(abs(landmarks[2] - landmarks[0]) * 0.3)
                eye_height = int(eye_width * 0.5)
                
                # Extract eye regions using landmarks
                left_x1 = max(0, int(landmarks[0] - eye_width/2))
                left_y1 = max(0, int(landmarks[1] - eye_height/2))
                left_x2 = min(w, int(landmarks[0] + eye_width/2))
                left_y2 = min(h, int(landmarks[1] + eye_height/2))
                
                right_x1 = max(0, int(landmarks[2] - eye_width/2))
                right_y1 = max(0, int(landmarks[3] - eye_height/2))
                right_x2 = min(w, int(landmarks[2] + eye_width/2))
                right_y2 = min(h, int(landmarks[3] + eye_height/2))
                
                left_eye_region = img[left_y1:left_y2, left_x1:left_x2]
                right_eye_region = img[right_y1:right_y2, right_x1:right_x2]
                
                if left_eye_region.size == 0 or right_eye_region.size == 0:
                    # Fallback if regions are invalid
                    return (np.sqrt((landmarks[0]-landmarks[2])**2 + (landmarks[1]-landmarks[3])**2) / w) * 500
            
            # Calculate variance (contrast) in eye regions - higher for open eyes
            left_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
            left_var = np.var(left_gray)
            right_var = np.var(right_gray)
            
            # Edge detection (more edges in open eyes)
            left_edges = cv2.Canny(left_gray, 50, 150)
            right_edges = cv2.Canny(right_gray, 50, 150)
            left_edge_density = np.count_nonzero(left_edges) / left_edges.size
            right_edge_density = np.count_nonzero(right_edges) / right_edges.size
            
            # Combine metrics
            var_score = min(100, max(0, ((left_var + right_var) / 2) * 0.4))
            edge_score = min(100, max(0, ((left_edge_density + right_edge_density) / 2) * 1000))
            
            return min(100, max(0, var_score * 0.7 + edge_score * 0.3))
        except: return 50

    def detect_closed_mouth(self, face_img):
        """
        Estimates if the mouth is closed using image analysis.
        A higher score indicates a higher probability of a closed mouth.
        This is a heuristic and works best on well-lit, frontal faces.
        """
        if face_img.size == 0: return 0
        try:
            h, w, _ = face_img.shape
            # Define ROI for the mouth area (lower part of the face)
            mouth_roi_y_start = int(h * 0.65)
            mouth_roi_y_end = int(h * 0.95)
            mouth_roi_x_start = int(w * 0.25)
            mouth_roi_x_end = int(w * 0.75)
            
            mouth_roi = face_img[mouth_roi_y_start:mouth_roi_y_end, mouth_roi_x_start:mouth_roi_x_end]
            
            if mouth_roi.size == 0: return 50 # Neutral score if ROI is empty

            gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to find dark regions (potential open mouth)
            binary = cv2.adaptiveThreshold(gray_mouth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Calculate the percentage of dark pixels
            dark_pixel_percentage = np.sum(binary == 255) / binary.size
            
            # Score is inversely proportional to the percentage of dark pixels
            # We map the percentage to a 0-100 score.
            # If 0% dark pixels, score is 100. If >15% dark pixels, score is 0.
            score = 100 * (1 - min(1, dark_pixel_percentage / 0.15))
            
            return score
        except Exception as e:
            # print(f"Mouth detection error: {e}")
            return 50 # Return a neutral score on error

    def align_face(self, img, landmarks):
        try:
            if landmarks is None or len(landmarks) < 4: return img
            left_eye = np.array([landmarks[0], landmarks[1]])
            right_eye = np.array([landmarks[2], landmarks[3]])
            angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        except: return img

    def improved_align_face(self, face_img, landmarks):
        """
        Improved face alignment with safety checks to prevent cropping and inversion issues.
        """
        try:
            # If no landmarks or not enough landmarks, return original
            if landmarks is None or len(landmarks) < 4:
                return face_img
            
            # Get image dimensions
            h, w = face_img.shape[:2]
            if h == 0 or w == 0:
                return face_img  # Safety check for empty images
            
            # Extract eye landmarks
            left_eye = np.array([landmarks[0], landmarks[1]])
            right_eye = np.array([landmarks[2], landmarks[3]])
            
            # Validate landmarks are within image
            if (not (0 <= left_eye[0] < w and 0 <= left_eye[1] < h) or
                not (0 <= right_eye[0] < w and 0 <= right_eye[1] < h)):
                return face_img  # Invalid landmarks, return original
            
            # Calculate eye center positions
            left_eye_center = left_eye.astype(np.int32)
            right_eye_center = right_eye.astype(np.int32)
            
            # Calculate angle between eyes
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            
            # Get angle in degrees
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Sanity check on angle - if extreme, return original
            if abs(angle) > 30:
                return face_img  # Angle too extreme, likely bad landmarks
            
            # Add padding to prevent cropping after rotation
            padding = int(0.2 * max(h, w))  # 20% padding
            padded_img = np.zeros((h + 2*padding, w + 2*padding, 3), dtype=face_img.dtype)
            padded_img[padding:padding+h, padding:padding+w] = face_img
            
            # Adjust the center point for padded image
            center = (w//2 + padding, h//2 + padding)
            
            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to padded image
            rotated_img = cv2.warpAffine(
                padded_img, 
                rotation_matrix, 
                (padded_img.shape[1], padded_img.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # Crop back to original size (centered)
            y1 = padded_img.shape[0]//2 - h//2
            y2 = y1 + h
            x1 = padded_img.shape[1]//2 - w//2
            x2 = x1 + w
            
            # Ensure crop region is within bounds
            y1 = max(0, min(y1, padded_img.shape[0] - 1))
            y2 = max(0, min(y2, padded_img.shape[0]))
            x1 = max(0, min(x1, padded_img.shape[1] - 1))
            x2 = max(0, min(x2, padded_img.shape[1]))
            
            aligned_img = rotated_img[y1:y2, x1:x2]
            
            # Verify result isn't significantly smaller than input
            if aligned_img.size > 0:
                result_h, result_w = aligned_img.shape[:2]
                if result_h < 0.8*h or result_w < 0.8*w:
                    return face_img  # Too much content lost, return original
                
                return aligned_img
            else:
                return face_img
        except Exception as e:
            print(f"Error in face alignment: {e}")
            return face_img  # Return original on error

    def save_aligned_face(self, face_img, landmarks, output_filename):
        """
        Safely align and save a face with validation and original backup.
        """
        try:
            # Save original face for comparison
            original_filename = output_filename.replace('.jpg', '_original.jpg')
            # cv2.imwrite(original_filename, face_img)
            
            # Align face
            aligned_face = self.improved_align_face(face_img, landmarks)
            
            # Verify alignment result
            if aligned_face is not face_img:  # If alignment returned a new image
                # Get face dimensions
                orig_h, orig_w = face_img.shape[:2]
                aligned_h, aligned_w = aligned_face.shape[:2]
                
                # Verify dimensions haven't changed too much
                if (0.7 <= aligned_h/orig_h <= 1.3 and 
                    0.7 <= aligned_w/orig_w <= 1.3):
                    # Dimensions are acceptable
                    img_resized = cv2.resize(aligned_face, (160, 160))
                    cv2.imwrite(output_filename, img_resized)
                    return True
                else:
                    # Dimensions changed too much, use original
                    img_resized = cv2.resize(face_img, (160, 160))
                    cv2.imwrite(output_filename, img_resized)
                    print(f"Warning: Used original face due to alignment issues: {output_filename}")
                    return False
            else:
                # Alignment returned original, use it
                img_resized = cv2.resize(face_img, (160, 160))
                cv2.imwrite(output_filename, img_resized)
                return False
        except Exception as e:
            print(f"Error saving aligned face: {e}")
            # In case of error, save original
            try:
                img_resized = cv2.resize(face_img, (160, 160))
                cv2.imwrite(output_filename, img_resized)
            except:
                pass
            return False

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
        x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i: return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        return intersection / (area1 + area2 - intersection) if (area1 + area2 - intersection) > 0 else 0

    def check_edge_proximity(self, box, frame_w, frame_h):
        """More aggressively penalize faces near frame edges"""
        x1, y1, x2, y2 = box
        min_dist = min(x1, frame_w-x2, y1, frame_h-y2)
        return 0.5 if min_dist <= 10 else (0.7 if min_dist <= self.edge_margin else 1.0)

    def update_tracks(self, frame, detections, video_path):
        h, w = frame.shape[:2]
        current_detections = []
        
        if hasattr(detections, 'boxes') and len(detections.boxes) > 0:
            for i, box in enumerate(detections.boxes):
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_w, box_h = x2-x1, y2-y1
                    if box_w < self.min_face_size or box_h < self.min_face_size: continue
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    edge_factor = self.check_edge_proximity((x1, y1, x2, y2), w, h)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    landmarks = None
                    if hasattr(detections, 'keypoints') and detections.keypoints is not None:
                        try:
                            kpts = detections.keypoints[i].xy[0].cpu().numpy()
                            landmarks = [float(kpts[0][0]), float(kpts[0][1]), float(kpts[1][0]), float(kpts[1][1])]
                        except: pass
                    
                    face_img = frame[y1:y2, x1:x2].copy()
                    completeness = self.check_face_completeness(face_img)
                    eye_openness = self.detect_open_eyes(face_img, landmarks)
                    frontality = self.calculate_frontality_score(face_img, landmarks)
                    mouth_openness = self.detect_closed_mouth(face_img) # New score
                    descriptor = self.calculate_face_descriptor(face_img)
                    blur = self.calculate_blur_score(face_img)
                    lighting = self.calculate_lighting_score(face_img)
                    size = self.calculate_box_size_score(box_w, box_h, w, h)
                    
                    # Combined score is now calculated in process_completed_track
                    combined = 0 # Placeholder
                    
                    current_detections.append({
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'face_img': face_img,
                        'descriptor': descriptor,
                        'blur': blur,
                        'lighting': lighting,
                        'size': size,
                        'completeness': completeness,
                        'eyes': eye_openness,
                        'frontality': frontality,
                        'mouth': mouth_openness,
                        'combined': combined,
                        'landmarks': landmarks,
                        'matched': False,
                        'frame_num': self.frame_count
                    })
                except: continue
        
        # Update existing tracks
        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]
            track['frames_since_seen'] += 1
            
            if track['frames_since_seen'] > self.max_frames_missing:
                if len(track['faces']) >= self.min_track_length:
                    self.completed_tracks[track_id] = track
                    self.process_completed_track(track_id, video_path)
                del self.active_tracks[track_id]
                continue
        
        # Match detections to tracks
        sorted_track_ids = sorted(self.active_tracks.keys(), key=lambda id: len(self.active_tracks[id]['faces']), reverse=True)
        
        for track_id in sorted_track_ids:
            track = self.active_tracks[track_id]
            if track['frames_since_seen'] == 0: continue
            
            best_match, best_score = None, -float('inf')
            last_box = track['boxes'][-1]
            last_center = track['centers'][-1]
            last_desc = track['faces'][-1].get('descriptor', None)
            
            for i, detection in enumerate(current_detections):
                if detection['matched']: continue
                
                iou = self.calculate_iou(last_box, detection['box'])
                curr_center = detection['center']
                distance = np.sqrt((last_center[0] - curr_center[0])**2 + (last_center[1] - curr_center[1])**2)
                norm_distance = 1.0 - min(1.0, distance / self.distance_threshold)
                
                appearance_similarity = 0
                if last_desc is not None:
                    appearance_similarity = self.compare_face_descriptors(last_desc, detection['descriptor'])
                
                match_score = iou * 0.3 + norm_distance * 0.3 + appearance_similarity * 0.4
                
                is_valid = (iou > self.iou_threshold or 
                           distance < self.distance_threshold or
                           (distance < self.distance_threshold * 1.5 and 
                            appearance_similarity > self.pose_change_tolerance))
                
                if is_valid and match_score > best_score:
                    best_match, best_score = i, match_score
            
            if best_match is not None:
                detection = current_detections[best_match]
                detection['matched'] = True
                
                track['boxes'].append(detection['box'])
                track['centers'].append(detection['center'])
                track['frames_since_seen'] = 0
                track['faces'].append({
                    'frame_num': self.frame_count,
                    'box': detection['box'],
                    'face_img': detection['face_img'],
                    'descriptor': detection['descriptor'],
                    'blur': detection['blur'],
                    'lighting': detection['lighting'],
                    'size': detection['size'],
                    'completeness': detection['completeness'],
                    'eyes': detection['eyes'],
                    'frontality': detection['frontality'],
                    'mouth': detection['mouth'],
                    'combined': detection['combined'],
                    'landmarks': detection['landmarks']
                })
        
        # Create new tracks for unmatched detections
        for detection in current_detections:
            if not detection['matched']:
                self.active_tracks[self.next_id] = {
                    'track_id': self.next_id,
                    'boxes': [detection['box']],
                    'centers': [detection['center']],
                    'frames_since_seen': 0,
                    'faces': [{
                        'frame_num': self.frame_count,
                        'box': detection['box'],
                        'face_img': detection['face_img'],
                        'descriptor': detection['descriptor'],
                        'blur': detection['blur'],
                        'lighting': detection['lighting'],
                        'size': detection['size'],
                        'completeness': detection['completeness'],
                        'eyes': detection['eyes'],
                        'frontality': detection['frontality'],
                        'mouth': detection['mouth'],
                        'combined': detection['combined'],
                        'landmarks': detection['landmarks']
                    }],
                    'start_frame': self.frame_count
                }
                self.next_id += 1

    def check_face_validity(self, face):
        comp = face.get('completeness', 0)
        eyes = face.get('eyes', 0)
        front = face.get('frontality', 0)
        blur = face.get('blur', 0)
        
        # Check against minimum thresholds
        if (comp < self.quality_threshold['completeness'] or
            eyes < self.quality_threshold['eyes'] or
            front < self.quality_threshold['frontality'] or
            blur < self.quality_threshold['blur']):
            return False
            
        # Additional checks for edge cases
        if comp < 60 and front < 60:
            return False
        
        if blur > 50:
            return True
        else:
            return False
        

    def is_half_face(self, face_img, box, frame_w, frame_h):
        """
        Direct detection of half-faces using multiple techniques.
        Returns True if image is likely a half-face, False otherwise.
        """
        try:
            if face_img.size == 0:
                return True  # Empty image
            
            x1, y1, x2, y2 = box
            face_w, face_h = x2 - x1, y2 - y1
            
            # Half-face detection logic
            boundary_margin = 0.05
            min_margin_px = max(10, int(min(frame_w, frame_h) * boundary_margin))
            
            if (x1 < min_margin_px or y1 < min_margin_px or 
                frame_w - x2 < min_margin_px or frame_h - y2 < min_margin_px):
                return True
            
            # Aspect ratio check
            aspect_ratio = face_w / face_h if face_h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 1.2:
                return True
            
            # Symmetry check
            gray = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
            left_half = gray[:, :32]
            right_half = np.fliplr(gray[:, 32:])
            symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if np.isnan(symmetry) or symmetry < 0.5:
                return True
            
            # Edge density check
            edges = cv2.Canny(gray, 100, 200)
            border_size = 5
            edge_regions = [
                edges[:, :border_size],
                edges[:, -border_size:],
                edges[:border_size, :],
                edges[-border_size:, :]
            ]
            edge_densities = [np.count_nonzero(region) / region.size for region in edge_regions]
            if max(edge_densities) > 0.25:
                return True
            
            # Face center position check
            face_center_x = x1 + face_w / 2
            relative_x = face_center_x / frame_w
            if relative_x < 0.25 or relative_x > 0.75:
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in half-face detection: {e}")
            return True

    def process_completed_track(self, track_id, video_path):
        """
        Strictly filter out half-faces before ranking and selecting top faces.
        """
        track = self.completed_tracks[track_id]
        if len(track['faces']) < self.min_track_length:
            return
        
        # Setup directories and frame dimensions
        person_all_dir = os.path.join(self.all_faces_dir, f"person_{track_id}")
        person_top_dir = os.path.join(self.top_faces_dir, f"person_{track_id}")
        os.makedirs(person_all_dir, exist_ok=True)
        os.makedirs(person_top_dir, exist_ok=True)
        
        frame_w, frame_h = 640, 480
        if track['faces'] and 'box' in track['faces'][0]:
            box = track['faces'][0]['box']
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            frame_w = max(640, int(box[2] * 1.5)) if x_center > box_w else int(box[2] * 3)
            frame_h = max(480, int(box[3] * 1.5)) if y_center > box_h else int(box[3] * 3)
        
        # Filter faces and save with classification
        full_faces = []
        half_faces = []
        
        for face in track['faces']:
            box = face.get('box', (0, 0, 1, 1))
            if self.is_half_face(face['face_img'], box, frame_w, frame_h):
                half_faces.append(face)
            else:
                full_faces.append(face)
        
        # Process and save faces
        candidate_faces = full_faces
        if len(full_faces) < self.max_faces_per_id:
            sorted_half = sorted(half_faces, key=lambda x: x.get('completeness', 0), reverse=True)
            needed = self.max_faces_per_id - len(full_faces)
            candidate_faces = full_faces + sorted_half[:needed]
        
        # Recalculate combined score for all valid faces using weighted scores
        for face in track['faces']:
            combined_score = 0
            total_weight = 0
            for metric, weight in self.score_weights.items():
                if metric in face:
                    combined_score += face[metric] * weight
                    total_weight += weight
            if total_weight > 0:
                combined_score /= total_weight
            face['combined'] = combined_score
        
        candidate_faces = sorted(candidate_faces, key=lambda x: x.get('combined', 0), reverse=True)
        top_faces = candidate_faces[:self.max_faces_per_id]
        
        # Save faces and create visualizations
        for face in track['faces']:            
            if face['blur'] > 50:
                cv2.imwrite(os.path.join(person_all_dir, f"frame_{face['frame_num']}_score_{face['combined']:.2f}.jpg"), face['face_img'])
        
        # Two-stage filtering: first filter based on validity, then sort by score
        valid_faces = [f for f in track['faces'] if self.check_face_validity(f)]
        
        # If no faces meet strict criteria, fall back to basic sorting
        if len(valid_faces) < 2:
            sorted_faces = sorted(track['faces'], key=lambda x: x['combined'], reverse=True)
        else:
            sorted_faces = sorted(valid_faces, key=lambda x: x['combined'], reverse=True)
        
        # Take top N faces
        top_faces = sorted_faces[:self.max_faces_per_id]
        
        # Final verification - ensure top faces don't include half-faces/closed eyes
        final_faces = []
        for face in top_faces:
            # Extra check for any remaining half-faces
            if face.get('completeness', 0) >= 60 and face.get('eyes', 0) >= 50 and face.get('blur', 0) > 50:
                final_faces.append(face)
        
        # If too few valid faces, add back some from sorted_faces
        if len(final_faces) < 2 and len(sorted_faces) >= 2:
            for face in sorted_faces:
                if face['blur'] > 50:
                    if face not in final_faces:
                        final_faces.append(face)
                        if len(final_faces) >= self.max_faces_per_id:
                            break
        
        # Save top ranked faces
        for rank, face in enumerate(final_faces[:self.max_faces_per_id]):
            img = face['face_img']
            landmarks = face['landmarks']
            
            filename = os.path.join(person_top_dir, 
                                  f"rank_{rank+1}_frame_{face.get('frame_num', 0)}_score_{face.get('combined', 0):.2f}.jpg")
            
            # Use safe alignment and saving
            if landmarks is not None:
                self.save_aligned_face(img, landmarks, filename)
            else:
                resized = cv2.resize(img, (160, 160))
                cv2.imwrite(filename, resized)
            
            now = datetime.now()
            formatted_time = now.strftime("At %d %b %H:%M")
            # Save metadata
            with open(filename.replace(".jpg", ".txt"), 'w') as f:
                f.write(f"Frame: {face['frame_num']}\n")
                f.write(f"Blur: {face.get('blur', 0):.2f}\n")
                f.write(f"Lighting: {face.get('lighting', 0):.2f}\n")
                f.write(f"Size: {face.get('size', 0):.2f}\n")
                f.write(f"Completeness: {face.get('completeness', 0):.2f}\n")
                f.write(f"Eyes: {face.get('eyes', 0):.2f}\n")  # Fixed double colon
                f.write(f"Frontality: {face.get('frontality', 0):.2f}\n")
                f.write(f"Combined: {face.get('combined', 0):.2f}\n")
                f.write(f"Box: {face['box']}\n")
                f.write(f"DateTime- {formatted_time}\n")
                f.write(f"Video: {video_path}\n")
        
        queue_process()
        print(f"✅ Person {track_id}: {len(track['faces'])} faces, saved top {len(final_faces[:self.max_faces_per_id])}")

    def track_and_rank_faces(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {frame_width}x{frame_height}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # START PROCESSING ON STREAMS
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            self.frame_count += 1
            # HERE WE CAN DO FACE DETECTION
            detections = self.detect_faces(frame)
            self.update_tracks(frame, detections, video_path)
            
            visualized_frame = frame.copy()
            
            # Visualize active tracks
            for track_id, track in self.active_tracks.items():
                if track['frames_since_seen'] == 0:
                    x1, y1, x2, y2 = track['boxes'][-1]
                    track_length = len(track['faces'])
                    stability = min(1.0, track_length / 20)
                    
                    latest = track['faces'][-1]
                    completeness = latest.get('completeness', 0)
                    eyes = latest.get('eyes', 0)
                    frontality = latest.get('frontality', 0)
                    combined = latest.get('combined', 0)
                    
                    # Color based on quality
                    quality = min(1.0, combined / 100)
                    color = (int(255 * (1 - quality)), int(255 * quality), 0)
                    thickness = max(1, min(3, int(stability * 3)))
                    
                    # cv2.rectangle(visualized_frame, (x1, y1), (x2, y2), color, thickness)
                    # cv2.putText(visualized_frame, f"ID:{track_id} ({track_length})", (x1, y1 - 10), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                    # cv2.putText(visualized_frame, f"C:{int(completeness)} E:{int(eyes)} F:{int(frontality)}", 
                    #             (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
            
            # Add frame counter and info
            # cv2.putText(visualized_frame, f"Frame: {self.frame_count}", (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(visualized_frame, "C: Completeness, E: Eyes, F: Frontality", 
            #             (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(visualized_frame, "Space: Pause/Resume, Q: Quit", 
            #             (frame_width - 280, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            resized_frame = cv2.resize(visualized_frame, (500, 500))
            cv2.imshow("Face Tracking", resized_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Process any remaining active tracks
        for track_id in list(self.active_tracks.keys()):
            self.completed_tracks[track_id] = self.active_tracks[track_id]
            self.process_completed_track(track_id, video_path)
        
        print(f"Processing completed: {len(self.completed_tracks)} unique faces.")

    def run(self, video_path):
        print(f"Processing video: {video_path}")
        start_time = time.time()
        self.track_and_rank_faces(video_path)
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds")
        
        total_people = len(self.completed_tracks)
        total_faces = sum(len(track['faces']) for track in self.completed_tracks.values())
        avg_faces = total_faces / total_people if total_people > 0 else 0
        
        print(f"\nSummary: {total_people} people, {total_faces} faces, {avg_faces:.1f} avg faces/person")
        
        try: os.startfile(self.save_dir)
        except: print(f"Please open results folder: {self.save_dir}")

if __name__ == '__main__':
    # video_path = 0 # for webcam
    video_path = r'd:\AI Work\Face_Recognition_College\test_videos\Knowns.mp4'
    ranker = FaceDetectionRanker()
    ranker.run(video_path)