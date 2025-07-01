import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO
from collections import defaultdict
import torch
import onnxruntime as ort
import threading
import time

# Live Face Processor Added

class OFIQScorer:
    """Scores face quality using a pre-trained OFIQ ONNX model."""
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        providers = ['CUDAExecutionProvider'] if device == 'cuda' and ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)
        self.recognition_folder = 'recognition_folder'

    def _preprocess(self, face_img):
        """Preprocesses a single face image (numpy array) for the ONNX model."""
        img = cv2.resize(face_img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        return img

    def get_score(self, face_img):
        """Calculates the quality score for a single face image."""
        if face_img is None or face_img.size == 0:
            return 0.0
        try:
            preprocessed_img = self._preprocess(face_img)
            output = self.session.run(None, {self.input_name: preprocessed_img})
            score = output[0].item()
            return score
        except Exception as e:
            print(f"Error scoring image: {e}")
            return 0.0

class LiveFaceProcessor:
    """Processes a video in real-time to track, save, and score faces."""

    def __init__(self, yolo_model_path='models/yolov8n-face.pt', ofiq_model_path='OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx', save_dir='live_results'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.yolo_model = YOLO(yolo_model_path)
        self.scorer = OFIQScorer(ofiq_model_path, device=self.device)
        self.save_dir = save_dir
        self.tracked_faces = defaultdict(list)
        self.active_track_ids = set()

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def _process_lost_tracks(self, lost_track_ids):
        """Process folders of tracks that are no longer active."""
        for track_id in lost_track_ids:
            print(f"Person with ID {track_id} has left. Processing their folder.")
            person_dir = os.path.join(self.save_dir, f"person_{track_id}")
            # Run processing in a separate thread to avoid blocking the main loop
            thread = threading.Thread(target=self._score_and_rename_faces, args=(person_dir,))
            thread.start()

    def _score_and_rename_faces(self, person_dir):
        """Scores all faces in a directory and renames them with the score."""
        if not os.path.isdir(person_dir):
            return

        # Get only cropped face images (not full frames) for scoring
        face_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png')) and '_full' not in f]
        
        for filename in face_files:
            img_path = os.path.join(person_dir, filename)
            face_img = cv2.imread(img_path)
            if face_img is not None:
                score = self.scorer.get_score(face_img)
                new_filename = f"{os.path.splitext(filename)[0]}_score_{score:.2f}.jpg"
                new_path = os.path.join(person_dir, new_filename)
                
                # Check if corresponding full frame exists
                frame_num = filename.replace('frame_', '').replace('.jpg', '')
                full_frame_filename = f"frame_{frame_num}_full.jpg"
                full_frame_path = os.path.join(person_dir, full_frame_filename)
                
                try:
                    # if score > 15:
                        # For high quality faces: move both cropped face and full frame to recognition folder
                    person_folder_name = os.path.basename(person_dir)
                    recognition_person_dir = os.path.join(self.scorer.recognition_folder, person_folder_name)
                    os.makedirs(recognition_person_dir, exist_ok=True)
                    
                    # Move cropped face with score
                    rec_face_path = os.path.join(recognition_person_dir, new_filename)
                    os.rename(img_path, rec_face_path)
                    
                    # Move full frame with score (for Discord display)
                    if os.path.exists(full_frame_path):
                        full_frame_new_filename = f"frame_{frame_num}_full_score_{score:.2f}.jpg"
                        rec_full_frame_path = os.path.join(recognition_person_dir, full_frame_new_filename)
                        os.rename(full_frame_path, rec_full_frame_path)
                        print(f"Moved {filename} and {full_frame_filename} to recognition folder with score {score:.2f}")
                    else:
                        print(f"Moved {filename} to recognition folder as {new_filename}")
                    # else:
                    #     # For lower quality faces: just rename with score in original location
                    #     os.rename(img_path, new_path)
                    #     if os.path.exists(full_frame_path):
                    #         full_frame_new_filename = f"frame_{frame_num}_full_score_{score:.2f}.jpg"
                    #         full_frame_new_path = os.path.join(person_dir, full_frame_new_filename)
                    #         os.rename(full_frame_path, full_frame_new_path)
                    #     print(f"Renamed {filename} to {new_filename}")
                except OSError as e:
                    print(f"Error renaming file {img_path}: {e}")
            time.sleep(0.01) # Small delay to yield CPU

    def process_video(self, video_source):
        """Main loop to process video stream from a file or camera."""
        results_generator = self.yolo_model.track(source=video_source, stream=True, persist=True, show=True, device=self.device)
        frame_num = 0

        try:
            for results in results_generator:
                frame_num += 1
                current_frame_track_ids = set()

                if results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results.boxes.id.cpu().numpy().astype(int)
                    current_frame_track_ids.update(track_ids)

                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        face_img = results.orig_img[y1:y2, x1:x2]
                        cv2.rectangle(results.orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(results.orig_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Create a directory for the person if it doesn't exist
                        person_dir = os.path.join(self.save_dir, f"person_{track_id}")
                        os.makedirs(person_dir, exist_ok=True)

                        # Save the cropped face image for scoring
                        face_filename = f"frame_{frame_num}.jpg"
                        face_path = os.path.join(person_dir, face_filename)
                        cv2.imwrite(face_path, face_img)
                        
                        # Save the whole frame with face marked for Discord display
                        frame_filename = f"frame_{frame_num}_full.jpg"
                        frame_path = os.path.join(person_dir, frame_filename)
                        cv2.imwrite(frame_path, results.orig_img)
                        
                # Detect which tracks were lost in this frame
                lost_track_ids = self.active_track_ids - current_frame_track_ids
                if lost_track_ids:
                    self._process_lost_tracks(lost_track_ids)

                # Update the set of active tracks
                self.active_track_ids = current_frame_track_ids

        except (KeyboardInterrupt, StopIteration):
            print("\nProcessing stopped.")
        finally:
            # Process any remaining active tracks after the video ends
            print("Video finished. Processing remaining tracks...")
            if self.active_track_ids:
                self._process_lost_tracks(self.active_track_ids)
            print("All processing complete.")

if __name__ == '__main__':
    # Use 0 for webcam, or provide a path to a video file
    video_source = r'C:\Users\raoit\Downloads\Unknown.mp4'
    # video_source = 0 # Uncomment for webcam

    if not isinstance(video_source, int) and not os.path.exists(video_source):
        print(f"Error: Video file not found at {video_source}")
    else:
        processor = LiveFaceProcessor()
        processor.process_video(video_source)
        print(f"\nLive processing finished. Results are in '{os.path.abspath(processor.save_dir)}'.")
