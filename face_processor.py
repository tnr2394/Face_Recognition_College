import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO
from collections import defaultdict
import torch
# import cv2
import onnxruntime as ort

class OFIQScorer:
    """Scores face quality using a pre-trained OFIQ ONNX model."""
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        providers = ['CUDAExecutionProvider'] if device == 'cuda' and ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)

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
        except Exception:
            return 0.0

class VideoProcessor:
    """Processes a video to track, score, and save the best faces."""

    def __init__(self, model_path='models/yolov8n-face.pt', save_dir='results', top_n=5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initialized VideoProcessor on device: '{self.device}'")
        self.model = YOLO(model_path)
        # Use the OFIQ model for scoring
        ofiq_model_path = "OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx"
        self.scorer = OFIQScorer(ofiq_model_path, device=self.device)
        self.save_dir = save_dir
        self.top_n = top_n
        self.all_faces = defaultdict(list)

    def process_video(self, video_path, output_filename="output_video.mp4"):
        """
        Processes a video to track faces, scores their quality, and saves the annotated
        video to a file. Also saves the best face for each track.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties for VideoWriter
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Save the output video in the same directory as the input video
        output_path = os.path.join(os.path.dirname(video_path) or '.', output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Processing video... Annotated output will be saved to '{output_path}'")

        # Clean and prepare save directory
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        try:
            results_generator = self.model.track(video_path, stream=True, persist=True, device=self.device, iou=0.4)
            frame_num = 0
            for results in results_generator:
                frame = results.orig_img
                annotated_frame = frame.copy()
                frame_num += 1

                if results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results.boxes.id.cpu().numpy().astype(int)
                    
                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        face_img = frame[y1:y2, x1:x2]
                        
                        quality_score = self.scorer.get_score(face_img)

                        self.all_faces[track_id].append({
                            'image': face_img,
                            'quality_score': quality_score,
                            'frame_num': frame_num
                        })
                        
                        label = f"ID {track_id}: {quality_score:.2f}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"Processed Frame: {frame_num}, Active Tracks: {len(track_ids)}", end='\r')
                
                # Write the annotated frame to the output video
                out.write(annotated_frame)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user (Ctrl+C).")
        finally:
            print(f"\nFinished processing. Video saved to {output_path}")
            cap.release()
            out.release()
            self.save_best_faces()

    def save_best_faces(self):
        """Sorts through all tracked faces and saves the best N for each track."""
        print("\nSaving best faces for each track...")
        for track_id, faces in self.all_faces.items():
            if not faces:
                continue

            sorted_faces = sorted(faces, key=lambda x: x['quality_score'], reverse=True)
            top_faces = sorted_faces[:self.top_n]

            person_dir = os.path.join(self.save_dir, f"person_{track_id}")
            os.makedirs(person_dir, exist_ok=True)

            for rank, face_data in enumerate(top_faces):
                filename = os.path.join(person_dir, f"rank_{rank+1}_score_{face_data['quality_score']:.2f}.jpg")
                cv2.imwrite(filename, face_data['image'])
            
            print(f"  - Saved top {len(top_faces)} faces for person {track_id}")

if __name__ == '__main__':
    video_path = r'C:\Users\raoit\Downloads\Knowns.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        processor = VideoProcessor()
        processor.process_video(video_path)
        print(f"\nProcessing complete. Results saved in '{os.path.abspath(processor.save_dir)}' directory.")
        try:
            os.startfile(os.path.abspath(processor.save_dir))
        except (AttributeError, FileNotFoundError):
            print("Could not automatically open the results folder. Please open it manually.")
