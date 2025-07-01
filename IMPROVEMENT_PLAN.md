# Face Recognition System Improvement Plan

This document outlines the strategy for enhancing the face detection and recognition system. The primary goal is to create a robust pre-processing module that intelligently selects the highest quality face images from a video stream before sending them to an external recognition API. This will improve accuracy and optimize API usage.

## Core Problem Statement

The current system needs to be improved to handle multiple people effectively and to be more selective about which faces are sent for recognition. We need to move from random selection to a quality-based scoring and ranking system.

## Proposed Architecture

The system will be split into two distinct, decoupled modules:

1. **Module 1: Face Tracking and Quality Scoring (`main.py`)**
   * **Input**: Live video stream (e.g., from an IP camera).
   * **Functionality**: This module will be responsible for detecting, tracking, scoring, and ranking faces in real-time.
   * **Output**: A directory (`ranked_faces/top_faces`) containing subdirectories for each unique tracked person (e.g., `track_1`, `track_2`). Each subdirectory will contain the top N (e.g., 3-5) best-quality, aligned, and cropped face images for that person.

2. **Module 2: Face Recognition (`recognize_faces.py`)**
   * **Input**: The output directory from Module 1.
   * **Functionality**: This script will monitor the output directory. When new face images appear, it will send them to the external recognition API.
   * **Output**: The final recognized identity for each person, potentially determined by a majority vote on the results from the N images.

---

## Detailed Enhancement Plan

### 1. Enhance the Scoring & Ranking Model (in `FaceDetectionRanker` class)

The key to selecting the "best" face is a comprehensive scoring model. We will enhance the existing model with the following:

* [ ] **Add "Closed Mouth" Detection**: Utilize facial landmarks (already extracted by the YOLO model) to measure the distance between the upper and lower lip. A smaller distance indicates a closed mouth, which is preferable for recognition. This will be added as a new scoring metric.
* [ ] **Refine Score Weighting**: Introduce a configurable dictionary for the weights of each quality metric (e.g., `score_weights = {'blur': 0.2, 'frontality': 0.4, 'eyes': 0.2, 'mouth': 0.2}`). This will allow for easy fine-tuning of what the system considers a "high-quality" face.
* [ ] **Improve Completeness Score**: Enhance the `check_face_completeness` and `is_half_face` functions to be more robust in rejecting partially visible faces.

### 2. Improve Face Normalization

Recognition APIs perform best with well-normalized images.

* [ ] **Enforce Face Alignment**: Ensure that every face saved to the `top_faces` directory is first passed through the `improved_align_face` function. This will standardize the rotation and orientation of the faces, making them consistent.

### 3. Refactor the Workflow for Decoupling

To create a clean separation of concerns, we will refactor the code execution flow.

* [ ] **Remove Direct API Calls from `main.py`**: The `process_completed_track` function will be modified. Instead of calling `queue_process`, its sole responsibility will be to save the top-ranked aligned faces to the correct directory (`top_faces/[track_id]/`).
* [ ] **Create `recognize_faces.py`**: A new script will be created with the following logic:
  * Use a file system watcher (like `watchdog`) or a simple timed loop to monitor the `top_faces` directory for new subdirectories.
  * For each new track ID folder, collect the face images.
  * Call the external recognition API for each of the top N images.
  * Implement a voting mechanism (e.g., the most frequent name returned) to determine the final identity.
  * Move the processed track ID folder to an `archive` directory to prevent re-processing.

### 4. Improve Tracking Robustness

* [ ] **Review Tracking Parameters**: Re-evaluate parameters like `max_frames_missing` and `iou_threshold` to optimize tracking performance in crowded scenes.

