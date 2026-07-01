"""Landmark-based face quality scoring and hard gates (Stage 2)."""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pipeline_io import bbox_intersection_over_face, bbox_iou, clamp_bbox, point_inside_bbox


class FaceQualityEngine:
    """Detect faces on full frame, filter by person bbox, score with landmark heuristics."""

    def __init__(self, config, device=None, ofiq_scorer=None):
        self.config = config
        self.ofiq_scorer = ofiq_scorer
        face_cfg = config["face"]
        self.conf = face_cfg["conf"]
        self.iou = face_cfg.get("iou", 0.35)
        self.min_face_height = face_cfg["min_face_height_px"]
        self.gates = face_cfg["gates"]
        self.score_weights = face_cfg["score_weights"]
        self.edge_margin = 20
        ranker_cfg = config.get("face_ranker", {})
        self.bbox_expand_ratio = ranker_cfg.get("bbox_expand_ratio", 0.05)
        self.face_crop_expand_ratio = ranker_cfg.get("face_crop_expand_ratio", 0.20)
        self.face_crop_expand_min_px = ranker_cfg.get("face_crop_expand_min_px", 20)
        self.reject_half_face = ranker_cfg.get("reject_half_face", False)
        self.max_face_area_ratio = ranker_cfg.get("max_face_area_ratio", 0.22)
        self.min_face_person_overlap = ranker_cfg.get("min_face_person_overlap", 0.12)
        self.multi_face_blur_ratio = ranker_cfg.get("multi_face_blur_ratio", 0.35)
        self.head_anchor_ratio = ranker_cfg.get("head_anchor_ratio", 0.12)
        self.skip_receding = ranker_cfg.get("skip_receding_samples", True)
        self.check_back_facing = ranker_cfg.get("check_back_facing", True)
        self.max_face_center_y_ratio = ranker_cfg.get("max_face_center_y_ratio", 0.52)
        self.min_frontality_back = ranker_cfg.get("min_frontality_back", 40)
        self.min_eyes_back = ranker_cfg.get("min_eyes_back", 32)
        self.align_faces = ranker_cfg.get("align_faces", True)
        self.align_max_angle_deg = ranker_cfg.get("align_max_angle_deg", 30)
        self.camera_profile = config.get("camera_profile", "standard")

        model_path = face_cfg["model"]
        if not __import__("os").path.isfile(model_path):
            model_path = face_cfg.get("model_fallback", model_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

    def predict(self, frame, conf=None):
        return self.model.predict(
            frame,
            conf=conf if conf is not None else self.conf,
            device=self.device,
            iou=self.iou,
            verbose=False,
        )[0]

    @staticmethod
    def calculate_blur_score(img):
        if img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def normalize_blur_score(raw_blur, img):
        """Scale blur so small/distant overhead faces are not auto-rejected."""
        if img is None or img.size == 0:
            return raw_blur
        h, w = img.shape[:2]
        ref_area = 112.0 * 112.0
        return raw_blur * ref_area / max(h * w, 1.0)

    @staticmethod
    def calculate_lighting_score(img):
        if img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        return (1 - abs(mean_brightness - 127) / 255) * 70 + (std_brightness / 128) * 30

    @staticmethod
    def calculate_box_size_score(w, h, fw, fh):
        size_pct = (w * h) / (fw * fh)
        if size_pct < 0.005:
            return size_pct * 20000
        if size_pct > 0.5:
            return (1 - size_pct) * 200
        return 100 * size_pct * 2

    def check_face_completeness(self, face_img):
        try:
            if face_img.size == 0:
                return 0.0
            face = cv2.resize(face_img, (64, 64))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            left = gray[:, :32]
            right = np.fliplr(gray[:, 32:])
            symmetry = np.corrcoef(left.flatten(), right.flatten())[0, 1]
            if np.isnan(symmetry):
                symmetry = 0
            edges = cv2.Canny(gray, 100, 200)
            left_density = np.count_nonzero(edges[:, :32]) / (32 * 64)
            right_density = np.count_nonzero(edges[:, 32:]) / (32 * 64)
            balance = (
                min(left_density, right_density) / max(left_density, right_density)
                if left_density + right_density > 0
                else 0
            )
            y, x = np.mgrid[0:64, 0:64]
            center_x = np.sum(x * gray) / np.sum(gray) if np.sum(gray) > 0 else 32
            center_score = 1 - abs(center_x - 32) / 32
            score = (symmetry * 0.35 + balance * 0.3 + center_score * 0.35) * 100
            sym_floor = 0.22 if self.camera_profile == "overhead" else 0.3
            if balance < sym_floor or symmetry < sym_floor:
                score *= 0.7
            return max(0.0, min(100.0, score))
        except Exception:
            return 0.0

    def calculate_frontality_score(self, img, landmarks):
        try:
            if img.size == 0:
                return 0.0
            if landmarks is None or len(landmarks) < 4:
                face = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                left_half = gray[:, :32]
                right_half = np.fliplr(gray[:, 32:])
                symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if np.isnan(symmetry):
                    symmetry = 0
                h, w = img.shape[:2]
                aspect_score = 1.0 - min(1.0, abs((w / h if h > 0 else 0) - 0.8) / 0.4)
                return (symmetry * 0.7 + aspect_score * 0.3) * 100

            left_eye = (landmarks[0], landmarks[1])
            right_eye = (landmarks[2], landmarks[3])
            eye_dx = abs(right_eye[0] - left_eye[0])
            eye_dy = abs(right_eye[1] - left_eye[1])
            horizontal_alignment = max(0, 1.0 - (eye_dy / eye_dx if eye_dx > 0 else 1.0))
            h, w = img.shape[:2]
            eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
            center_alignment = max(0, 1.0 - abs(eye_midpoint_x - (w / 2)) / (w / 2))
            eye_distance = abs(right_eye[0] - left_eye[0])
            distance_ratio = eye_distance / w
            ratio_target = 0.38 if self.camera_profile == "overhead" else 0.43
            ratio_tolerance = 0.35 if self.camera_profile == "overhead" else 0.2
            distance_score = 1.0 - min(1.0, abs(distance_ratio - ratio_target) / ratio_tolerance)
            return min(
                100,
                max(
                    0,
                    (horizontal_alignment * 0.4 + center_alignment * 0.4 + distance_score * 0.2)
                    * 100,
                ),
            )
        except Exception:
            return 50.0

    def detect_open_eyes(self, img, landmarks):
        try:
            if img.size == 0:
                return 0.0
            h, w = img.shape[:2]
            if landmarks is None or len(landmarks) < 4:
                left_eye_region = img[int(h * 0.2) : int(h * 0.45), int(w * 0.15) : int(w * 0.45)]
                right_eye_region = img[int(h * 0.2) : int(h * 0.45), int(w * 0.55) : int(w * 0.85)]
                if left_eye_region.size == 0 or right_eye_region.size == 0:
                    return 50.0
            else:
                eye_width = int(abs(landmarks[2] - landmarks[0]) * 0.3)
                eye_height = int(eye_width * 0.5)
                left_x1 = max(0, int(landmarks[0] - eye_width / 2))
                left_y1 = max(0, int(landmarks[1] - eye_height / 2))
                left_x2 = min(w, int(landmarks[0] + eye_width / 2))
                left_y2 = min(h, int(landmarks[1] + eye_height / 2))
                right_x1 = max(0, int(landmarks[2] - eye_width / 2))
                right_y1 = max(0, int(landmarks[3] - eye_height / 2))
                right_x2 = min(w, int(landmarks[2] + eye_width / 2))
                right_y2 = min(h, int(landmarks[3] + eye_height / 2))
                left_eye_region = img[left_y1:left_y2, left_x1:left_x2]
                right_eye_region = img[right_y1:right_y2, right_x1:right_x2]
                if left_eye_region.size == 0 or right_eye_region.size == 0:
                    return (
                        np.sqrt((landmarks[0] - landmarks[2]) ** 2 + (landmarks[1] - landmarks[3]) ** 2)
                        / w
                    ) * 500

            left_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
            left_var = np.var(left_gray)
            right_var = np.var(right_gray)
            left_edges = cv2.Canny(left_gray, 50, 150)
            right_edges = cv2.Canny(right_gray, 50, 150)
            left_edge_density = np.count_nonzero(left_edges) / left_edges.size
            right_edge_density = np.count_nonzero(right_edges) / right_edges.size
            var_score = min(100, max(0, ((left_var + right_var) / 2) * 0.4))
            edge_score = min(100, max(0, ((left_edge_density + right_edge_density) / 2) * 1000))
            return min(100, max(0, var_score * 0.7 + edge_score * 0.3))
        except Exception:
            return 50.0

    @staticmethod
    def _mouth_roi(face_img, landmarks=None):
        """Mouth region — landmark-based when available, else proportional band."""
        h, w = face_img.shape[:2]
        if landmarks is not None and len(landmarks) >= 4:
            eye_cy = (landmarks[1] + landmarks[3]) / 2
            eye_cx = (landmarks[0] + landmarks[2]) / 2
            eye_dist = max(4.0, abs(landmarks[2] - landmarks[0]))
            mouth_cy = eye_cy + 0.9 * eye_dist
            half_h = 0.45 * eye_dist
            half_w = 0.65 * eye_dist
            y1 = int(max(0, mouth_cy - half_h))
            y2 = int(min(h, mouth_cy + half_h))
            x1 = int(max(0, eye_cx - half_w))
            x2 = int(min(w, eye_cx + half_w))
        else:
            # Overhead / no landmarks: wider band than old fixed bottom slice.
            y1 = int(h * 0.52)
            y2 = int(h * 0.92)
            x1 = int(w * 0.18)
            x2 = int(w * 0.82)
        if y2 <= y1 or x2 <= x1:
            return None
        return face_img[y1:y2, x1:x2]

    def score_mouth_visibility(self, face_img, landmarks=None):
        """
        Mouth quality for recognition — texture/edge based, not closed-vs-open.
        Works with overhead cameras where chin shadow fooled the old dark-pixel test.
        """
        if face_img.size == 0:
            return 0.0
        try:
            mouth_roi = self._mouth_roi(face_img, landmarks)
            if mouth_roi is None or mouth_roi.size == 0:
                return 50.0
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 40, 120)
            edge_density = np.count_nonzero(edges) / edges.size
            texture = min(100.0, max(0.0, lap * 0.35))
            edge_score = min(100.0, max(0.0, edge_density * 600))
            return min(100.0, max(0.0, texture * 0.55 + edge_score * 0.45))
        except Exception:
            return 50.0

    def detect_closed_mouth(self, face_img, landmarks=None):
        """Backward-compatible alias for score_mouth_visibility."""
        return self.score_mouth_visibility(face_img, landmarks)

    def is_half_face(self, face_img, box, frame_w, frame_h):
        try:
            if face_img.size == 0:
                return True
            x1, y1, x2, y2 = box
            face_w, face_h = x2 - x1, y2 - y1
            boundary_margin = 0.05
            min_margin_px = max(10, int(min(frame_w, frame_h) * boundary_margin))
            if (
                x1 < min_margin_px
                or y1 < min_margin_px
                or frame_w - x2 < min_margin_px
                or frame_h - y2 < min_margin_px
            ):
                return True
            aspect_ratio = face_w / face_h if face_h > 0 else 0
            max_aspect = 1.35 if self.camera_profile == "overhead" else 1.2
            min_symmetry = 0.42 if self.camera_profile == "overhead" else 0.5
            if aspect_ratio < 0.5 or aspect_ratio > max_aspect:
                return True
            gray = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
            left_half = gray[:, :32]
            right_half = np.fliplr(gray[:, 32:])
            symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if np.isnan(symmetry) or symmetry < min_symmetry:
                return True
            edges = cv2.Canny(gray, 100, 200)
            border_size = 5
            edge_regions = [
                edges[:, :border_size],
                edges[:, -border_size:],
                edges[:border_size, :],
                edges[-border_size:, :],
            ]
            edge_densities = [np.count_nonzero(region) / region.size for region in edge_regions]
            if max(edge_densities) > 0.25:
                return True
            face_center_x = x1 + face_w / 2
            relative_x = face_center_x / frame_w
            edge_lo = 0.12 if self.camera_profile == "overhead" else 0.25
            edge_hi = 0.88 if self.camera_profile == "overhead" else 0.75
            if relative_x < edge_lo or relative_x > edge_hi:
                return True
            return False
        except Exception:
            return True

    def combined_score(self, metrics):
        total = 0.0
        weight_sum = 0.0
        for metric, weight in self.score_weights.items():
            if metric in metrics:
                total += metrics[metric] * weight
                weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def passes_gates(self, metrics):
        blur_gate = self.gates.get("blur", 0)
        if blur_gate > 0:
            blur_val = metrics.get("blur_norm", metrics.get("blur", 0))
            if blur_val < blur_gate:
                return "low_blur"
        gate_checks = (
            ("completeness", "low_completeness"),
            ("frontality", "low_frontality"),
            ("eyes", "low_eyes"),
            ("mouth", "low_mouth"),
            ("combined", "low_combined"),
        )
        for metric_key, reject_reason in gate_checks:
            threshold = self.gates.get(metric_key, 0)
            if threshold > 0 and metrics.get(metric_key, 0) < threshold:
                return reject_reason
        return None

    def is_receding_sample(self, person_meta):
        if not self.skip_receding:
            return False
        return bool(person_meta.get("receding"))

    def is_face_in_head_region(self, face_box, person_meta):
        """True when the face center sits in the upper portion of the person box."""
        y1 = person_meta.get("y1")
        y2 = person_meta.get("y2")
        if y1 is None or y2 is None or y2 <= y1:
            return True
        person_h = y2 - y1
        face_cy = (face_box[1] + face_box[3]) / 2
        max_y = y1 + self.max_face_center_y_ratio * person_h
        return face_cy <= max_y

    def back_facing_reason(self, face, person_meta):
        """
        Back/profile rejection — requires combined signals so distant front-facing
        people (small face in a large person box) are not dropped.
        """
        if not self.check_back_facing:
            return None

        metrics = face.get("metrics") or {}
        frontality = metrics.get("frontality", 0)
        eyes = metrics.get("eyes", 0)
        very_low_front = frontality < self.min_frontality_back * 0.7
        low_front = frontality < self.min_frontality_back
        low_eyes = eyes < self.min_eyes_back
        head_ok = self.is_face_in_head_region(face["face_box"], person_meta)

        if very_low_front:
            return "back_facing_low_frontality"
        if low_front and low_eyes:
            return "back_facing"
        if low_front and not head_ok and self.camera_profile != "overhead":
            return "back_facing_head_position"
        return None

    def is_back_turned_face(self, face, person_meta):
        return self.back_facing_reason(face, person_meta) is not None

    def _extract_landmarks(self, detections, index):
        if not hasattr(detections, "keypoints") or detections.keypoints is None:
            return None
        try:
            kpts = detections.keypoints[index].xy[0].cpu().numpy()
            return [
                float(kpts[0][0]),
                float(kpts[0][1]),
                float(kpts[1][0]),
                float(kpts[1][1]),
            ]
        except Exception:
            return None

    @staticmethod
    def expand_bbox(bbox, frame_w, frame_h, ratio, min_pad_px=0):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(min_pad_px, int(bw * ratio))
        pad_y = max(min_pad_px, int(bh * ratio))
        return clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, frame_w, frame_h)

    def expand_face_crop_bbox(self, face_box, frame_w, frame_h):
        """Pad face detector box before export so FaceNet/MTCNN has margin (legacy used ~35px)."""
        return self.expand_bbox(
            face_box,
            frame_w,
            frame_h,
            self.face_crop_expand_ratio,
            self.face_crop_expand_min_px,
        )

    @staticmethod
    def _landmarks_in_crop(landmarks, crop_x1, crop_y1):
        if landmarks is None or len(landmarks) < 4:
            return None
        return [
            landmarks[0] - crop_x1,
            landmarks[1] - crop_y1,
            landmarks[2] - crop_x1,
            landmarks[3] - crop_y1,
        ]

    def align_face_roll(self, face_img, landmarks):
        """
        Roll correction from eye landmarks (ported from main.py improved_align_face).
        Does not frontalize side/profile faces — only levels head tilt.
        """
        try:
            if landmarks is None or len(landmarks) < 4:
                return face_img

            h, w = face_img.shape[:2]
            if h == 0 or w == 0:
                return face_img

            left_eye = np.array([landmarks[0], landmarks[1]])
            right_eye = np.array([landmarks[2], landmarks[3]])

            if (
                not (0 <= left_eye[0] < w and 0 <= left_eye[1] < h)
                or not (0 <= right_eye[0] < w and 0 <= right_eye[1] < h)
            ):
                return face_img

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) > self.align_max_angle_deg:
                return face_img

            padding = int(0.2 * max(h, w))
            padded_img = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=face_img.dtype)
            padded_img[padding : padding + h, padding : padding + w] = face_img
            center = (w // 2 + padding, h // 2 + padding)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(
                padded_img,
                rotation_matrix,
                (padded_img.shape[1], padded_img.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            y1 = rotated_img.shape[0] // 2 - h // 2
            y2 = y1 + h
            x1 = rotated_img.shape[1] // 2 - w // 2
            x2 = x1 + w
            y1 = max(0, min(y1, rotated_img.shape[0] - 1))
            y2 = max(0, min(y2, rotated_img.shape[0]))
            x1 = max(0, min(x1, rotated_img.shape[1] - 1))
            x2 = max(0, min(x2, rotated_img.shape[1]))
            aligned_img = rotated_img[y1:y2, x1:x2]

            if aligned_img.size == 0:
                return face_img
            result_h, result_w = aligned_img.shape[:2]
            if result_h < 0.8 * h or result_w < 0.8 * w:
                return face_img
            return aligned_img
        except Exception as exc:
            print(f"Face roll alignment error: {exc}")
            return face_img

    def _prepare_face_crop(self, face_img, landmarks, crop_x1, crop_y1):
        if not self.align_faces:
            return face_img
        local_landmarks = self._landmarks_in_crop(landmarks, crop_x1, crop_y1)
        return self.align_face_roll(face_img, local_landmarks)

    def _face_associated_with_person(self, face_box, person_bbox, frame_w, frame_h, expand_ratio):
        if person_bbox is None:
            return True
        if bbox_intersection_over_face(face_box, person_bbox) >= self.min_face_person_overlap:
            return True
        cx = (face_box[0] + face_box[2]) / 2
        cy = (face_box[1] + face_box[3]) / 2
        expanded = self.expand_bbox(person_bbox, frame_w, frame_h, expand_ratio)
        return point_inside_bbox(cx, cy, expanded)

    @staticmethod
    def _offset_face_box(face_box, dx, dy):
        x1, y1, x2, y2 = face_box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    @staticmethod
    def _dedupe_face_candidates(candidates):
        if len(candidates) <= 1:
            return candidates
        kept = []
        for candidate in sorted(
            candidates, key=lambda c: c["metrics"]["blur"], reverse=True
        ):
            if any(
                bbox_iou(candidate["face_box"], other["face_box"]) > 0.45
                for other in kept
            ):
                continue
            kept.append(candidate)
        return kept

    def _pick_best_face_candidate(self, candidates, person_bbox=None):
        """Multiple faces in one person box: blur beats OFIQ; occluders are blurry."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        max_blur = max(c["metrics"]["blur"] for c in candidates)
        if max_blur >= 10:
            sharp = [
                c
                for c in candidates
                if c["metrics"]["blur"] >= max_blur * self.multi_face_blur_ratio
            ]
            if sharp:
                candidates = sharp

        anchor = None
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            anchor = (
                (x1 + x2) / 2,
                y1 + self.head_anchor_ratio * (y2 - y1),
            )

        for candidate in candidates:
            x1, y1, x2, y2 = candidate["face_box"]
            face_area = max(1, (x2 - x1) * (y2 - y1))
            candidate["_face_area"] = face_area
            if person_bbox is not None:
                pb_area = max(1, (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1]))
                candidate["_area_ratio"] = face_area / pb_area
            else:
                candidate["_area_ratio"] = 0.0
            if self.ofiq_scorer is not None:
                candidate["_ofiq"] = self.ofiq_scorer.get_score(candidate["face_crop"])
            else:
                candidate["_ofiq"] = 0.0
            if anchor is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                candidate["_head_dist"] = ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
            else:
                candidate["_head_dist"] = 0.0

        def rank_key(candidate):
            oversize = max(0.0, candidate["_area_ratio"] - self.max_face_area_ratio)
            metrics = candidate["metrics"]
            half_penalty = 1 if candidate.get("half_face") else 0
            return (
                metrics["blur"],
                candidate["_ofiq"],
                metrics["combined"],
                -half_penalty,
                -candidate["_head_dist"],
                -oversize,
                -candidate["_face_area"],
                candidate["conf"],
            )

        return max(candidates, key=rank_key)

    def _collect_face_candidates(
        self,
        image,
        detections,
        frame_w,
        frame_h,
        *,
        person_bbox=None,
        expand_ratio=None,
        min_face_height=None,
        reject_half_face=None,
        box_offset=(0, 0),
        assoc_frame_w=None,
        assoc_frame_h=None,
    ):
        min_h = min_face_height if min_face_height is not None else self.min_face_height
        reject_half = (
            self.reject_half_face if reject_half_face is None else reject_half_face
        )
        ratio = expand_ratio if expand_ratio is not None else self.bbox_expand_ratio
        ox, oy = box_offset
        assoc_frame_w = frame_w if assoc_frame_w is None else assoc_frame_w
        assoc_frame_h = frame_h if assoc_frame_h is None else assoc_frame_h
        if not hasattr(detections, "boxes") or detections.boxes is None:
            return []

        candidates = []
        for i, box in enumerate(detections.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_h, face_w = y2 - y1, x2 - x1
            if face_h < min_h or face_w < min_h:
                continue
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)
            full_box = self._offset_face_box((x1, y1, x2, y2), ox, oy)
            if not self._face_associated_with_person(
                full_box,
                person_bbox,
                assoc_frame_w,
                assoc_frame_h,
                ratio,
            ):
                continue

            face_img_raw = image[y1:y2, x1:x2].copy()
            landmarks = self._extract_landmarks(detections, i)
            local_landmarks = self._landmarks_in_crop(landmarks, x1, y1)
            # Quality metrics on tight detector box; export crop is padded for FaceNet.
            if self.face_crop_expand_ratio > 0 or self.face_crop_expand_min_px > 0:
                ecx1, ecy1, ecx2, ecy2 = self.expand_face_crop_bbox(
                    (x1, y1, x2, y2), frame_w, frame_h
                )
                face_export_raw = image[ecy1:ecy2, ecx1:ecx2].copy()
                face_img = self._prepare_face_crop(face_export_raw, landmarks, ecx1, ecy1)
            else:
                face_img = self._prepare_face_crop(face_img_raw, landmarks, x1, y1)
            half_face = reject_half and self.is_half_face(
                face_img_raw, (x1, y1, x2, y2), frame_w, frame_h
            )

            raw_blur = self.calculate_blur_score(face_img)
            metrics = {
                "blur": raw_blur,
                "blur_norm": self.normalize_blur_score(raw_blur, face_img),
                "lighting": self.calculate_lighting_score(face_img_raw),
                "size": self.calculate_box_size_score(face_w, face_h, frame_w, frame_h),
                "completeness": self.check_face_completeness(face_img_raw),
                "eyes": self.detect_open_eyes(face_img_raw, local_landmarks),
                "frontality": self.calculate_frontality_score(face_img_raw, local_landmarks),
                "mouth": self.score_mouth_visibility(face_img_raw, local_landmarks),
            }
            metrics["combined"] = self.combined_score(metrics)
            candidates.append(
                {
                    "face_crop": face_img,
                    "face_crop_tight": face_img_raw,
                    "face_box": full_box,
                    "conf": float(box.conf[0]),
                    "landmarks": landmarks,
                    "metrics": metrics,
                    "half_face": half_face,
                }
            )
        return candidates

    def _find_face(
        self,
        full_frame,
        person_meta,
        person_crop=None,
        *,
        conf=None,
        min_face_height=None,
        bbox_expand_ratio=None,
        reject_half_face=None,
    ):
        h, w = full_frame.shape[:2]
        bbox = clamp_bbox(
            person_meta["x1"], person_meta["y1"], person_meta["x2"], person_meta["y2"], w, h
        )
        expand = bbox_expand_ratio if bbox_expand_ratio is not None else self.bbox_expand_ratio
        detections = self.predict(full_frame, conf=conf)
        candidates = self._collect_face_candidates(
            full_frame,
            detections,
            w,
            h,
            person_bbox=bbox,
            expand_ratio=expand,
            min_face_height=min_face_height,
            reject_half_face=reject_half_face,
        )
        if person_crop is not None and person_crop.size > 0:
            ph, pw = person_crop.shape[:2]
            crop_det = self.predict(person_crop, conf=conf)
            crop_candidates = self._collect_face_candidates(
                person_crop,
                crop_det,
                pw,
                ph,
                person_bbox=bbox,
                expand_ratio=expand,
                min_face_height=min_face_height,
                reject_half_face=reject_half_face,
                box_offset=(bbox[0], bbox[1]),
                assoc_frame_w=w,
                assoc_frame_h=h,
            )
            candidates.extend(crop_candidates)

        candidates = self._dedupe_face_candidates(candidates)
        return self._pick_best_face_candidate(candidates, bbox)

    def find_best_face_in_person_bbox(self, full_frame, person_bbox):
        """Return best face associated with person_bbox."""
        fh, fw = full_frame.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3], fw, fh)
        tight = (x1, y1, x2, y2)
        detections = self.predict(full_frame)
        candidates = self._collect_face_candidates(
            full_frame,
            detections,
            fw,
            fh,
            person_bbox=tight,
            expand_ratio=self.bbox_expand_ratio,
        )
        return self._pick_best_face_candidate(candidates, tight)

    def find_best_face_in_crop(self, person_crop):
        """Detect face directly inside the saved person crop."""
        if person_crop is None or person_crop.size == 0:
            return None
        ph, pw = person_crop.shape[:2]
        detections = self.predict(person_crop)
        candidates = self._collect_face_candidates(
            person_crop,
            detections,
            pw,
            ph,
            person_bbox=(0, 0, pw, ph),
            expand_ratio=0.0,
        )
        return self._pick_best_face_candidate(candidates, (0, 0, pw, ph))

    def find_face_in_sample(self, full_frame, person_meta, person_crop=None):
        """
        Find best face for a buffered sample.
        Returns (face_dict, hard_reject) where hard_reject is only 'no_face'.
        """
        if self.is_receding_sample(person_meta):
            return None, "receding"
        face = self._find_face(full_frame, person_meta, person_crop)
        if face is None:
            return None, "no_face"
        back_reason = self.back_facing_reason(face, person_meta)
        if back_reason:
            face["gate_reject"] = back_reason
            return face, "back_facing"
        if self.reject_half_face and face.get("half_face"):
            face["gate_reject"] = face.get("gate_reject") or "half_face"
        else:
            face["gate_reject"] = self.passes_gates(face["metrics"])
        return face, None

    def find_face_lenient(self, full_frame, person_meta, person_crop=None):
        """Low-threshold face search when normal detection finds nothing."""
        if self.is_receding_sample(person_meta):
            return None, "receding"
        lenient = self.config.get("face", {}).get("lenient", {})
        lenient_reject_half = lenient.get("reject_half_face", self.reject_half_face)
        face = self._find_face(
            full_frame,
            person_meta,
            person_crop,
            conf=lenient.get("conf", 0.25),
            min_face_height=lenient.get("min_face_height_px", 20),
            bbox_expand_ratio=lenient.get("bbox_expand_ratio", 0.25),
            reject_half_face=lenient_reject_half,
        )
        if face is None:
            return None, "no_face"
        back_reason = self.back_facing_reason(face, person_meta)
        if back_reason:
            face["gate_reject"] = back_reason
            return face, "back_facing"
        if lenient_reject_half and face.get("half_face"):
            face["gate_reject"] = face.get("gate_reject") or "half_face"
        else:
            face["gate_reject"] = self.passes_gates(face["metrics"])
        return face, None

    def evaluate_sample(self, full_frame, person_meta, person_crop=None):
        """Evaluate one buffered sample. Returns (result_dict, reject_reason)."""
        face, reject = self.find_face_in_sample(full_frame, person_meta, person_crop)
        if reject:
            return None, reject
        gate_reject = face.get("gate_reject")
        if gate_reject:
            return face, gate_reject
        return face, None
