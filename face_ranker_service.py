"""Stage 2: Watch person_queue/, rank faces, export best per person to ranked_faces/."""

import json
import os
import shutil
import time
import traceback
from collections import Counter

import cv2

from face_quality import FaceQualityEngine
from ofiq_scorer import OFIQScorer
from pipeline_io import (
    bbox_in_roi,
    clamp_bbox,
    crop_person_from_full,
    ensure_dir,
    get_pending_folders,
    is_roi_active,
    iter_staging_samples,
    load_config,
    load_roi,
    clear_processed,
    mark_processed,
    folder_inactive,
    parse_track_id_from_folder,
    write_ready,
)


RANK_MANIFEST = "_rank_manifest.json"
FACE_CACHE_DIR = ".face_cache"


def _ofiq_input_crop(face_result):
    """Pick tight crop for OFIQ; never use `or` — numpy images are ambiguous in boolean context."""
    tight = face_result.get("face_crop_tight")
    if tight is not None:
        return tight
    return face_result["face_crop"]


class FaceRankerService:
    def __init__(self, config=None):
        self.config = config or load_config()
        person_cfg = self.config["person"]
        ofiq_cfg = self.config["ofiq"]
        ranker_cfg = self.config.get("face_ranker", {})
        watcher_cfg = self.config.get("watcher", {})

        self.queue_dir = person_cfg["queue_dir"]
        recognition_mode = os.environ.get("PIPELINE_RECOGNITION_MODE") == "1"
        self.rank_only = False if recognition_mode else ranker_cfg.get("rank_only", True)
        self.recognition_minimal = recognition_mode or ranker_cfg.get(
            "recognition_minimal_export", False
        )
        if self.rank_only:
            self.output_dir = ranker_cfg.get("output_dir", "ranked_faces")
        else:
            self.output_dir = ofiq_cfg.get("recognition_dir", "recognition_folder")
        self.merge_batches = ranker_cfg.get("merge_batches_per_track", True)
        if recognition_mode:
            self.write_ready = True
        elif not self.rank_only:
            # Recognition output path — default write_ready on unless explicitly false.
            self.write_ready = ranker_cfg.get("write_ready", True)
        else:
            self.write_ready = ranker_cfg.get("write_ready", False)
        self.min_export = ranker_cfg.get("min_export_per_person", 1)
        self.always_export_person = ranker_cfg.get("always_export_person", True)
        self.skip_fallback_edge_bbox = ranker_cfg.get("skip_fallback_edge_bbox", True)
        self.max_bbox_y1_ratio = ranker_cfg.get(
            "max_bbox_y1_ratio", self.config["person"].get("max_bbox_y1_ratio", 0.72)
        )
        self.ofiq_threshold = ofiq_cfg["threshold"]
        self.low_ofiq_dir = ofiq_cfg.get("low_ofiq_dir", "low_ofiq_faces")
        self.export_top_n = ofiq_cfg["export_top_n"]
        self.check_interval = watcher_cfg.get("check_interval_sec", 3)
        self.inactivity_wait = watcher_cfg.get("inactivity_wait_sec", 5)
        self.export_only_passing_gates = ranker_cfg.get("export_only_passing_gates", True)

        self.ofiq = OFIQScorer(ofiq_cfg["model"])
        self.face_engine = FaceQualityEngine(self.config, ofiq_scorer=self.ofiq)
        roi_cfg = self.config.get("roi") or {}
        self.roi = load_roi(self.config) if is_roi_active(self.config) else None
        self.roi_face_mode = roi_cfg.get("face_mode", roi_cfg.get("mode", "overlap"))
        self.roi_face_min_overlap = roi_cfg.get(
            "face_min_overlap", roi_cfg.get("min_overlap", 0.15)
        )
        self.outside_roi_dir = roi_cfg.get("outside_roi_dir", "outside_roi_faces")
        self.rejected_dir = ranker_cfg.get("rejected_dir", "rejected_faces")
        ensure_dir(self.queue_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.low_ofiq_dir)
        ensure_dir(self.outside_roi_dir)
        ensure_dir(self.rejected_dir)
        profile = self.config.get("camera_profile", "standard")
        print(f"Camera profile: {profile}")
        print(f"OFIQ threshold: {self.ofiq_threshold} (pass if score >= threshold)")
        print(f"Half-face filter: {'on' if self.face_engine.reject_half_face else 'off'}")
        if self.roi:
            print(
                f"ROI export filter: ({self.roi['x1']:.3f},{self.roi['y1']:.3f})-"
                f"({self.roi['x2']:.3f},{self.roi['y2']:.3f}), "
                f"mode={self.roi_face_mode}, min_overlap={self.roi_face_min_overlap}"
            )

    @staticmethod
    def _rank_key(ofiq_score, tier, combined, person_area=0):
        # OFIQ is primary — best face for recognition, then detection tier as tiebreaker.
        return (ofiq_score, tier, combined, person_area)

    def _load_sample(self, frame_num, meta, full_path, person_path, batch_dir):
        full_frame = cv2.imread(full_path)
        if full_frame is None:
            return None
        person_crop = crop_person_from_full(full_frame, meta)
        return {
            "frame_num": frame_num,
            "meta": meta,
            "full_path": full_path,
            "person_path": person_path,
            "batch_dir": batch_dir,
            "full_frame": full_frame,
            "person_crop": person_crop,
            "person_area": meta.get("area", 0),
        }

    def _candidate_from_face(self, sample, face_result, tier, source):
        ofiq_crop = _ofiq_input_crop(face_result)
        ofiq_score = self.ofiq.get_score(ofiq_crop)
        combined = face_result["metrics"]["combined"]
        passes_ofiq = ofiq_score >= self.ofiq_threshold
        gate = face_result.get("gate_reject") or "ok"
        if not passes_ofiq:
            gate = gate if gate != "ok" else "below_ofiq"
        fh, fw = sample["full_frame"].shape[:2]
        return {
            "rank": self._rank_key(ofiq_score, tier, combined, sample["person_area"]),
            "ofiq": ofiq_score,
            "combined": combined,
            "gate": gate,
            "source": source,
            "frame_num": sample["frame_num"],
            "meta": sample["meta"],
            "full_path": sample["full_path"],
            "batch_dir": sample["batch_dir"],
            "face_crop": face_result["face_crop"],
            "face_box": face_result.get("face_box"),
            "frame_w": fw,
            "frame_h": fh,
        }

    def _candidate_in_roi(self, candidate):
        if not self.roi:
            return True
        face_box = candidate.get("face_box")
        if not face_box:
            return False
        return bbox_in_roi(
            face_box,
            candidate["frame_w"],
            candidate["frame_h"],
            self.roi,
            mode=self.roi_face_mode,
            min_overlap=self.roi_face_min_overlap,
            anchor_y=0.5,
        )

    def _entry_in_roi(self, entry):
        if not self.roi:
            return True
        face_box = entry.get("face_box")
        if not face_box:
            return False
        fw = entry.get("frame_w")
        fh = entry.get("frame_h")
        if not fw or not fh:
            img = cv2.imread(entry.get("full_path", ""))
            if img is None:
                return False
            fh, fw = img.shape[:2]
        return bbox_in_roi(
            face_box,
            fw,
            fh,
            self.roi,
            mode=self.roi_face_mode,
            min_overlap=self.roi_face_min_overlap,
            anchor_y=0.5,
        )

    def _filter_candidates_by_roi(self, candidates):
        if not self.roi:
            return candidates, []
        kept = [c for c in candidates if self._candidate_in_roi(c)]
        outside = [c for c in candidates if c not in kept]
        return kept, outside

    def _filter_entries_by_roi(self, entries):
        if not self.roi:
            return entries, []
        kept = [e for e in entries if self._entry_in_roi(e)]
        outside = [e for e in entries if e not in kept]
        return kept, outside

    def _exportable_candidates(self, candidates):
        if not self.export_only_passing_gates:
            return [c for c in candidates if c.get("face_crop") is not None]
        return [
            c for c in candidates
            if c.get("gate") == "ok" and c.get("face_crop") is not None
        ]

    def _add_face_candidate(self, candidates, rejected, stats, sample, face, tier, source):
        if face.get("gate_reject"):
            stats["gated"] = stats.get("gated", 0) + 1
            self._note_rejected_face(rejected, sample, face, face["gate_reject"])
            return
        candidates.append(self._candidate_from_face(sample, face, tier, source))

    @staticmethod
    def _note_rejected_face(rejected, sample, face, reason):
        if face is None or face.get("face_crop") is None:
            return
        rejected.append(
            {
                "reason": reason,
                "frame_num": sample["frame_num"],
                "face_crop": face["face_crop"],
                "metrics": face.get("metrics"),
                "meta": sample["meta"],
                "full_path": sample["full_path"],
                "batch_dir": sample["batch_dir"],
            }
        )

    def _collect_candidates(self, samples):
        candidates = []
        rejected = []
        stats = {
            "no_face": 0, "receding": 0, "back_facing": 0, "gated": 0,
            "normal": 0, "lenient": 0, "fallback": 0,
        }

        for sample in samples:
            if sample["meta"].get("receding"):
                stats["receding"] += 1
                continue

            face, reject = self.face_engine.find_face_in_sample(
                sample["full_frame"], sample["meta"], sample["person_crop"]
            )
            if reject == "receding":
                stats["receding"] += 1
                continue
            if reject == "back_facing":
                stats["back_facing"] += 1
                self._note_rejected_face(
                    rejected, sample, face, face.get("gate_reject", "back_facing")
                )
                continue
            if reject == "no_face":
                face, reject = self.face_engine.find_face_lenient(
                    sample["full_frame"], sample["meta"], sample["person_crop"]
                )
                if reject == "receding":
                    stats["receding"] += 1
                    continue
                if reject == "back_facing":
                    stats["back_facing"] += 1
                    self._note_rejected_face(
                        rejected, sample, face, face.get("gate_reject", "back_facing")
                    )
                    continue
                if reject == "no_face":
                    stats["no_face"] += 1
                    continue
                stats["lenient"] += 1
                self._add_face_candidate(
                    candidates, rejected, stats, sample, face, tier=2, source="lenient"
                )
                continue

            stats["normal"] += 1
            self._add_face_candidate(
                candidates, rejected, stats, sample, face, tier=3, source="normal"
            )

        if not candidates and self.always_export_person and samples and not self.recognition_minimal:
            fallback_samples = samples
            if self.skip_fallback_edge_bbox:
                fallback_samples = [
                    s
                    for s in samples
                    if not self._is_edge_partial_sample(s)
                ]
            if fallback_samples:
                best = max(fallback_samples, key=lambda s: s["person_area"])
                stats["fallback"] += 1
                candidates.append(
                    {
                        "rank": self._rank_key(0.0, 1, 0.0, best["person_area"]),
                        "ofiq": 0.0,
                        "combined": 0.0,
                        "gate": "person_fallback",
                        "source": "person_fallback",
                        "frame_num": best["frame_num"],
                        "meta": best["meta"],
                        "full_path": best["full_path"],
                        "batch_dir": best["batch_dir"],
                        "face_crop": None,
                    }
                )

        return candidates, stats, rejected

    def _is_edge_partial_sample(self, sample):
        full_frame = cv2.imread(os.path.abspath(sample["full_path"]))
        if full_frame is None:
            return False
        y1 = sample["meta"].get("y1", 0)
        return y1 > full_frame.shape[0] * self.max_bbox_y1_ratio

    @staticmethod
    def _draw_bbox_on_full(full_frame, meta):
        vis = full_frame.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(
            meta["x1"], meta["y1"], meta["x2"], meta["y2"], w, h
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return vis

    def _resolve_face_crop(self, full_frame, meta, person_crop, cached_face_crop):
        """Re-detect on raw full frame so exports never use preview bbox overlays."""
        if full_frame is None:
            return cached_face_crop
        for finder in (
            self.face_engine.find_face_in_sample,
            self.face_engine.find_face_lenient,
        ):
            face, reject = finder(full_frame, meta, person_crop)
            if face is not None:
                return face["face_crop"]
        return cached_face_crop

    @staticmethod
    def _folder_has_face_export(out_dir):
        if not os.path.isdir(out_dir):
            return False
        return any(
            "_face_score_" in name and name.endswith(".jpg")
            for name in os.listdir(out_dir)
        )

    def _write_exports(self, out_dir, to_export):
        for item in to_export:
            frame_num = item["frame_num"]
            ofiq_score = item["ofiq"]
            combined = item["combined"]
            tag = item["source"]
            meta = item.get("meta") or {}

            face_crop = item.get("face_crop")

            if self.recognition_minimal:
                if face_crop is not None and face_crop.size > 0:
                    face_scored = (
                        f"frame_{frame_num}_face_score_{ofiq_score:.2f}_{tag}.jpg"
                    )
                    cv2.imwrite(os.path.join(out_dir, face_scored), face_crop)
                    full_path = item.get("full_path")
                    full_frame = cv2.imread(full_path) if full_path else None
                    if full_frame is None and meta:
                        batch_dir = item.get("batch_dir")
                        if batch_dir:
                            alt = os.path.join(
                                batch_dir, f"frame_{frame_num}_full.jpg"
                            )
                            full_frame = cv2.imread(alt)
                    if full_frame is not None:
                        full_scored = (
                            f"frame_{frame_num}_full_score_{ofiq_score:.2f}_{tag}.jpg"
                        )
                        cv2.imwrite(os.path.join(out_dir, full_scored), full_frame)
                    print(
                        f"  export frame {frame_num} [{tag}]: OFIQ={ofiq_score:.2f} "
                        f"(face + full frame -> recognition)"
                    )
                else:
                    print(
                        f"  skip frame {frame_num} [{tag}]: no face for recognition export"
                    )
                continue

            person_scored = (
                f"frame_{frame_num}_person_score_{ofiq_score:.2f}_c{combined:.0f}_{tag}.jpg"
            )
            full_scored = f"frame_{frame_num}_full_score_{ofiq_score:.2f}_{tag}.jpg"
            bbox_debug = f"frame_{frame_num}_bbox_debug_{tag}.jpg"
            meta_file = f"frame_{frame_num}_meta.json"

            full_path = os.path.abspath(item["full_path"])
            full_frame = cv2.imread(full_path)
            person_crop = None
            if full_frame is not None and meta:
                person_crop = crop_person_from_full(full_frame, meta)
                if person_crop.size > 0:
                    cv2.imwrite(os.path.join(out_dir, person_scored), person_crop)
                cv2.imwrite(os.path.join(out_dir, full_scored), full_frame)
                cv2.imwrite(
                    os.path.join(out_dir, bbox_debug),
                    self._draw_bbox_on_full(full_frame, meta),
                )
                with open(os.path.join(out_dir, meta_file), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            if face_crop is None:
                face_crop = self._resolve_face_crop(
                    full_frame, meta, person_crop, item.get("face_crop")
                )
            if face_crop is not None:
                face_scored = f"frame_{frame_num}_face_score_{ofiq_score:.2f}_{tag}.jpg"
                cv2.imwrite(os.path.join(out_dir, face_scored), face_crop)

            print(
                f"  export frame {frame_num} [{tag}]: OFIQ={ofiq_score:.2f} "
                f"combined={combined:.1f} gate={item['gate']} "
                f"bbox=({meta.get('x1')},{meta.get('y1')})-({meta.get('x2')},{meta.get('y2')})"
            )

    @staticmethod
    def _clear_frame_exports(out_dir):
        for name in os.listdir(out_dir):
            if name in (RANK_MANIFEST, FACE_CACHE_DIR):
                continue
            if name.startswith("frame_"):
                os.remove(os.path.join(out_dir, name))

    def _manifest_entry_from_candidate(self, out_dir, candidate):
        frame_num = candidate["frame_num"]
        face_cache = None
        if candidate.get("face_crop") is not None:
            cache_dir = os.path.join(out_dir, FACE_CACHE_DIR)
            os.makedirs(cache_dir, exist_ok=True)
            face_cache = os.path.join(FACE_CACHE_DIR, f"frame_{frame_num}_{candidate['source']}.jpg")
            cv2.imwrite(os.path.join(out_dir, face_cache), candidate["face_crop"])
        return {
            "rank": list(candidate["rank"]),
            "ofiq": candidate["ofiq"],
            "combined": candidate["combined"],
            "gate": candidate["gate"],
            "source": candidate["source"],
            "frame_num": frame_num,
            "meta": candidate["meta"],
            "full_path": os.path.abspath(candidate["full_path"]),
            "face_cache": face_cache,
            "face_box": candidate.get("face_box"),
            "frame_w": candidate.get("frame_w"),
            "frame_h": candidate.get("frame_h"),
        }


    def _accumulate_and_export_merged(self, out_dir, new_candidates, folder_name):
        """Merge candidates across batches and export global top-N for this track."""
        manifest_path = os.path.join(out_dir, RANK_MANIFEST)
        entries_by_frame = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                for entry in json.load(f).get("entries", []):
                    entries_by_frame[entry["frame_num"]] = entry

        for candidate in new_candidates:
            entry = self._manifest_entry_from_candidate(out_dir, candidate)
            frame_num = entry["frame_num"]
            prev = entries_by_frame.get(frame_num)
            if prev is None or tuple(entry["rank"]) > tuple(prev["rank"]):
                entries_by_frame[frame_num] = entry

        sorted_entries = sorted(
            entries_by_frame.values(),
            key=lambda e: tuple(e["rank"]),
            reverse=True,
        )
        if self.recognition_minimal:
            sorted_entries = [e for e in sorted_entries if e.get("face_cache")]

        if self.export_only_passing_gates:
            export_pool = [e for e in sorted_entries if e.get("gate") == "ok"]
        else:
            export_pool = sorted_entries

        roi_entries, outside_entries = self._filter_entries_by_roi(export_pool)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"entries": sorted_entries}, f, indent=2)

        export_count = max(
            self.min_export,
            min(self.export_top_n, len(roi_entries)),
        )
        top_entries = roi_entries[:export_count]
        to_export = []
        for entry in top_entries:
            item = {
                "rank": tuple(entry["rank"]),
                "ofiq": entry["ofiq"],
                "combined": entry["combined"],
                "gate": entry["gate"],
                "source": entry["source"],
                "frame_num": entry["frame_num"],
                "meta": entry["meta"],
                "full_path": entry["full_path"],
                "face_crop": None,
            }
            if entry.get("face_cache"):
                cache_path = os.path.join(out_dir, entry["face_cache"])
                if os.path.isfile(cache_path):
                    item["face_crop"] = cv2.imread(cache_path)
            to_export.append(item)
        if to_export:
            self._clear_frame_exports(out_dir)
            self._write_exports(out_dir, to_export)
        outside_roi_n = self._export_outside_roi(
            outside_entries, folder_name, cache_root=out_dir, from_entries=True
        )
        return len(to_export), len(sorted_entries), outside_roi_n

    def _cleanup_empty_recognition_folder(self, out_dir):
        """Remove recognition handoff folder when it has no face exports."""
        if not os.path.isdir(out_dir):
            return
        if self._folder_has_face_export(out_dir):
            return
        for name in list(os.listdir(out_dir)):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        try:
            os.rmdir(out_dir)
            print(f"Removed empty recognition folder {out_dir}")
        except OSError:
            print(f"No face exports for {out_dir} — removed artifacts (folder not empty)")

    def _low_ofiq_dir_for_batch(self, folder_name):
        if self.merge_batches:
            track_id = parse_track_id_from_folder(folder_name)
            if track_id is not None:
                return os.path.join(self.low_ofiq_dir, f"person_{track_id}")
        return os.path.join(self.low_ofiq_dir, folder_name)

    def _export_below_ofiq(self, candidates, folder_name):
        """Save sub-threshold faces to low_ofiq_dir (no .ready / no watcher)."""
        below = [
            c
            for c in candidates
            if c.get("gate") == "below_ofiq" and c.get("face_crop") is not None
        ]
        if not below:
            return 0
        below.sort(key=lambda c: c["rank"], reverse=True)
        out_dir = self._low_ofiq_dir_for_batch(folder_name)
        os.makedirs(out_dir, exist_ok=True)
        export_count = max(
            self.min_export,
            min(self.export_top_n, len(below)),
        )
        to_export = below[:export_count]
        self._write_exports(out_dir, to_export)
        print(
            f"  below OFIQ (< {self.ofiq_threshold}) -> {out_dir} "
            f"({len(to_export)} face(s))"
        )
        return len(to_export)

    def _outside_roi_dir_for_batch(self, folder_name):
        if self.merge_batches:
            track_id = parse_track_id_from_folder(folder_name)
            if track_id is not None:
                return os.path.join(self.outside_roi_dir, f"person_{track_id}")
        return os.path.join(self.outside_roi_dir, folder_name)

    def _export_outside_roi(self, outside, folder_name, *, cache_root=None, from_entries=False):
        """Save passing faces outside ROI (no .ready / no watcher)."""
        if not outside or not self.roi or not folder_name:
            return 0

        items = []
        if from_entries:
            for entry in outside:
                item = {
                    "rank": tuple(entry["rank"]),
                    "ofiq": entry["ofiq"],
                    "combined": entry["combined"],
                    "gate": entry["gate"],
                    "source": entry["source"],
                    "frame_num": entry["frame_num"],
                    "meta": entry["meta"],
                    "full_path": entry["full_path"],
                    "face_crop": None,
                }
                if entry.get("face_cache") and cache_root:
                    cache_path = os.path.join(cache_root, entry["face_cache"])
                    if os.path.isfile(cache_path):
                        item["face_crop"] = cv2.imread(cache_path)
                if item["face_crop"] is not None:
                    items.append(item)
        else:
            items = [c for c in outside if c.get("face_crop") is not None]

        if not items:
            return 0

        items.sort(key=lambda c: c["rank"], reverse=True)
        out_dir = self._outside_roi_dir_for_batch(folder_name)
        os.makedirs(out_dir, exist_ok=True)
        export_count = max(
            self.min_export,
            min(self.export_top_n, len(items)),
        )
        to_export = items[:export_count]
        self._write_exports(out_dir, to_export)
        print(f"  outside ROI -> {out_dir} ({len(to_export)} face(s))")
        return len(to_export)

    def _rejected_dir_for_batch(self, folder_name):
        if self.merge_batches:
            track_id = parse_track_id_from_folder(folder_name)
            if track_id is not None:
                return os.path.join(self.rejected_dir, f"person_{track_id}")
        return os.path.join(self.rejected_dir, folder_name)

    @staticmethod
    def _format_reject_stats(stats, sample_count):
        return (
            f"samples={sample_count}, no_face={stats['no_face']}, "
            f"receding={stats['receding']}, back_facing={stats['back_facing']}, "
            f"gated={stats.get('gated', 0)}"
        )

    def _recognition_diagnosis(self, candidates):
        """Summarize OFIQ/ROI state for reject reports."""
        if not candidates:
            return {}
        passing = [c for c in candidates if c.get("gate") == "ok"]
        below = [c for c in candidates if c.get("gate") == "below_ofiq"]
        in_roi = [c for c in passing if self._candidate_in_roi(c)]
        outside = [c for c in passing if not self._candidate_in_roi(c)]
        diagnosis = {
            "passing_ofiq_count": len(passing),
            "below_ofiq_count": len(below),
            "passing_in_roi_count": len(in_roi),
            "outside_roi_count": len(outside),
        }
        if passing:
            best = max(passing, key=lambda c: c["rank"])
            diagnosis["best_passing"] = {
                "frame": best["frame_num"],
                "ofiq": best.get("ofiq", 0),
                "in_roi": self._candidate_in_roi(best),
                "source": best.get("source"),
            }
        if below:
            best_below = max(below, key=lambda c: c.get("ofiq", 0))
            diagnosis["best_below_ofiq"] = {
                "frame": best_below["frame_num"],
                "ofiq": best_below.get("ofiq", 0),
            }
        return diagnosis

    def _recognition_failure_reason(self, candidates, exported_count, stats, rejected):
        """
        Why nothing reached recognition_folder — not why alternate folders have files.
        """
        if exported_count > 0:
            return None

        passing = [c for c in (candidates or []) if c.get("gate") == "ok"]
        if passing:
            in_roi = [c for c in passing if self._candidate_in_roi(c)]
            if not in_roi:
                return "outside_roi"
            return "export_failed"

        below = [c for c in (candidates or []) if c.get("gate") == "below_ofiq"]
        if below:
            return "below_ofiq"

        if rejected:
            return Counter(
                item.get("reason", "unknown") for item in rejected
            ).most_common(1)[0][0]

        if stats.get("no_face"):
            return "no_face"
        if stats.get("back_facing"):
            return "back_facing"
        if stats.get("receding"):
            return "receding"
        if stats.get("gated"):
            return "gated"
        return "unknown"

    def _clear_rejected_dir_for_batch(self, folder_name):
        out_dir = self._rejected_dir_for_batch(folder_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

    def _export_rejected_batch(
        self,
        batch_dir,
        folder_name,
        samples,
        stats,
        rejected,
        *,
        candidates=None,
        primary_reason=None,
        low_ofiq_n=0,
        outside_roi_n=0,
        alternate_dirs=None,
    ):
        """Debug folder when a batch does not produce recognition exports."""
        if not samples:
            return

        primary_reason = primary_reason or self._recognition_failure_reason(
            candidates, 0, stats, rejected
        )
        out_dir = self._rejected_dir_for_batch(folder_name)
        os.makedirs(out_dir, exist_ok=True)

        best = max(samples, key=lambda s: s["person_area"])
        frame_num = best["frame_num"]
        for kind in ("full", "person"):
            src = os.path.join(batch_dir, f"frame_{frame_num}_{kind}.jpg")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_dir, f"frame_{frame_num}_{kind}.jpg"))

        if rejected:
            best_reject = max(
                rejected,
                key=lambda item: (item.get("metrics") or {}).get("combined", 0),
            )
            reject_frame = best_reject["frame_num"]
            reason = best_reject["reason"]
            cv2.imwrite(
                os.path.join(out_dir, f"frame_{reject_frame}_face_{reason}.jpg"),
                best_reject["face_crop"],
            )
        elif candidates:
            best_candidate = max(candidates, key=lambda c: c["rank"])
            if best_candidate.get("face_crop") is not None:
                gate = best_candidate.get("gate", "unknown")
                cv2.imwrite(
                    os.path.join(
                        out_dir,
                        f"frame_{best_candidate['frame_num']}_face_{gate}.jpg",
                    ),
                    best_candidate["face_crop"],
                )

        summary = {
            "batch": folder_name,
            "primary_reason": primary_reason,
            "ofiq_threshold": self.ofiq_threshold,
            "camera_profile": self.config.get("camera_profile", "standard"),
            "queue_dir": os.path.abspath(batch_dir),
            "stats": stats,
            "best_sample_frame": frame_num,
            "alternate_dirs": alternate_dirs or {},
            "recognition_diagnosis": self._recognition_diagnosis(candidates),
            "candidate_gates": [
                {
                    "frame": c["frame_num"],
                    "gate": c.get("gate"),
                    "ofiq": c.get("ofiq"),
                    "source": c.get("source"),
                }
                for c in (candidates or [])
            ],
            "rejected_face_attempts": [
                {
                    "frame": item["frame_num"],
                    "reason": item["reason"],
                    "metrics": item.get("metrics"),
                }
                for item in rejected
            ],
        }
        with open(os.path.join(out_dir, "_reject_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        reason_line = (
            f"PRIMARY REASON: {primary_reason}\n"
            f"OFIQ threshold: {self.ofiq_threshold} (pass if score >= threshold)\n"
            f"{self._format_reject_stats(stats, len(samples))}\n"
        )
        diagnosis = self._recognition_diagnosis(candidates)
        if diagnosis.get("best_passing"):
            bp = diagnosis["best_passing"]
            reason_line += (
                f"Best OFIQ-passing frame {bp['frame']}: ofiq={bp['ofiq']:.2f}, "
                f"in_roi={bp['in_roi']}, source={bp.get('source')}\n"
            )
        if diagnosis.get("below_ofiq_count"):
            reason_line += (
                f"Also exported {diagnosis['below_ofiq_count']} lower-OFIQ frame(s) "
                f"to low_ofiq_faces (not the recognition blocker if a passing face exists)\n"
            )
        if diagnosis.get("outside_roi_count"):
            reason_line += (
                f"OFIQ-passing outside ROI: {diagnosis['outside_roi_count']} frame(s) "
                f"-> outside_roi_faces\n"
            )
        if alternate_dirs:
            for label, path in alternate_dirs.items():
                if path:
                    reason_line += f"{label}: {path}\n"
        with open(os.path.join(out_dir, "REJECT_REASON.txt"), "w", encoding="utf-8") as f:
            f.write(reason_line)

        print(
            f"  not recognized -> {out_dir} "
            f"(primary={primary_reason}, {self._format_reject_stats(stats, len(samples))})"
        )

    def _output_dir_for_batch(self, folder_name):
        if self.merge_batches:
            track_id = parse_track_id_from_folder(folder_name)
            if track_id is not None:
                return os.path.join(self.output_dir, f"person_{track_id}")
        return os.path.join(self.output_dir, folder_name)

    def _alternate_export_dirs(self, folder_name, low_ofiq_n, outside_roi_n):
        alternates = {}
        if low_ofiq_n > 0:
            alternates["low_ofiq"] = self._low_ofiq_dir_for_batch(folder_name)
        if outside_roi_n > 0:
            alternates["outside_roi"] = self._outside_roi_dir_for_batch(folder_name)
        return alternates

    def process_batch(self, batch_dir, folder_name):
        samples = []
        for frame_num, meta, full_path, person_path in iter_staging_samples(batch_dir):
            sample = self._load_sample(frame_num, meta, full_path, person_path, batch_dir)
            if sample:
                samples.append(sample)

        if not samples:
            print(f"No samples in {folder_name}")
            return False

        candidates, stats, rejected = self._collect_candidates(samples)
        out_dir = self._output_dir_for_batch(folder_name)
        low_ofiq_n = 0
        outside_roi_n = 0
        exported_count = 0

        if not candidates:
            self._export_rejected_batch(
                batch_dir,
                folder_name,
                samples,
                stats,
                rejected,
                primary_reason=self._recognition_failure_reason([], 0, stats, rejected),
            )
            self._cleanup_empty_recognition_folder(out_dir)
            print(
                f"No export for {folder_name}: {self._format_reject_stats(stats, len(samples))}. "
                f"See {self._rejected_dir_for_batch(folder_name)}/REJECT_REASON.txt"
            )
            return False

        candidates.sort(key=lambda c: c["rank"], reverse=True)
        export_pool = self._exportable_candidates(candidates)
        roi_candidates, outside_candidates = self._filter_candidates_by_roi(export_pool)

        if self.merge_batches:
            os.makedirs(out_dir, exist_ok=True)
            exported_count, pool_size, outside_roi_n = self._accumulate_and_export_merged(
                out_dir, candidates, folder_name
            )
            low_ofiq_n = self._export_below_ofiq(candidates, folder_name)
            print(
                f"Batch {folder_name}: exported={exported_count}/{pool_size} -> {out_dir} "
                f"(normal={stats['normal']}, lenient={stats['lenient']}, "
                f"fallback={stats['fallback']}, receding={stats['receding']}, "
                f"back_facing={stats['back_facing']}, gated={stats.get('gated', 0)}, "
                f"outside_roi={outside_roi_n}, below_ofiq={low_ofiq_n}, "
                f"no_face_frames={stats['no_face']})"
            )
        else:
            if roi_candidates:
                os.makedirs(out_dir, exist_ok=True)
            export_count = max(self.min_export, min(self.export_top_n, len(roi_candidates)))
            to_export = roi_candidates[:export_count]
            if to_export:
                self._write_exports(out_dir, to_export)
                exported_count = len(to_export)
            outside_roi_n = self._export_outside_roi(outside_candidates, folder_name)
            low_ofiq_n = self._export_below_ofiq(candidates, folder_name)
            print(
                f"Batch {folder_name}: exported={exported_count}/{len(candidates)} -> {out_dir} "
                f"(normal={stats['normal']}, lenient={stats['lenient']}, "
                f"fallback={stats['fallback']}, receding={stats['receding']}, "
                f"back_facing={stats['back_facing']}, gated={stats.get('gated', 0)}, "
                f"outside_roi={outside_roi_n}, below_ofiq={low_ofiq_n}, "
                f"no_face_frames={stats['no_face']})"
            )

        has_recognition_export = self._folder_has_face_export(out_dir)
        alternates = self._alternate_export_dirs(folder_name, low_ofiq_n, outside_roi_n)

        if has_recognition_export:
            self._clear_rejected_dir_for_batch(folder_name)
            clear_processed(out_dir)
            if self.write_ready and not self.rank_only:
                write_ready(out_dir)
            return True

        self._cleanup_empty_recognition_folder(out_dir)
        primary_reason = self._recognition_failure_reason(
            candidates, exported_count, stats, rejected
        )
        self._export_rejected_batch(
            batch_dir,
            folder_name,
            samples,
            stats,
            rejected,
            candidates=candidates,
            primary_reason=primary_reason,
            low_ofiq_n=low_ofiq_n,
            outside_roi_n=outside_roi_n,
            alternate_dirs=alternates,
        )
        print(
            f"No recognition export for {folder_name} (primary={primary_reason}). "
            f"See {self._rejected_dir_for_batch(folder_name)}/REJECT_REASON.txt"
        )
        return False

    def run_once(self):
        processed_any = False
        for _, folder_name, batch_dir, _ in get_pending_folders(self.queue_dir):
            if not folder_inactive(batch_dir, self.inactivity_wait):
                continue
            print(f"Ranking {batch_dir}...")
            self.process_batch(batch_dir, folder_name)
            mark_processed(batch_dir)
            processed_any = True
        return processed_any

    def run_forever(self):
        merge_note = "one folder per track ID" if self.merge_batches else "one folder per batch"
        print(
            f"Face ranker watching '{self.queue_dir}' -> '{self.output_dir}' ({merge_note})"
        )
        if self.rank_only:
            print(
                f"Top {self.export_top_n} per person; min {self.min_export}; "
                f"detect-only (ranked_faces). Use --with-recognition for API stage."
            )
        else:
            print(
                f"Top {self.export_top_n} per person -> recognition; "
                f"write_ready={self.write_ready}; "
                f"minimal_export={self.recognition_minimal}"
            )
        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"Face ranker error: {e}")
                traceback.print_exc()
            time.sleep(self.check_interval)


def main():
    FaceRankerService().run_forever()


if __name__ == "__main__":
    main()
