"""Stage 2: Watch person_queue/, rank faces, export best per person to ranked_faces/."""

import json
import os
import shutil
import time

import cv2

from face_quality import FaceQualityEngine
from ofiq_scorer import OFIQScorer
from pipeline_io import (
    clamp_bbox,
    crop_person_from_full,
    ensure_dir,
    get_pending_folders,
    iter_staging_samples,
    load_config,
    mark_processed,
    folder_inactive,
    parse_track_id_from_folder,
    write_ready,
)


RANK_MANIFEST = "_rank_manifest.json"
FACE_CACHE_DIR = ".face_cache"


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
        self.write_ready = True if recognition_mode else ranker_cfg.get("write_ready", False)
        self.min_export = ranker_cfg.get("min_export_per_person", 1)
        self.always_export_person = ranker_cfg.get("always_export_person", True)
        self.skip_fallback_edge_bbox = ranker_cfg.get("skip_fallback_edge_bbox", True)
        self.max_bbox_y1_ratio = ranker_cfg.get(
            "max_bbox_y1_ratio", self.config["person"].get("max_bbox_y1_ratio", 0.72)
        )
        self.ofiq_threshold = ofiq_cfg["threshold"]
        self.export_top_n = ofiq_cfg["export_top_n"]
        self.check_interval = watcher_cfg.get("check_interval_sec", 3)
        self.inactivity_wait = watcher_cfg.get("inactivity_wait_sec", 5)

        self.ofiq = OFIQScorer(ofiq_cfg["model"])
        self.face_engine = FaceQualityEngine(self.config, ofiq_scorer=self.ofiq)
        ensure_dir(self.queue_dir)
        ensure_dir(self.output_dir)

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
        ofiq_score = self.ofiq.get_score(face_result["face_crop"])
        combined = face_result["metrics"]["combined"]
        passes_ofiq = ofiq_score > self.ofiq_threshold
        gate = face_result.get("gate_reject") or "ok"
        if not passes_ofiq:
            gate = gate if gate != "ok" else "below_ofiq"
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
        }

    def _collect_candidates(self, samples):
        candidates = []
        stats = {"no_face": 0, "receding": 0, "back_facing": 0, "normal": 0, "lenient": 0, "fallback": 0}

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
                    continue
                if reject == "no_face":
                    stats["no_face"] += 1
                    continue
                stats["lenient"] += 1
                candidates.append(
                    self._candidate_from_face(sample, face, tier=2, source="lenient")
                )
                continue

            stats["normal"] += 1
            candidates.append(
                self._candidate_from_face(sample, face, tier=3, source="normal")
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

        return candidates, stats

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
        }


    def _accumulate_and_export_merged(self, out_dir, new_candidates):
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
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"entries": sorted_entries}, f, indent=2)

        export_count = max(
            self.min_export,
            min(self.export_top_n, len(sorted_entries)),
        )
        top_entries = sorted_entries[:export_count]
        self._clear_frame_exports(out_dir)
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
        self._write_exports(out_dir, to_export)
        return len(to_export), len(sorted_entries)

    def _cleanup_empty_recognition_folder(self, out_dir):
        """No face exports in API mode — drop manifest-only folders."""
        if not self.recognition_minimal or self._folder_has_face_export(out_dir):
            return
        manifest_path = os.path.join(out_dir, RANK_MANIFEST)
        if os.path.isfile(manifest_path):
            os.remove(manifest_path)
        cache_dir = os.path.join(out_dir, FACE_CACHE_DIR)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        ready_file = os.path.join(out_dir, ".ready")
        if os.path.isfile(ready_file):
            os.remove(ready_file)
        print(f"No face exports for {out_dir} — removed manifest-only artifacts")

    def _output_dir_for_batch(self, folder_name):
        if self.merge_batches:
            track_id = parse_track_id_from_folder(folder_name)
            if track_id is not None:
                return os.path.join(self.output_dir, f"person_{track_id}")
        return os.path.join(self.output_dir, folder_name)

    def process_batch(self, batch_dir, folder_name):
        samples = []
        for frame_num, meta, full_path, person_path in iter_staging_samples(batch_dir):
            sample = self._load_sample(frame_num, meta, full_path, person_path, batch_dir)
            if sample:
                samples.append(sample)

        if not samples:
            print(f"No samples in {folder_name}")
            return False

        candidates, stats = self._collect_candidates(samples)
        if not candidates:
            print(f"No export for {folder_name} (empty batch)")
            return False

        candidates.sort(key=lambda c: c["rank"], reverse=True)
        out_dir = self._output_dir_for_batch(folder_name)
        os.makedirs(out_dir, exist_ok=True)

        if self.merge_batches:
            exported, pool_size = self._accumulate_and_export_merged(out_dir, candidates)
            print(
                f"Exported {exported}/{pool_size} global best from {folder_name} -> {out_dir} "
                f"(normal={stats['normal']}, lenient={stats['lenient']}, "
                f"fallback={stats['fallback']}, receding={stats['receding']}, "
                f"back_facing={stats['back_facing']}, no_face_frames={stats['no_face']})"
            )
        else:
            export_count = max(self.min_export, min(self.export_top_n, len(candidates)))
            to_export = candidates[:export_count]
            self._write_exports(out_dir, to_export)
            self._cleanup_empty_recognition_folder(out_dir)
            print(
                f"Exported {len(to_export)}/{len(candidates)} from {folder_name} -> {out_dir} "
                f"(normal={stats['normal']}, lenient={stats['lenient']}, "
                f"fallback={stats['fallback']}, receding={stats['receding']}, "
                f"back_facing={stats['back_facing']}, no_face_frames={stats['no_face']})"
            )

        if self.merge_batches:
            self._cleanup_empty_recognition_folder(out_dir)

        if self.write_ready and not self.rank_only:
            if self.recognition_minimal and not self._folder_has_face_export(out_dir):
                print(f"No .ready for {out_dir} — no face image to send to API")
            else:
                write_ready(out_dir)

        return True

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
            time.sleep(self.check_interval)


def main():
    FaceRankerService().run_forever()


if __name__ == "__main__":
    main()
