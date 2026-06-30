"""Offline calibration: run Stage 2 face+OFIQ on a person_queue batch and print score stats."""

import argparse
import os
import statistics

import cv2

from face_quality import FaceQualityEngine
from ofiq_scorer import OFIQScorer
from pipeline_io import crop_person_from_full, iter_staging_samples, load_config


def calibrate_batch(batch_dir, config=None):
    config = config or load_config()
    face_engine = FaceQualityEngine(config)
    ofiq = OFIQScorer(config["ofiq"]["model"])
    threshold = config["ofiq"]["threshold"]

    rows = []
    for frame_num, meta, full_path, person_path in iter_staging_samples(batch_dir):
        full_frame = cv2.imread(full_path)
        if full_frame is None:
            continue
        person_crop = cv2.imread(person_path) if os.path.isfile(person_path) else crop_person_from_full(full_frame, meta)
        face_result, reject = face_engine.find_face_in_sample(full_frame, meta, person_crop)
        row = {"frame": frame_num, "gate_reject": reject}
        if face_result is None:
            rows.append(row)
            continue
        m = face_result["metrics"]
        row.update(
            {
                "blur": m["blur"],
                "frontality": m["frontality"],
                "completeness": m["completeness"],
                "eyes": m["eyes"],
                "mouth": m["mouth"],
                "combined": m["combined"],
            }
        )
        if reject:
            rows.append(row)
            continue
        ofiq_score = ofiq.get_score(face_result["face_crop"])
        row["ofiq"] = ofiq_score
        row["would_export"] = ofiq_score >= threshold
        rows.append(row)

    print(f"\n=== Calibration: {batch_dir} ===")
    print(f"Samples: {len(rows)}")
    gate_fails = [r for r in rows if r.get("gate_reject") and r["gate_reject"] != "no_face"]
    no_face = [r for r in rows if r.get("gate_reject") == "no_face"]
    passed_gates = [r for r in rows if r.get("gate_reject") is None]
    exported = [r for r in passed_gates if r.get("would_export")]

    print(f"  no_face: {len(no_face)}")
    print(f"  gate_fail: {len(gate_fails)}")
    print(f"  passed_gates: {len(passed_gates)}")
    print(f"  would_export (OFIQ>{threshold}): {len(exported)}")

    if passed_gates:
        ofiq_scores = [r["ofiq"] for r in passed_gates if "ofiq" in r]
        if ofiq_scores:
            print(
                f"  OFIQ on gate-pass: min={min(ofiq_scores):.2f} "
                f"max={max(ofiq_scores):.2f} mean={statistics.mean(ofiq_scores):.2f}"
            )

    print("\nPer-frame detail:")
    for r in rows:
        if r.get("gate_reject"):
            print(f"  frame {r['frame']}: REJECT {r['gate_reject']}")
        elif r.get("would_export"):
            print(f"  frame {r['frame']}: EXPORT ofiq={r['ofiq']:.2f} combined={r['combined']:.1f}")
        else:
            print(
                f"  frame {r['frame']}: low_ofiq={r.get('ofiq', 0):.2f} "
                f"combined={r['combined']:.1f} front={r['frontality']:.0f}"
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Calibrate face/OFIQ thresholds on a queue batch")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to person_queue/person_ID_bN_timestamp folder",
    )
    parser.add_argument("--config", default="pipeline_config.yaml")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Not a directory: {args.input}")
        return

    config = load_config(args.config)
    calibrate_batch(args.input, config)


if __name__ == "__main__":
    main()
