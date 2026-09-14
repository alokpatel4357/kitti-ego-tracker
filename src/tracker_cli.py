import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import argparse
import time
import cv2
from src.flow_math import (
    compute_dense_flow,
    estimate_epipolar_inliers,
    isolate_residual_motion_pca,
    extract_moving_clusters,
)


def process_sequence(input_dir, output_path, fps=10, motion_thresh=2.2, max_frames=None):
    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' not found.")
        sys.exit(1)

    frames = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
    )

    if len(frames) < 2:
        print(f"Error: need at least 2 frames to compute motion, found {len(frames)}.")
        sys.exit(1)

    if max_frames:
        frames = frames[:max_frames]

    first_img = cv2.imread(frames[0])
    h, w = first_img.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Input: {input_dir} ({len(frames)} frames)")
    print(f"Output: {output_path}")
    print(f"Resolution: {w}x{h} @ {fps} fps")
    print(f"Threshold: {motion_thresh}px")
    print("Processing...")

    prev_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    t_start = time.time()

    for i in range(1, len(frames)):
        curr_frame = cv2.imread(frames[i])
        if curr_frame is None:
            continue

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Optical flow
        flow = compute_dense_flow(prev_gray, curr_gray)

        # 2. Filter background using RANSAC
        inliers, _ = estimate_epipolar_inliers(flow)

        # 3. Model camera motion with PCA & subtract
        residual, _ = isolate_residual_motion_pca(flow, inliers)

        # 4. Cluster residual points into object bounding boxes
        boxes, _ = extract_moving_clusters(residual, motion_threshold=motion_thresh)

        # Render bounding boxes
        annotated = curr_frame.copy()
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"Obj {idx + 1}",
                (bx, max(15, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Top-left info display
        info = f"Frame: {i}/{len(frames) - 1} | Objects: {len(boxes)}"
        cv2.putText(annotated, info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(annotated)
        prev_gray = curr_gray

        if i % 10 == 0 or i == len(frames) - 1:
            print(f"  Frame {i}/{len(frames) - 1} completed")

    out.release()
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.2f}s (avg {(len(frames) - 1) / elapsed:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Ego-motion compensation tracker on KITTI sequence")
    parser.add_argument("--input_dir", type=str, default="data/raw_frames", help="Path to input frames folder")
    parser.add_argument("--output_video", type=str, default="output/kitti_tracked.mp4", help="Path to output mp4 file")
    parser.add_argument("--fps", type=int, default=10, help="Output framerate")
    parser.add_argument("--motion_thresh", type=float, default=2.2, help="Pixel displacement threshold")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")

    args = parser.parse_args()
    process_sequence(
        args.input_dir,
        args.output_video,
        args.fps,
        args.motion_thresh,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
