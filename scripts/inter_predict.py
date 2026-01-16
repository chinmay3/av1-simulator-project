import argparse

import cv2
import numpy as np

from av1sim.inter import inter_predict_frame, serialize_motion_vectors
from av1sim.io import read_video_frames, write_video_frames


def main():
    parser = argparse.ArgumentParser(description="Inter prediction visualization")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("pred_output", help="Predicted video path")
    parser.add_argument("--residual-output", help="Residual video path")
    parser.add_argument("--mv-output", help="Motion vectors JSONL path")
    parser.add_argument("--combo-output", help="Side-by-side original/pred/residual path")
    parser.add_argument("--mode", choices=["full", "diamond"], default="full")
    parser.add_argument("--search", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=64)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--variance", type=float, default=100.0)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    pred_frames = []
    residual_frames = []
    mv_lines = []
    combo_frames = []

    prev_frame = frames[0]
    first_luma = np.mean(prev_frame.astype(np.float32), axis=2)
    first_bgr = np.stack([first_luma, first_luma, first_luma], axis=2).astype(np.uint8)
    pred_frames.append(first_bgr)
    residual_frames.append(np.full_like(first_bgr, 128, dtype=np.uint8))
    if args.combo_output:
        combo_frames.append(_build_combo(prev_frame, first_bgr, residual_frames[-1], []))

    for frame in frames[1:]:
        pred, residual, vectors = inter_predict_frame(
            frame,
            prev_frame,
            mode=args.mode,
            search_range=args.search,
            max_size=args.max_size,
            min_size=args.min_size,
            variance_threshold=args.variance,
        )
        pred_vis = np.clip(pred, 0, 255).astype(np.uint8)
        pred_bgr = np.stack([pred_vis, pred_vis, pred_vis], axis=2)
        pred_frames.append(pred_bgr)

        resid_vis = np.clip(residual + 128.0, 0, 255).astype(np.uint8)
        resid_bgr = np.stack([resid_vis, resid_vis, resid_vis], axis=2)
        residual_frames.append(resid_bgr)

        mv_lines.append(serialize_motion_vectors(vectors))
        if args.combo_output:
            combo_frames.append(_build_combo(frame, pred_bgr, resid_bgr, vectors))
        prev_frame = frame

    write_video_frames(args.pred_output, pred_frames, fps)
    if args.residual_output:
        write_video_frames(args.residual_output, residual_frames, fps)
    if args.mv_output:
        with open(args.mv_output, "w", encoding="utf-8") as handle:
            handle.write("\n".join(mv_lines))
    if args.combo_output:
        write_video_frames(args.combo_output, combo_frames, fps)


def _build_combo(original, predicted, residual, vectors):
    pred_overlay = _draw_motion_vectors(predicted, vectors)
    return np.concatenate([original, pred_overlay, residual], axis=1)


def _draw_motion_vectors(frame, vectors, color=(0, 200, 255), thickness=1, tip_length=0.3):
    overlay = frame.copy()
    for mv in vectors:
        cx = mv.x + mv.size // 2
        cy = mv.y + mv.size // 2
        ex = cx + mv.dx
        ey = cy + mv.dy
        cv2.arrowedLine(
            overlay,
            (int(cx), int(cy)),
            (int(ex), int(ey)),
            color,
            thickness,
            tipLength=tip_length,
        )
    return overlay


if __name__ == "__main__":
    main()
