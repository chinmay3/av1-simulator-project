import argparse

import numpy as np

from av1sim.intra import intra_predict_frame
from av1sim.io import read_video_frames, write_video_frames


def main():
    parser = argparse.ArgumentParser(description="Intra prediction visualization")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--mode", choices=["dc", "horizontal", "vertical", "angular"], default="dc")
    parser.add_argument("--angle", type=float, default=45.0)
    parser.add_argument("--max-size", type=int, default=64)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--variance", type=float, default=100.0)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    predicted_frames = []
    for frame in frames:
        pred = intra_predict_frame(
            frame,
            mode=args.mode,
            angle=args.angle,
            max_size=args.max_size,
            min_size=args.min_size,
            variance_threshold=args.variance,
        )
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        bgr = np.stack([pred, pred, pred], axis=2)
        predicted_frames.append(bgr)

    write_video_frames(args.output, predicted_frames, fps)


if __name__ == "__main__":
    main()
