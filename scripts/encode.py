import argparse
import os
import re

from av1sim.codec import encode_video
from av1sim.io import read_video_frames, read_yuv420_frames


def main():
    parser = argparse.ArgumentParser(description="AV1SIM encoder (parts 1-8)")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output bitstream path (.av1s)")
    parser.add_argument("--yuv-width", type=int, default=0, help="Width for raw .yuv input")
    parser.add_argument("--yuv-height", type=int, default=0, help="Height for raw .yuv input")
    parser.add_argument("--yuv-fps", type=float, default=0.0, help="FPS for raw .yuv input")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--qp", type=float, default=20.0)
    parser.add_argument("--min-qp", type=float, default=8.0)
    parser.add_argument("--max-qp", type=float, default=40.0)
    parser.add_argument("--target-kbps", type=float, default=0.0)
    parser.add_argument("--search", type=int, default=8)
    parser.add_argument("--mode", choices=["full", "diamond"], default="diamond")
    parser.add_argument("--transform", choices=["dct", "wht"], default="dct")
    parser.add_argument("--grain", action="store_true")
    args = parser.parse_args()

    if args.input.lower().endswith(".yuv"):
        width, height, fps = _infer_yuv_params(
            args.input, args.yuv_width, args.yuv_height, args.yuv_fps
        )
        frames = read_yuv420_frames(args.input, width, height)
    else:
        frames, fps = read_video_frames(args.input)
    encode_video(
        frames,
        fps,
        args.output,
        block_size=args.block,
        qp=args.qp,
        min_qp=args.min_qp,
        max_qp=args.max_qp,
        target_kbps=args.target_kbps,
        search_range=args.search,
        mode=args.mode,
        transform=args.transform,
        use_grain=args.grain,
    )

    input_size = os.path.getsize(args.input)
    output_size = os.path.getsize(args.output)
    ratio = output_size / input_size if input_size else 0.0
    print(f"Input size: {input_size} bytes")
    print(f"Output size: {output_size} bytes")
    print(f"Compression ratio: {ratio:.3f}x")


def _infer_yuv_params(path, width, height, fps):
    if width <= 0 or height <= 0:
        dims = _parse_dims(path)
        if dims:
            width, height = dims
    if fps <= 0:
        fps = _parse_fps(path)
    if width <= 0 or height <= 0:
        raise ValueError("Provide --yuv-width and --yuv-height for .yuv input")
    if fps <= 0:
        fps = 30.0
    return width, height, fps


def _parse_dims(path):
    match = re.search(r"(\d+)x(\d+)", path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _parse_fps(path):
    match = re.search(r"_(\d+)_", path)
    if match:
        return float(match.group(1))
    return 0.0


if __name__ == "__main__":
    main()
