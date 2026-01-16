import argparse
import os
import re

import numpy as np

from av1sim.codec import decode_bitstream, encode_video
from av1sim.io import read_video_frames, read_yuv420_frames


def main():
    parser = argparse.ArgumentParser(description="Compression metrics + RD curve")
    parser.add_argument("input", help="Input video path (.yuv or .mp4)")
    parser.add_argument("--bitstream", help="Existing .av1s bitstream for single metrics")
    parser.add_argument("--yuv-width", type=int, default=0)
    parser.add_argument("--yuv-height", type=int, default=0)
    parser.add_argument("--yuv-fps", type=float, default=0.0)
    parser.add_argument("--qp-list", default="12,16,20,24,28,32")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--search", type=int, default=8)
    parser.add_argument("--mode", choices=["full", "diamond"], default="diamond")
    parser.add_argument("--transform", choices=["dct", "wht"], default="dct")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-csv", default="rd_curve.csv")
    args = parser.parse_args()

    frames, fps = _read_input_frames(
        args.input, args.yuv_width, args.yuv_height, args.yuv_fps
    )
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    duration = len(frames) / fps if fps else 0.0

    if args.bitstream:
        metrics = _metrics_for_bitstream(args.bitstream, frames, fps, duration)
        _print_single(metrics)
        return

    qps = [int(q.strip()) for q in args.qp_list.split(",") if q.strip()]
    rows = []
    for qp in qps:
        out_path = f"tmp_qp_{qp}.av1s"
        encode_video(
            frames,
            fps,
            out_path,
            block_size=args.block,
            qp=qp,
            search_range=args.search,
            mode=args.mode,
            transform=args.transform,
        )
        metrics = _metrics_for_bitstream(out_path, frames, fps, duration)
        metrics["qp"] = qp
        rows.append(metrics)
        os.remove(out_path)

    _write_csv(args.output_csv, rows)
    _print_table(rows, args.output_csv)


def _metrics_for_bitstream(bitstream, original_frames, fps, duration):
    decoded_frames, _ = decode_bitstream(bitstream)
    if len(decoded_frames) != len(original_frames):
        min_len = min(len(decoded_frames), len(original_frames))
        decoded_frames = decoded_frames[:min_len]
        original_frames = original_frames[:min_len]

    psnr = _psnr_luma(original_frames, decoded_frames)
    out_size = os.path.getsize(bitstream)
    bitrate_kbps = (out_size * 8 / duration / 1000.0) if duration > 0 else 0.0
    in_size = _input_size_bytes(original_frames)
    ratio = out_size / in_size if in_size else 0.0
    return {
        "bitrate_kbps": bitrate_kbps,
        "psnr": psnr,
        "input_bytes": in_size,
        "output_bytes": out_size,
        "ratio": ratio,
    }


def _input_size_bytes(frames):
    if not frames:
        return 0
    h, w = frames[0].shape[:2]
    return int(len(frames) * w * h * 3 // 2)


def _psnr_luma(orig_frames, dec_frames):
    mse_sum = 0.0
    count = 0
    for orig, dec in zip(orig_frames, dec_frames):
        o = _to_luma(orig)
        d = _to_luma(dec)
        diff = o - d
        mse_sum += float(np.mean(diff * diff))
        count += 1
    if count == 0:
        return 0.0
    mse = mse_sum / count
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def _to_luma(frame):
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _read_input_frames(path, width, height, fps):
    if path.lower().endswith(".yuv"):
        width, height, fps = _infer_yuv_params(path, width, height, fps)
        frames = read_yuv420_frames(path, width, height)
        return frames, fps
    return read_video_frames(path)


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


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("qp,bitrate_kbps,psnr,input_bytes,output_bytes,ratio\n")
        for row in rows:
            handle.write(
                f"{row['qp']},{row['bitrate_kbps']:.2f},{row['psnr']:.2f},"
                f"{row['input_bytes']},{row['output_bytes']},{row['ratio']:.4f}\n"
            )


def _print_table(rows, csv_path):
    print("QP  Bitrate(kbps)  PSNR(dB)  Ratio")
    for row in rows:
        print(
            f"{row['qp']:<3} {row['bitrate_kbps']:<13.2f} {row['psnr']:<8.2f} {row['ratio']:.4f}"
        )
    print(f"Saved CSV: {csv_path}")


def _print_single(metrics):
    print("Compression ratio table")
    print(f"Input bytes:  {metrics['input_bytes']}")
    print(f"Output bytes: {metrics['output_bytes']}")
    print(f"Ratio:        {metrics['ratio']:.4f}")
    print("")
    print("RD point")
    print(f"Bitrate (kbps): {metrics['bitrate_kbps']:.2f}")
    print(f"PSNR (dB):      {metrics['psnr']:.2f}")


if __name__ == "__main__":
    main()
