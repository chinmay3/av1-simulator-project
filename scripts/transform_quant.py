import argparse

import numpy as np

from av1sim.io import read_video_frames, write_video_frames
from av1sim.transform import transform_quant_frame


def main():
    parser = argparse.ArgumentParser(description="Transform + quantization demo")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--mode", choices=["dct", "wht"], default="dct")
    parser.add_argument("--qp", type=float, default=20.0)
    parser.add_argument("--qp-mode", choices=["constant", "adaptive"], default="constant")
    parser.add_argument("--min-qp", type=float, default=10.0)
    parser.add_argument("--max-qp", type=float, default=40.0)
    parser.add_argument("--max-size", type=int, default=64)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--variance", type=float, default=100.0)
    parser.add_argument("--var-min", type=float, default=50.0)
    parser.add_argument("--var-max", type=float, default=500.0)
    parser.add_argument("--combo-output", help="Side-by-side original/recon/residual path")
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    out_frames = []
    combo_frames = []
    for frame in frames:
        recon = transform_quant_frame(
            frame,
            mode=args.mode,
            qp=args.qp,
            qp_mode=args.qp_mode,
            min_qp=args.min_qp,
            max_qp=args.max_qp,
            variance_threshold=args.variance,
            max_size=args.max_size,
            min_size=args.min_size,
            var_min=args.var_min,
            var_max=args.var_max,
        )
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        recon_bgr = np.stack([recon, recon, recon], axis=2)
        out_frames.append(recon_bgr)

        if args.combo_output:
            orig_luma = np.mean(frame.astype(np.float32), axis=2)
            orig = np.stack(
                [orig_luma, orig_luma, orig_luma],
                axis=2,
            ).astype(np.uint8)
            residual = np.clip(recon.astype(np.float32) - orig_luma + 128.0, 0, 255).astype(
                np.uint8
            )
            residual_bgr = np.stack([residual, residual, residual], axis=2)
            combo_frames.append(np.concatenate([orig, recon_bgr, residual_bgr], axis=1))

    write_video_frames(args.output, out_frames, fps)
    if args.combo_output:
        write_video_frames(args.combo_output, combo_frames, fps)


if __name__ == "__main__":
    main()
