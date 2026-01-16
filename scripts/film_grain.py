import argparse

import numpy as np

from av1sim.grain import (
    analyze_grain,
    apply_grain_to_base,
    save_params,
    split_base_grain,
)
from av1sim.io import read_video_frames, write_video_frames


def main():
    parser = argparse.ArgumentParser(description="Film grain analysis and synthesis")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--base-output", help="Denoised base video path")
    parser.add_argument("--synth-output", help="Synthesized grain video path")
    parser.add_argument("--combo-output", help="Side-by-side original/base/synth path")
    parser.add_argument("--params-output", help="JSON grain parameters path")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--frames", type=int, default=0, help="Limit frames for analysis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    analyze_frames = frames if args.frames <= 0 else frames[: args.frames]
    params = analyze_grain(analyze_frames, bins=args.bins, blur_ksize=args.blur)

    if args.params_output:
        save_params(args.params_output, params)

    rng = np.random.default_rng(args.seed)
    base_frames = []
    synth_frames = []
    combo_frames = []
    for frame in frames:
        base, _grain = split_base_grain(frame, blur_ksize=args.blur)
        synth = apply_grain_to_base(base, params, rng=rng)
        base_vis = np.clip(base, 0, 255).astype(np.uint8)
        synth_vis = np.clip(synth, 0, 255).astype(np.uint8)
        base_bgr = np.stack([base_vis, base_vis, base_vis], axis=2)
        synth_bgr = np.stack([synth_vis, synth_vis, synth_vis], axis=2)
        base_frames.append(base_bgr)
        synth_frames.append(synth_bgr)
        if args.combo_output:
            orig_luma = np.mean(frame.astype(np.float32), axis=2)
            orig = np.stack([orig_luma, orig_luma, orig_luma], axis=2).astype(np.uint8)
            combo_frames.append(np.concatenate([orig, base_bgr, synth_bgr], axis=1))

    if args.base_output:
        write_video_frames(args.base_output, base_frames, fps)
    if args.synth_output:
        write_video_frames(args.synth_output, synth_frames, fps)
    if args.combo_output:
        write_video_frames(args.combo_output, combo_frames, fps)


if __name__ == "__main__":
    main()
