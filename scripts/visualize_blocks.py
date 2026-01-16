import argparse

from av1sim.blocks import draw_block_overlay, leaf_blocks, partition_frame
from av1sim.io import read_video_frames, write_video_frames


def main():
    parser = argparse.ArgumentParser(description="Visualize block partitions")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--max-size", type=int, default=64)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--variance", type=float, default=100.0)
    parser.add_argument("--thickness", type=int, default=1)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    visualized = []
    for frame in frames:
        blocks = partition_frame(
            frame,
            max_size=args.max_size,
            min_size=args.min_size,
            variance_threshold=args.variance,
        )
        leaves = leaf_blocks(blocks)
        visualized.append(draw_block_overlay(frame, leaves, thickness=args.thickness))

    write_video_frames(args.output, visualized, fps)


if __name__ == "__main__":
    main()
