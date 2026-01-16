import argparse

from av1sim.codec import decode_video


def main():
    parser = argparse.ArgumentParser(description="AV1SIM decoder (part 9)")
    parser.add_argument("input", help="Input bitstream path (.av1s)")
    parser.add_argument("output", help="Output video path (.mp4)")
    args = parser.parse_args()

    decode_video(args.input, args.output)


if __name__ == "__main__":
    main()
