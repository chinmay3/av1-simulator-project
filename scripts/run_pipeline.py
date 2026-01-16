import argparse

from av1sim.pipeline import run_baseline_pipeline


def main():
    parser = argparse.ArgumentParser(description="Baseline AV1 simulator pipeline")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    args = parser.parse_args()

    run_baseline_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
