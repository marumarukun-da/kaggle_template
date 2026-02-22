"""Minimal baseline training script."""

import argparse

import config
import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Baseline training")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")
    df.write_csv(config.OUTPUT_DIR / "submission.csv")
    print(f"Submission saved to: {config.OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
