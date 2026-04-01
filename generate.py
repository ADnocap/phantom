#!/usr/bin/env python
"""CLI entry point for synthetic dataset generation."""

import argparse
import time

from src.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SDE trajectory shards for Phantom."
    )
    parser.add_argument(
        "--n_shards", type=int, default=5,
        help="Number of shard files to create (default: 5)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10_000_000,
        help="Samples per shard (default: 10,000,000)",
    )
    parser.add_argument(
        "--context_len", type=int, default=60,
        help="Context window length in days (default: 60)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Parallel workers (default: cpu_count, max 8)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=5000,
        help="Samples per worker task (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master seed for reproducibility",
    )
    args = parser.parse_args()

    t0 = time.time()
    generate_dataset(
        n_shards=args.n_shards,
        n_samples_per_shard=args.n_samples,
        output_dir=args.output_dir,
        context_len=args.context_len,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
