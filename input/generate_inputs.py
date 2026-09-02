#!/usr/bin/env python3
"""Generate sorted, unique key arrays for 5 patterns across 3 sizes.
Output: {pattern}_{size}.txt files (one key per line).
"""

import argparse
import random
from pathlib import Path

PATTERNS = ["sequential", "uniform", "clustered", "exponential", "adversarial"]
DEFAULT_SIZES = [1_000, 10_000, 100_000]
DEFAULT_FIT_SAMPLE_SIZE = 200

# Fraction of outlier "poison" points in adversarial pattern.
ADVERSARIAL_POISON_FRACTION = 0.001


def sequential(n: int, seed: int) -> list[int]:
    return list(range(n))


def uniform(n: int, seed: int) -> list[int]:
    if n == 0:
        return []
    rng = random.Random(seed)
    cap = min(10**11, max(n * 1000, 10**6))
    return sorted(rng.sample(range(cap), n))


def clustered(n: int, seed: int) -> list[int]:
    # Multiple tight clusters separated by gaps.
    if n == 0:
        return []
    rng = random.Random(seed)
    num_clusters = min(5, n)
    sizes = [n // num_clusters] * num_clusters
    for i in range(n % num_clusters):
        sizes[i] += 1

    cluster_gap = 10**7  # Ensures clusters don't overlap.
    values: list[int] = []
    for c, size in enumerate(sizes):
        if size == 0:
            continue
        width = max(size * 3, 10)
        offset = c * cluster_gap
        values.extend(offset + v for v in rng.sample(range(width), size))
    return sorted(values)


def exponential(n: int, seed: int) -> list[int]:
    # Geometrically growing gaps (1000x span).
    if n == 0:
        return []
    growth = 1000 ** (1.0 / max(n - 1, 1))
    gap = 1.0
    key = 0
    values = []
    for _ in range(n):
        values.append(key)
        gap *= growth
        key += max(1, round(gap))
    return values


def adversarial(n: int, seed: int) -> list[int]:
    # Bulk keys plus far-away outlier points.
    if n == 0:
        return []
    k = min(n, max(1, round(n * ADVERSARIAL_POISON_FRACTION)))
    bulk_n = n - k
    rng = random.Random(seed)

    bulk_range = max(bulk_n * 100, 10**6)
    bulk = sorted(rng.sample(range(bulk_range), bulk_n)) if bulk_n > 0 else []

    poison_lo = 50 * bulk_range
    poison = sorted(rng.sample(range(poison_lo, poison_lo + 1000 * k), k))
    return bulk + poison


GENERATORS = {
    "sequential": sequential,
    "uniform": uniform,
    "clustered": clustered,
    "exponential": exponential,
    "adversarial": adversarial,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    p.add_argument("--seed", type=int, default=42, help="Seeds every pattern's random component")
    p.add_argument("--fit-sample-size", type=int, default=DEFAULT_FIT_SAMPLE_SIZE,
                    help="Written to input/fit_sample_size.txt -- what FitSampled (v2) draws")
    args = p.parse_args()

    out_dir = Path(__file__).parent
    for pattern in args.patterns:
        for n in args.sizes:
            values = GENERATORS[pattern](n, args.seed)
            assert values == sorted(set(values)), f"{pattern} n={n}: keys must be sorted and unique"
            path = out_dir / f"{pattern}_{n}.txt"
            path.write_text("\n".join(map(str, values)) + "\n")
            print(f"wrote {path} ({len(values)} keys)")

    sample_size_path = out_dir / "fit_sample_size.txt"
    sample_size_path.write_text(f"{args.fit_sample_size}\n")
    print(f"wrote {sample_size_path} (fit_sample_size={args.fit_sample_size})")


if __name__ == "__main__":
    main()
