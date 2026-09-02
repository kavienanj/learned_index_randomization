#!/usr/bin/env python3
"""Generate real-world test data from system dictionary words using polynomial hashing.
Output: real_wordhash_{size}.txt files (one hash per line).
"""

import argparse
import random
from pathlib import Path

DEFAULT_WORDLIST_CANDIDATES = [
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
]

# Modulus keeps hashed keys under 2^53 (double's exact-integer limit).
HASH_MODULUS = 10**11


def find_default_wordlist() -> Path | None:
    for candidate in DEFAULT_WORDLIST_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def poly_hash(word: str, modulus: int = HASH_MODULUS) -> int:
    h = 0
    for c in word:
        h = (h * 131 + ord(c)) % modulus
    return h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wordlist", type=Path, default=None,
                    help="Newline-delimited real text file (default: system dictionary)")
    p.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    p.add_argument("--seed", type=int, default=42, help="Seeds which words get sampled")
    args = p.parse_args()

    wordlist = args.wordlist or find_default_wordlist()
    if wordlist is None:
        raise SystemExit(
            "No real wordlist found at the usual system paths and none given via "
            "--wordlist. Point --wordlist at any real (non-synthetic) text file, "
            "one token per line."
        )

    words = [w.strip() for w in wordlist.read_text(errors="ignore").splitlines() if w.strip()]
    print(f"loaded {len(words)} real words from {wordlist}")

    out_dir = Path(__file__).parent
    rng = random.Random(args.seed)
    shuffled = words[:]
    rng.shuffle(shuffled)

    for n in args.sizes:
        # Collect unique hash values from shuffled words.
        unique: dict[int, str] = {}
        idx = 0
        while len(unique) < n and idx < len(shuffled):
            w = shuffled[idx]
            unique[poly_hash(w)] = w
            idx += 1
        if len(unique) < n:
            print(f"skipping size {n}: only {len(unique)} unique keys available from {len(words)} words")
            continue

        values = sorted(unique.keys())
        assert len(values) == len(set(values)) == n
        path = out_dir / f"real_wordhash_{n}.txt"
        path.write_text("\n".join(map(str, values)) + "\n")
        print(f"wrote {path} ({n} keys, source: {wordlist.name})")


if __name__ == "__main__":
    main()
