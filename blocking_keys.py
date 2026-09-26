"""
Blocking-key feature engineering.

Takes the output of cleaning.clean_dataframe() (columns: name_clean,
name_core, name_sorted, address_clean, address_tokens, address_numbers,
country_clean, ...) and adds the extra columns blocking actually joins on:

  postal_code    - isolated 4-6 digit postal/PIN token (kept separate from
                   address_numbers, which mixes in street/plot numbers)
  phonetic_code  - soundex of the first token of name_core (Latin-script
                   names only; empty for Devanagari etc. — see note below)
  band_0..band_B - MinHash/LSH bucket keys over the token set of
                   name_core + address_tokens (order-invariant, tolerates
                   partial token overlap — catches what name_sorted's exact
                   match misses)

name_sorted and country_clean (already in your cleaned df) are used as
blocking keys directly by candidate_generation.py — nothing to add for them
here.
"""
import re
import zlib

import numpy as np
import pandas as pd

_POSTAL_RE = re.compile(r"(?<!/)\b\d{4,6}\b(?!/)")


def extract_postal_code(address_clean):
    """Last standalone (non slash-joined) 4-6 digit token in the cleaned
    address. The negative lookaround excludes plot-number groups like
    '9/1/3' so those don't get mistaken for a postal code."""
    if not address_clean:
        return ""
    matches = _POSTAL_RE.findall(str(address_clean))
    return matches[-1] if matches else ""


def soundex(word):
    """Pure-Python Soundex (no external deps). Only meaningful for
    Latin-script tokens — Devanagari/other-script names will hash their
    individual unicode code points, which is not a real phonetic code, so
    treat empty/garbage soundex output as "no signal" for those records
    rather than a false negative on this key. name_sorted and the MinHash
    bands below are the keys that actually cover non-Latin names."""
    word = "".join(c for c in str(word).upper() if c.isascii() and c.isalpha())
    if not word:
        return ""
    codes = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first = word[0]
    encoded = first
    prev = codes.get(first, "")
    for c in word[1:]:
        code = codes.get(c, "")
        if code and code != prev:
            encoded += code
        if c not in ("H", "W"):
            prev = code
    return (encoded + "000")[:4]


class MinHasher:
    """Universal-hashing MinHash: one base hash per token (zlib.crc32,
    C-speed), then `num_hashes` permuted variants via (a*h + b) mod prime,
    vectorized with numpy. Swap for datasketch.MinHash locally if you have
    internet access and want a faster/streaming implementation — the
    banding function below only needs a fixed-length uint array back."""

    def __init__(self, num_hashes=24, seed=42):
        self.num_hashes = num_hashes
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**31 - 1, size=num_hashes).astype(np.uint64)
        self.b = rng.randint(0, 2**31 - 1, size=num_hashes).astype(np.uint64)
        self.prime = np.uint64((1 << 61) - 1)

    def signature(self, tokens):
        if not tokens:
            return np.zeros(self.num_hashes, dtype=np.uint64)
        token_hashes = np.array(
            [zlib.crc32(t.encode("utf-8")) for t in tokens], dtype=np.uint64
        )
        vals = (np.outer(token_hashes, self.a) + self.b) % self.prime
        return vals.min(axis=0)


def lsh_band_keys(signature, num_bands, rows_per_band):
    """Split a MinHash signature into `num_bands` bands, one string
    bucket-key per band. Two records sharing ANY band's exact key are
    treated as candidates — approximates "Jaccard similarity above some
    threshold" without full pairwise comparison."""
    keys = []
    for band in range(num_bands):
        start = band * rows_per_band
        chunk = signature[start:start + rows_per_band]
        keys.append(str(hash(chunk.tobytes())))
    return keys


def add_blocking_keys(df, minhasher, num_bands, rows_per_band):
    """Add postal_code, phonetic_code, and band_0..band_{num_bands-1}
    columns to df (mutates and returns it — no defensive copy, since a full
    copy roughly doubles memory at multi-million-row scale; copy it
    yourself first if you need the original df untouched).

    Requires columns name_core, address_clean, address_tokens to already
    exist (i.e. run cleaning.clean_dataframe() first). Deliberately written
    as one row-by-row pass with only transient per-row objects (no big
    intermediate Series of Python sets held in memory at once) — the
    vectorized-looking version of this is much more memory-hungry at
    millions-of-rows scale and was what caused an OOM kill in testing.
    """
    postal_codes = []
    phonetic_codes = []
    band_cols = [[] for _ in range(num_bands)]

    for name_core, address_clean, address_tokens in zip(
        df["name_core"], df["address_clean"], df["address_tokens"]
    ):
        postal_codes.append(extract_postal_code(address_clean))

        name_toks = name_core.split() if name_core else []
        phonetic_codes.append(soundex(name_toks[0]) if name_toks else "")

        addr_toks = address_tokens.split() if address_tokens else []
        tokens = set(name_toks)
        tokens.update(addr_toks)

        sig = minhasher.signature(list(tokens))
        keys = lsh_band_keys(sig, num_bands, rows_per_band)
        for i, k in enumerate(keys):
            band_cols[i].append(k)

    df["postal_code"] = postal_codes
    df["phonetic_code"] = phonetic_codes
    for i in range(num_bands):
        df[f"band_{i}"] = band_cols[i]

    return df


def band_columns(num_bands):
    return [f"band_{i}" for i in range(num_bands)]
