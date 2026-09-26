

"""
Blocking-only pipeline.

Uses already cleaned Parquet files produced by data_cleaning.py.

Input:
    student_resource/dataset/processed/
        train_source1_clean.parquet
        train_source2_clean.parquet
        train_source3_clean.parquet

Output:
    output/candidate_pairs_train.tsv
"""

import os
import time
import pandas as pd

from blocking_keys import MinHasher, add_blocking_keys, band_columns
from candidate_generation import generate_candidates, pairs_to_id_list_rows


# ============================================================
# CONFIGURATION
# ============================================================

NUM_BANDS = 8
ROWS_PER_BAND = 3
MAX_BLOCK_SIZE = 500
RANDOM_SEED = 42


# ============================================================
# FIND PROJECT ROOT
# ============================================================

# This script is inside:
#
# ML Challenge/
#     data_preparation/
#         run_blocking_clean_only.py
#
# Therefore:
# parent of data_preparation = project root

CURRENT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_FOLDER,
        ".."
    )
)


# ============================================================
# DIRECTORIES
# ============================================================

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "student_resource",
    "dataset",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

SOURCE1_PATH = os.path.join(
    PROCESSED_DIR,
    "train_source1_clean.parquet"
)

SOURCE2_PATH = os.path.join(
    PROCESSED_DIR,
    "train_source2_clean.parquet"
)

SOURCE3_PATH = os.path.join(
    PROCESSED_DIR,
    "train_source3_clean.parquet"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_train.tsv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "entity_id",
    "name_core",
    "address_clean",
    "address_tokens",
    "name_sorted",
    "country_clean"
}


# ============================================================
# LOAD CLEAN PARQUET
# ============================================================

def load_clean_parquet(path):

    print(f"\nLoading:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    required_columns = [
        "entity_id",
        "name_core",
        "address_clean",
        "address_tokens",
        "name_sorted",
        "country_clean",
    ]

    df = pd.read_parquet(path, columns=required_columns)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:

        raise ValueError(
            f"\n{path}\n"
            f"Missing expected cleaned columns: {missing}\n"
            f"Make sure this is the output from data_cleaning.py."
        )

    print(
        f"Loaded successfully: {df.shape}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BLOCKING-ONLY ENTITY MATCHING PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nProcessed data directory:")
    print(PROCESSED_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING CLEANED DATA")
    print("=" * 70)

    start_time = time.time()

    s1c = load_clean_parquet(
        SOURCE1_PATH
    )

    s2c = load_clean_parquet(
        SOURCE2_PATH
    )

    s3c = load_clean_parquet(
        SOURCE3_PATH
    )

    all_s1_ids = s1c[
        "entity_id"
    ].tolist()

    print(
        f"\nS1 = {len(s1c)}"
    )

    print(
        f"S2 = {len(s2c)}"
    )

    print(
        f"S3 = {len(s3c)}"
    )

    print(
        f"Loading took "
        f"{time.time() - start_time:.1f} seconds"
    )


    # --------------------------------------------------------
    # COMBINE S2 + S3
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING POOL")
    print("=" * 70)

    pool = pd.concat(
        [
            s2c,
            s3c
        ],
        ignore_index=True
    )

    # We don't need separate S2/S3 dataframes anymore
    del s2c
    del s3c

    print(
        f"Pool size: {len(pool)}"
    )


    # --------------------------------------------------------
    # ADD BLOCKING FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADDING BLOCKING-KEY FEATURES")
    print("=" * 70)

    start_time = time.time()

    minhasher = MinHasher(
        num_hashes=
            NUM_BANDS * ROWS_PER_BAND,
        seed=RANDOM_SEED
    )

    print("\nProcessing Source 1...")

    s1_feat = add_blocking_keys(
        s1c,
        minhasher,
        NUM_BANDS,
        ROWS_PER_BAND
    )

    print("Processing S2 + S3 pool...")

    pool_feat = add_blocking_keys(
        pool,
        minhasher,
        NUM_BANDS,
        ROWS_PER_BAND
    )

    print(
        f"\nBlocking features added in "
        f"{time.time() - start_time:.1f} seconds"
    )


    # --------------------------------------------------------
    # BLOCKING COLUMNS
    # --------------------------------------------------------

    key_columns = [
        "name_sorted",
        "postal_code",
        "phonetic_code"
    ]

    key_columns += band_columns(
        NUM_BANDS
    )

    print("\nBlocking columns:")

    for column in key_columns:
        print(
            f"  - {column}"
        )


    # --------------------------------------------------------
    # GENERATE CANDIDATES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING CANDIDATE PAIRS")
    print("=" * 70)

    start_time = time.time()

    pairs_df = generate_candidates(
        s1_feat,
        pool_feat,
        key_columns,
        max_block_size=MAX_BLOCK_SIZE
    )

    print(
        f"\nCandidate generation took "
        f"{time.time() - start_time:.1f} seconds"
    )

    print(
        f"Total candidate pairs: "
        f"{len(pairs_df)}"
    )


    # --------------------------------------------------------
    # CONVERT TO ID LIST FORMAT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPARING OUTPUT")
    print("=" * 70)

    out_df = pairs_to_id_list_rows(
        pairs_df,
        all_s1_ids
    )

    print(
        f"Output rows: {len(out_df)}"
    )


    # --------------------------------------------------------
    # SAVE TSV
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING OUTPUT")
    print("=" * 70)

    out_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print(
        "\nSuccessfully wrote:"
    )

    print(
        OUTPUT_PATH
    )


    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    avg_candidates = (
        len(pairs_df)
        / max(len(all_s1_ids), 1)
    )

    print(
        f"\nAverage candidates per S1 entity: "
        f"{avg_candidates:.1f}"
    )

    print("\n" + "=" * 70)
    print("BLOCKING COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()