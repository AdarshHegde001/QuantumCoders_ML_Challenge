"""
Blocking-only entity matching pipeline.

Memory-safe version.

Pipeline:

    cleaned Parquet
        |
        v
    batch feature generation
        |
        v
    temporary blocking-feature Parquet
        |
        v
    DuckDB blocking joins
        |
        v
    persistent candidate-pair table
        |
        v
    final TSV

Designed for a ~2.3 GB total input dataset on a 24 GB RAM machine.

The main goal is to avoid materializing the entire candidate-pair
dataset inside pandas.
"""

import os
import time
import shutil

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking_keys import (
    MinHasher,
    add_blocking_keys,
    band_columns,
)

from candidate_generation import (
    configure_duckdb,
    inspect_blocks,
    generate_candidates,
    write_candidate_output,
    candidate_statistics,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_BANDS = 8
ROWS_PER_BAND = 3

MAX_BLOCK_SIZE = 500

RANDOM_SEED = 42

# Number of rows processed by pandas at one time while generating
# blocking features.
FEATURE_BATCH_SIZE = 50_000

# DuckDB is deliberately limited below the machine's 24 GB RAM.
DUCKDB_MEMORY_LIMIT = "14GB"

# Leave CPU and RAM for macOS/Python.
DUCKDB_THREADS = 6

# Disk used by DuckDB for temporary spill files.
DUCKDB_TEMP_LIMIT = "20GB"


# ============================================================
# FIND PROJECT ROOT
# ============================================================

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

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "_blocking_tmp"
)

DUCKDB_TEMP_DIR = os.path.join(
    TEMP_DIR,
    "duckdb_tmp"
)

FEATURE_DIR = os.path.join(
    TEMP_DIR,
    "features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)

os.makedirs(
    DUCKDB_TEMP_DIR,
    exist_ok=True
)

os.makedirs(
    FEATURE_DIR,
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
# TEMP FEATURE FILES
# ============================================================

S1_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "s1_blocking_features.parquet"
)

S2_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "s2_blocking_features.parquet"
)

S3_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "s3_blocking_features.parquet"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_train.tsv"
)

DUCKDB_PATH = os.path.join(
    TEMP_DIR,
    "blocking.duckdb"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "entity_id",
    "name_core",
    "address_clean",
    "address_tokens",
    "name_sorted",
    "country_clean",
]


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(path):

    print(
        f"\nChecking input:"
    )

    print(
        path
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    parquet_file = pq.ParquetFile(
        path
    )

    available = set(
        parquet_file.schema_arrow.names
    )

    missing = set(
        REQUIRED_COLUMNS
    ) - available

    if missing:

        raise ValueError(
            f"\n{path}\n"
            f"Missing expected columns: {missing}"
        )

    print(
        "Input OK"
    )

    print(
        f"Rows: {parquet_file.metadata.num_rows:,}"
    )

    return parquet_file.metadata.num_rows


# ============================================================
# GENERATE BLOCKING FEATURES IN BATCHES
# ============================================================

def create_feature_parquet(
    input_path,
    output_path,
    minhasher,
):
    """
    Read the cleaned Parquet in batches, generate blocking features
    using the existing blocking_keys.py implementation, and write the
    result back to Parquet.

    At no point is the complete source loaded into pandas.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING BLOCKING FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    if os.path.exists(output_path):

        print(
            "Removing previous feature file..."
        )

        os.remove(
            output_path
        )

    parquet_file = pq.ParquetFile(
        input_path
    )

    writer = None

    total_rows = 0

    start_time = time.time()

    try:

        batch_number = 0

        for record_batch in parquet_file.iter_batches(
            batch_size=FEATURE_BATCH_SIZE,
            columns=REQUIRED_COLUMNS
        ):

            batch_number += 1

            print(
                f"\nBatch {batch_number}"
            )

            print(
                f"Rows: {record_batch.num_rows:,}"
            )

            df = record_batch.to_pandas()

            # ------------------------------------------------
            # EXISTING BLOCKING FEATURE LOGIC
            # ------------------------------------------------

            df = add_blocking_keys(
                df,
                minhasher,
                NUM_BANDS,
                ROWS_PER_BAND
            )

            # ------------------------------------------------
            # WRITE BATCH
            # ------------------------------------------------

            table = pa.Table.from_pandas(
                df,
                preserve_index=False
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd"
                )

            writer.write_table(
                table
            )

            total_rows += len(df)

            del table
            del df

            print(
                f"Processed total: {total_rows:,}"
            )

    finally:

        if writer is not None:

            writer.close()

    elapsed = time.time() - start_time

    print(
        f"\nFeature generation complete:"
    )

    print(
        f"Rows: {total_rows:,}"
    )

    print(
        f"Time: {elapsed:.1f} seconds"
    )

    return total_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "MEMORY-SAFE ENTITY RESOLUTION BLOCKING PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nInput directory:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        "\nTemporary directory:"
    )

    print(
        TEMP_DIR
    )

    print(
        "\nDuckDB memory limit:"
    )

    print(
        DUCKDB_MEMORY_LIMIT
    )


    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATING INPUT DATA"
    )

    print(
        "=" * 70
    )

    s1_rows = validate_input(
        SOURCE1_PATH
    )

    s2_rows = validate_input(
        SOURCE2_PATH
    )

    s3_rows = validate_input(
        SOURCE3_PATH
    )

    print(
        "\nTotal rows:"
    )

    print(
        f"S1: {s1_rows:,}"
    )

    print(
        f"S2: {s2_rows:,}"
    )

    print(
        f"S3: {s3_rows:,}"
    )

    print(
        f"TOTAL: {s1_rows + s2_rows + s3_rows:,}"
    )


    # ========================================================
    # MINHASH
    # ========================================================

    minhasher = MinHasher(
        num_hashes=(
            NUM_BANDS *
            ROWS_PER_BAND
        ),
        seed=RANDOM_SEED
    )


    # ========================================================
    # CREATE FEATURE FILES
    # ========================================================

    start_time = time.time()

    create_feature_parquet(
        SOURCE1_PATH,
        S1_FEATURE_PATH,
        minhasher
    )

    create_feature_parquet(
        SOURCE2_PATH,
        S2_FEATURE_PATH,
        minhasher
    )

    create_feature_parquet(
        SOURCE3_PATH,
        S3_FEATURE_PATH,
        minhasher
    )

    print(
        f"\nAll blocking features generated in "
        f"{time.time() - start_time:.1f} seconds"
    )


    # ========================================================
    # BLOCKING COLUMNS
    # ========================================================

    key_columns = [
        "name_sorted",
        "postal_code",
        "phonetic_code"
    ]

    key_columns += band_columns(
        NUM_BANDS
    )

    print(
        "\nBlocking columns:"
    )

    for column in key_columns:

        print(
            f"  - {column}"
        )


    # ========================================================
    # OPEN DUCKDB
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING DUCKDB"
    )

    print(
        "=" * 70
    )

    con = duckdb.connect(
        DUCKDB_PATH
    )

    configure_duckdb(
        con,
        memory_limit=DUCKDB_MEMORY_LIMIT,
        threads=DUCKDB_THREADS,
        temp_directory=DUCKDB_TEMP_DIR,
        max_temp_directory_size=DUCKDB_TEMP_LIMIT
    )


    # ========================================================
    # INSPECT BLOCKS FIRST
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INSPECTING BLOCK SIZES"
    )

    print(
        "=" * 70
    )

    block_stats = inspect_blocks(
        con,
        S1_FEATURE_PATH,
        [
            S2_FEATURE_PATH,
            S3_FEATURE_PATH
        ],
        key_columns,
        max_block_size=MAX_BLOCK_SIZE
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "BLOCKING SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        block_stats.to_string(
            index=False
        )
    )


    # ========================================================
    # GENERATE CANDIDATES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING CANDIDATE PAIRS"
    )

    print(
        "=" * 70
    )

    start_time = time.time()

    generate_candidates(
        con,
        S1_FEATURE_PATH,
        [
            S2_FEATURE_PATH,
            S3_FEATURE_PATH
        ],
        key_columns,
        max_block_size=MAX_BLOCK_SIZE
    )

    print(
        f"\nCandidate generation took "
        f"{time.time() - start_time:.1f} seconds"
    )


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    stats = candidate_statistics(
        con,
        S1_FEATURE_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL CANDIDATE STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"S1 entities:"
        f" {stats['s1_entities']:,}"
    )

    print(
        f"Candidate pairs:"
        f" {stats['total_candidate_pairs']:,}"
    )

    print(
        f"Average candidates per S1:"
        f" {stats['average_candidates_per_s1']:.1f}"
    )


    # ========================================================
    # WRITE FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "WRITING FINAL OUTPUT"
    )

    print(
        "=" * 70
    )

    start_time = time.time()

    write_candidate_output(
        con,
        S1_FEATURE_PATH,
        OUTPUT_PATH
    )

    print(
        f"\nOutput written in "
        f"{time.time() - start_time:.1f} seconds"
    )

    print(
        "\nSuccessfully wrote:"
    )

    print(
        OUTPUT_PATH
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    con.close()

    print(
        "\n" + "=" * 70
    )

    print(
        "BLOCKING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nTemporary DuckDB/feature files are located at:"
    )

    print(
        TEMP_DIR
    )

    print(
        "\nYou can delete this directory after verifying the output."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()