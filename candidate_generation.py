"""
Candidate generation (blocking) using DuckDB.

This replaces the original pandas-based candidate generation.

Why DuckDB:
    - Candidate pairs are never materialized as one giant pandas DataFrame.
    - Each blocking key is processed independently.
    - Duplicate pairs are removed incrementally using a composite primary key.
    - DuckDB can spill joins/grouping/sorting to disk when memory is insufficient.
    - Final aggregation is performed inside DuckDB instead of pandas.

Input:
    S1 blocking-feature Parquet
    S2 blocking-feature Parquet
    S3 blocking-feature Parquet

Output:
    DataFrame containing:
        source1_entity_id
        candidate_entity_id

The returned DataFrame is only intended for statistics/debugging.
The final submission TSV should be produced directly by DuckDB.
"""

import os
import duckdb
import pandas as pd


# ============================================================
# DUCKDB SETUP
# ============================================================

def configure_duckdb(
    con,
    memory_limit="14GB",
    threads=6,
    temp_directory=None,
    max_temp_directory_size="20GB",
):
    """
    Configure DuckDB for a 24 GB RAM machine.

    We intentionally do NOT allow DuckDB to consume all RAM.
    The remaining memory is left for macOS, Python and filesystem
    caching.

    DuckDB supports out-of-core execution for joins, grouping and
    sorting by spilling intermediate data to disk.
    """

    con.execute(
        f"SET memory_limit = '{memory_limit}'"
    )

    con.execute(
        f"SET threads = {int(threads)}"
    )

    con.execute(
        "SET preserve_insertion_order = false"
    )

    if temp_directory is not None:

        os.makedirs(
            temp_directory,
            exist_ok=True
        )

        escaped = temp_directory.replace(
            "'",
            "''"
        )

        con.execute(
            f"SET temp_directory = '{escaped}'"
        )

        con.execute(
            f"SET max_temp_directory_size = '{max_temp_directory_size}'"
        )


# ============================================================
# HELPERS
# ============================================================

def _sql_path(path):
    """
    Safely quote a filesystem path for DuckDB SQL.
    """

    return "'" + str(path).replace(
        "'",
        "''"
    ) + "'"


def _parquet_relation(paths):
    """
    Return a DuckDB read_parquet expression for one or more files.
    """

    if isinstance(paths, (list, tuple)):

        values = ", ".join(
            _sql_path(path)
            for path in paths
        )

        return f"read_parquet([{values}], union_by_name=true)"

    return f"read_parquet({_sql_path(paths)})"


# ============================================================
# BLOCK STATISTICS
# ============================================================

def inspect_blocks(
    con,
    s1_path,
    pool_paths,
    key_columns,
    max_block_size=500,
):
    """
    Inspect block sizes BEFORE candidate generation.

    This is important because the raw dataset is only ~2.3 GB,
    but candidate generation can be orders of magnitude larger.

    Returns a pandas DataFrame containing one row per blocking key.
    """

    s1_relation = _parquet_relation(
        s1_path
    )

    pool_relation = _parquet_relation(
        pool_paths
    )

    results = []

    for key in key_columns:

        print(
            f"\nInspecting blocking key: {key}"
        )

        query = f"""
        WITH
        s1_blocks AS (
            SELECT
                "{key}" AS block_key,
                COUNT(*) AS s1_count
            FROM {s1_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
            GROUP BY 1
        ),

        pool_blocks AS (
            SELECT
                "{key}" AS block_key,
                COUNT(*) AS pool_count
            FROM {pool_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
            GROUP BY 1
        ),

        joined AS (
            SELECT
                s.block_key,
                s.s1_count,
                p.pool_count,
                s.s1_count * p.pool_count AS pair_count
            FROM s1_blocks s
            INNER JOIN pool_blocks p
                ON s.block_key = p.block_key
            WHERE p.pool_count <= {int(max_block_size)}
        )

        SELECT
            COUNT(*) AS usable_blocks,
            COALESCE(SUM(pair_count), 0) AS estimated_pairs,
            COALESCE(MAX(s1_count), 0) AS max_s1_block,
            COALESCE(MAX(pool_count), 0) AS max_pool_block,
            COALESCE(MAX(pair_count), 0) AS max_pair_block
        FROM joined
        """

        row = con.execute(
            query
        ).fetchone()

        usable_blocks = int(
            row[0] or 0
        )

        estimated_pairs = int(
            row[1] or 0
        )

        max_s1_block = int(
            row[2] or 0
        )

        max_pool_block = int(
            row[3] or 0
        )

        max_pair_block = int(
            row[4] or 0
        )

        results.append(
            {
                "blocking_key": key,
                "usable_blocks": usable_blocks,
                "estimated_pairs": estimated_pairs,
                "max_s1_block": max_s1_block,
                "max_pool_block": max_pool_block,
                "max_pair_block": max_pair_block,
            }
        )

        print(
            f"  usable blocks : {usable_blocks:,}"
        )

        print(
            f"  estimated pairs: {estimated_pairs:,}"
        )

        print(
            f"  max S1 block   : {max_s1_block:,}"
        )

        print(
            f"  max pool block : {max_pool_block:,}"
        )

        print(
            f"  max pair block : {max_pair_block:,}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    con,
    s1_path,
    pool_paths,
    key_columns,
    max_block_size=500,
    id_col="entity_id",
    s1_id_col="source1_entity_id",
    cand_id_col="candidate_entity_id",
):
    """
    Generate candidate pairs using DuckDB.

    Important:
        We process ONE blocking key at a time.

    The candidate table lives on disk in DuckDB and has a composite
    primary key, so duplicate candidate pairs are discarded as they
    are inserted rather than collecting all pairs in Python first.

    This preserves the original blocking logic:

        pool block size <= max_block_size

    We deliberately do NOT silently add an S1 block-size restriction,
    because that would change blocking recall.

    Instead, inspect_blocks() reports dangerous S1 blocks before
    candidate generation.
    """

    # --------------------------------------------------------
    # CREATE PERSISTENT CANDIDATE TABLE
    # --------------------------------------------------------

    con.execute(
        f"""
        DROP TABLE IF EXISTS candidate_pairs
        """
    )

    con.execute(
        f"""
        CREATE TABLE candidate_pairs (
            "{s1_id_col}" VARCHAR,
            "{cand_id_col}" VARCHAR,
            PRIMARY KEY (
                "{s1_id_col}",
                "{cand_id_col}"
            )
        )
        """
    )

    s1_relation = _parquet_relation(
        s1_path
    )

    pool_relation = _parquet_relation(
        pool_paths
    )

    total_inserted = 0

    # --------------------------------------------------------
    # PROCESS EACH BLOCKING KEY SEPARATELY
    # --------------------------------------------------------

    for key in key_columns:

        print(
            f"\nGenerating candidates using: {key}"
        )

        query = f"""
        WITH

        s1_blocks AS (
            SELECT
                CAST("{key}" AS VARCHAR) AS block_key,
                COUNT(*) AS block_count
            FROM {s1_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
            GROUP BY 1
        ),

        pool_blocks AS (
            SELECT
                CAST("{key}" AS VARCHAR) AS block_key,
                COUNT(*) AS block_count
            FROM {pool_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
            GROUP BY 1
            HAVING COUNT(*) <= {int(max_block_size)}
        ),

        usable_blocks AS (
            SELECT
                s.block_key
            FROM s1_blocks s
            INNER JOIN pool_blocks p
                ON s.block_key = p.block_key
        ),

        s1_filtered AS (
            SELECT
                CAST("{id_col}" AS VARCHAR) AS entity_id,
                CAST("{key}" AS VARCHAR) AS block_key
            FROM {s1_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
        ),

        pool_filtered AS (
            SELECT
                CAST("{id_col}" AS VARCHAR) AS entity_id,
                CAST("{key}" AS VARCHAR) AS block_key
            FROM {pool_relation}
            WHERE
                "{key}" IS NOT NULL
                AND CAST("{key}" AS VARCHAR) <> ''
        )

        SELECT
            s.entity_id AS "{s1_id_col}",
            p.entity_id AS "{cand_id_col}"
        FROM s1_filtered s
        INNER JOIN pool_filtered p
            ON s.block_key = p.block_key
        INNER JOIN usable_blocks u
            ON s.block_key = u.block_key
        """

        before = con.execute(
            """
            SELECT COUNT(*)
            FROM candidate_pairs
            """
        ).fetchone()[0]

        con.execute(
            f"""
            INSERT OR IGNORE INTO candidate_pairs
            (
                "{s1_id_col}",
                "{cand_id_col}"
            )
            {query}
            """
        )

        after = con.execute(
            """
            SELECT COUNT(*)
            FROM candidate_pairs
            """
        ).fetchone()[0]

        added = int(
            after - before
        )

        total_inserted += added

        print(
            f"  New unique candidates: {added:,}"
        )

        print(
            f"  Total unique candidates: {after:,}"
        )

    print(
        f"\nFinal unique candidate pairs: "
        f"{total_inserted:,}"
    )

    return con.table(
        "candidate_pairs"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

def write_candidate_output(
    con,
    s1_path,
    output_path,
    s1_id_col="source1_entity_id",
    cand_id_col="candidate_entity_id",
):
    """
    Create the exact submission-style output:

        source1_entity_id
        candidate_entity_ids

    Every S1 entity is retained, including entities with zero
    candidates.

    The aggregation happens inside DuckDB rather than pandas.
    """

    output_sql = _sql_path(
        output_path
    )

    s1_relation = _parquet_relation(
        s1_path
    )

    query = f"""
    WITH all_s1 AS (
        SELECT DISTINCT
            CAST("entity_id" AS VARCHAR) AS "{s1_id_col}"
        FROM {s1_relation}
    ),

    aggregated AS (
        SELECT
            "{s1_id_col}",
            string_agg(
                DISTINCT CAST("{cand_id_col}" AS VARCHAR),
                ',' ORDER BY CAST("{cand_id_col}" AS VARCHAR)
            ) AS candidate_entity_ids
        FROM candidate_pairs
        GROUP BY "{s1_id_col}"
    )

    SELECT
        a."{s1_id_col}",
        COALESCE(
            g.candidate_entity_ids,
            ''
        ) AS candidate_entity_ids
    FROM all_s1 a
    LEFT JOIN aggregated g
        ON a."{s1_id_col}" = g."{s1_id_col}"
    ORDER BY a."{s1_id_col}"
    """

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO {output_sql}
        (
            DELIMITER '\\t',
            HEADER true
        )
        """
    )


# ============================================================
# STATISTICS
# ============================================================

def candidate_statistics(
    con,
    s1_path,
    s1_id_col="source1_entity_id",
):
    """
    Calculate final candidate statistics without converting the
    entire candidate table to pandas.
    """

    total_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs
        """
    ).fetchone()[0]

    s1_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                CAST("entity_id" AS VARCHAR)
            FROM {_parquet_relation(s1_path)}
        )
        """
    ).fetchone()[0]

    average = (
        float(total_pairs) / max(
            int(s1_count),
            1
        )
    )

    return {
        "total_candidate_pairs": int(
            total_pairs
        ),
        "s1_entities": int(
            s1_count
        ),
        "average_candidates_per_s1": average,
    }


# ============================================================
# COMPATIBILITY HELPER
# ============================================================

def pairs_to_id_list_rows(
    pairs_df,
    all_s1_ids,
    s1_id_col="source1_entity_id",
    cand_id_col="candidate_entity_id",
    out_id_col_name="candidate_entity_ids",
):
    """
    Compatibility version of the old pandas helper.

    The main pipeline no longer needs this function because final
    aggregation is performed directly inside DuckDB.

    It is retained so existing imports do not break.
    """

    grouped = pairs_df.groupby(
        s1_id_col
    )[cand_id_col].apply(
        lambda ids: ",".join(
            sorted(
                set(
                    ids.astype(str)
                )
            )
        )
    )

    grouped = grouped.reindex(
        all_s1_ids,
        fill_value=""
    )

    out = grouped.reset_index()

    out.columns = [
        "source1_entity_id",
        out_id_col_name
    ]

    return out