"""
Candidate generation (blocking) — the actual S1 <-> S2/S3 join step.

Every blocking key is joined with `country_clean` as a mandatory co-key
(verified on training data: cross-country true matches are ~0%, see the
20%-sample check — 0 mismatches out of 1.53M true match pairs). This is
implemented cheaply by concatenating country into the key string itself
("us||07012" instead of just "07012"), so it's a single merge, not a
separate per-country loop.

Keys used (all must already exist on both dataframes — run
cleaning.clean_dataframe() then blocking_keys.add_blocking_keys() first):
  - name_sorted    (order-independent exact name match, from your cleaning)
  - postal_code    (isolated PIN/zip token)
  - phonetic_code  (soundex of first name token; Latin-script names only)
  - band_0..band_B (MinHash/LSH token-overlap buckets)

Any block (a single key VALUE, e.g. one specific postal code) whose
pool-side group exceeds `max_block_size` is dropped ("block purging") — an
overly generic key generates a huge number of low-value pairs and hurts
your reduction ratio without helping recall much.
"""
import pandas as pd


def _keyed(df, key_col):
    """Build the country-prefixed join key column, filtering out rows
    where the underlying key is empty (no signal from this key)."""
    has_value = df[key_col].astype(bool)
    combo = df["country_clean"].astype(str) + "||" + df[key_col].astype(str)
    return combo.where(has_value, other=None)


def generate_candidates(s1_df, pool_df, key_columns, max_block_size=500,
                         id_col="entity_id", s1_id_col="source1_entity_id",
                         cand_id_col="candidate_entity_id"):
    """For each key column, country-partitioned inner-join s1_df and
    pool_df on that column (dropping empty keys and purging oversized
    blocks on the pool side), then union all pairs found across every key.

    Returns a DataFrame [s1_id_col, cand_id_col], deduplicated.
    """
    all_pairs = []
    for key in key_columns:
        s1_join_key = _keyed(s1_df, key)
        pool_join_key = _keyed(pool_df, key)

        s1_slice = pd.DataFrame({
            "entity_id": s1_df[id_col].values,
            "_key": s1_join_key.values,
        }).dropna(subset=["_key"])
        pool_slice = pd.DataFrame({
            "entity_id": pool_df[id_col].values,
            "_key": pool_join_key.values,
        }).dropna(subset=["_key"])

        if s1_slice.empty or pool_slice.empty:
            continue

        # Block purging.
        group_sizes = pool_slice.groupby("_key")["entity_id"].transform("size")
        pool_slice = pool_slice.loc[group_sizes <= max_block_size]
        if pool_slice.empty:
            continue

        merged = s1_slice.merge(pool_slice, on="_key", suffixes=("_s1", "_pool"))
        pairs = merged[["entity_id_s1", "entity_id_pool"]].rename(
            columns={"entity_id_s1": s1_id_col, "entity_id_pool": cand_id_col}
        )
        all_pairs.append(pairs)

    if not all_pairs:
        return pd.DataFrame(columns=[s1_id_col, cand_id_col])

    combined = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    return combined


def pairs_to_id_list_rows(pairs_df, all_s1_ids, s1_id_col="source1_entity_id",
                           cand_id_col="candidate_entity_id",
                           out_id_col_name="candidate_entity_ids"):
    """Aggregate (s1, candidate) pairs into the submission row format: one
    row per S1 entity, comma-joined candidate IDs (empty string if none).
    `all_s1_ids` must include every S1 entity so every one gets a row, even
    with zero candidates."""
    grouped = pairs_df.groupby(s1_id_col)[cand_id_col].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    grouped = grouped.reindex(all_s1_ids, fill_value="")
    out = grouped.reset_index()
    out.columns = ["source1_entity_id", out_id_col_name]
    return out
