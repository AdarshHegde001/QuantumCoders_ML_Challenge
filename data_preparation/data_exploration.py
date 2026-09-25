import pandas as pd
import re
import random

# for source in ["source1", "source2", "source3"]:

#     path = f"../student_resource/dataset/train/train_{source}.tsv"

#     df = pd.read_csv(path, sep="\t")

#     print("\n" + "=" * 50)
#     print(source.upper())
#     print("=" * 50)

#     print("Shape:", df.shape)

#     print("\nColumns:")
#     print(df.columns.tolist())

#     print("\nMissing values:")
#     print(df.isnull().sum())

#     print("\nSample:")
#     print(df.head(3))

#     print("\nCountry distribution:")
#     print(df["country"].value_counts(dropna=False))




# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "../student_resource/dataset/train"

S1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
S2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
S3_PATH = f"{TRAIN_DIR}/train_source3.tsv"
GT_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

RANDOM_SEED = 42
SAMPLE_SIZE = 50


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(S1_PATH, sep="\t")
s2 = pd.read_csv(S2_PATH, sep="\t")
s3 = pd.read_csv(S3_PATH, sep="\t")

gt = pd.read_csv(GT_PATH, sep="\t")

print("Loaded.")
print("S1:", s1.shape)
print("S2:", s2.shape)
print("S3:", s3.shape)
print("Ground truth:", gt.shape)


# ============================================================
# BUILD FAST LOOKUPS
# ============================================================

# We don't want to repeatedly search the DataFrames.
# entity_id -> record

s1_lookup = s1.set_index("entity_id")
s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")


# ============================================================
# BASIC GROUND TRUTH STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH STATISTICS")
print("=" * 70)

# Empty matched_entity_ids = singleton
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("").astype(str)

gt["is_singleton"] = (
    gt["matched_entity_ids"].str.strip() == ""
)

print(
    "Total Source 1 entities:",
    len(gt)
)

print(
    "Entities with at least one match:",
    (~gt["is_singleton"]).sum()
)

print(
    "Singleton entities:",
    gt["is_singleton"].sum()
)


# ============================================================
# NUMBER OF MATCHES PER S1
# ============================================================

def count_matches(value):

    value = str(value).strip()

    if not value:
        return 0

    return len(
        [x for x in value.split(",") if x.strip()]
    )


gt["num_matches"] = gt["matched_entity_ids"].apply(
    count_matches
)

print("\nNumber of matches per Source 1 entity:")

print(
    gt["num_matches"].value_counts().sort_index()
)


# ============================================================
# SOURCE 2 vs SOURCE 3 MATCH DISTRIBUTION
# ============================================================

def split_matches(value):

    value = str(value).strip()

    if not value:
        return [], []

    ids = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    s2_ids = [x for x in ids if x.startswith("S2-")]
    s3_ids = [x for x in ids if x.startswith("S3-")]

    return s2_ids, s3_ids


s2_counts = []
s3_counts = []

for value in gt["matched_entity_ids"]:

    s2_ids, s3_ids = split_matches(value)

    s2_counts.append(len(s2_ids))
    s3_counts.append(len(s3_ids))


gt["num_s2_matches"] = s2_counts
gt["num_s3_matches"] = s3_counts


print("\nS2 match count distribution:")
print(gt["num_s2_matches"].value_counts().sort_index())

print("\nS3 match count distribution:")
print(gt["num_s3_matches"].value_counts().sort_index())


# ============================================================
# RANDOM POSITIVE PAIRS
# ============================================================

positive_gt = gt[
    ~gt["is_singleton"]
].copy()

positive_gt = positive_gt.sample(
    n=min(SAMPLE_SIZE, len(positive_gt)),
    random_state=RANDOM_SEED
)


# ============================================================
# PRINT ACTUAL MATCHING PAIRS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM TRUE MATCHES")
print("=" * 70)


def get_record(entity_id):

    if entity_id.startswith("S1-"):
        return s1_lookup.loc[entity_id]

    if entity_id.startswith("S2-"):
        return s2_lookup.loc[entity_id]

    if entity_id.startswith("S3-"):
        return s3_lookup.loc[entity_id]

    return None


for i, (_, row) in enumerate(
    positive_gt.iterrows(),
    start=1
):

    s1_id = row["source1_entity_id"]

    s1_record = s1_lookup.loc[s1_id]

    print("\n" + "-" * 70)
    print(f"MATCH EXAMPLE {i}")

    print("\nSOURCE 1")
    print("ID:      ", s1_id)
    print("Name:    ", s1_record["business_name"])
    print("Address: ", s1_record["business_address"])
    print("Country: ", s1_record["country"])

    s2_ids, s3_ids = split_matches(
        row["matched_entity_ids"]
    )

    # --------------------------
    # Source 2 matches
    # --------------------------

    for entity_id in s2_ids:

        record = s2_lookup.loc[entity_id]

        print("\nSOURCE 2 MATCH")
        print("ID:      ", entity_id)
        print("Name:    ", record["business_name"])
        print("Address: ", record["business_address"])
        print("Country: ", record["country"])

    # --------------------------
    # Source 3 matches
    # --------------------------

    for entity_id in s3_ids:

        record = s3_lookup.loc[entity_id]

        print("\nSOURCE 3 MATCH")
        print("ID:      ", entity_id)
        print("Name:    ", record["business_name"])
        print("Address: ", record["business_address"])
        print("Country: ", record["country"])


# ============================================================
# MISSING-FIELD ANALYSIS FOR TRUE MATCHES
# ============================================================

print("\n" + "=" * 70)
print("MISSING DATA IN TRUE MATCHES")
print("=" * 70)


missing_stats = {
    "S1_name_missing": 0,
    "S1_address_missing": 0,
    "S2_name_missing": 0,
    "S2_address_missing": 0,
    "S3_name_missing": 0,
    "S3_address_missing": 0,
}


for _, row in positive_gt.iterrows():

    s1_record = s1_lookup.loc[
        row["source1_entity_id"]
    ]

    if pd.isna(s1_record["business_name"]):
        missing_stats["S1_name_missing"] += 1

    if pd.isna(s1_record["business_address"]):
        missing_stats["S1_address_missing"] += 1

    s2_ids, s3_ids = split_matches(
        row["matched_entity_ids"]
    )

    for entity_id in s2_ids:

        record = s2_lookup.loc[entity_id]

        if pd.isna(record["business_name"]):
            missing_stats["S2_name_missing"] += 1

        if pd.isna(record["business_address"]):
            missing_stats["S2_address_missing"] += 1

    for entity_id in s3_ids:

        record = s3_lookup.loc[entity_id]

        if pd.isna(record["business_name"]):
            missing_stats["S3_name_missing"] += 1

        if pd.isna(record["business_address"]):
            missing_stats["S3_address_missing"] += 1


for key, value in missing_stats.items():

    print(f"{key}: {value}")


# ============================================================
# COUNTRY CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY CONSISTENCY IN TRUE MATCHES")
print("=" * 70)


country_mismatches = 0
country_matches = 0

for _, row in positive_gt.iterrows():

    s1_record = s1_lookup.loc[
        row["source1_entity_id"]
    ]

    s1_country = str(
        s1_record["country"]
    ).strip().lower()

    s2_ids, s3_ids = split_matches(
        row["matched_entity_ids"]
    )

    matched_ids = s2_ids + s3_ids

    for entity_id in matched_ids:

        record = get_record(entity_id)

        other_country = str(
            record["country"]
        ).strip().lower()

        if s1_country == other_country:
            country_matches += 1
        else:
            country_mismatches += 1

print("Country matches:", country_matches)
print("Country mismatches:", country_mismatches)


# ============================================================
# NAME / ADDRESS LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FIELD LENGTH STATISTICS")
print("=" * 70)


for name, df in [
    ("SOURCE 1", s1),
    ("SOURCE 2", s2),
    ("SOURCE 3", s3)
]:

    name_length = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    address_length = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(f"\n{name}")

    print(
        "Business name length:"
    )

    print(
        name_length.describe()
    )

    print(
        "Address length:"
    )

    print(
        address_length.describe()
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH EXPLORATION COMPLETE")
print("=" * 70)