import pandas as pd
import re
import unicodedata
import os


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """Normalize Unicode without removing non-English scripts."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    return text


def basic_clean(text):
    """
    Conservative text cleaning.

    Keeps Unicode characters and useful symbols.
    Does NOT transliterate or aggressively remove characters.
    """
    if pd.isna(text):
        return ""

    text = normalize_unicode(text)

    # Lowercase
    text = text.lower()

    # Normalize common whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces around punctuation
    text = re.sub(r"\s*([,./#&@'-])\s*", r"\1", text)

    return text.strip()


# ============================================================
# BUSINESS NAME CLEANING
# ============================================================

# Legal/business suffixes
LEGAL_SUFFIXES = {
    "private limited",
    "pvt limited",
    "pvt ltd",
    "private ltd",
    "limited",
    "ltd",
    "llc",
    "l.l.c",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "pllc",
    "llp",
    "lp",
}


def clean_business_name(name):
    """
    Conservative normalization of business names.
    """

    if pd.isna(name):
        return ""

    name = basic_clean(name)

    # Normalize ampersand
    name = name.replace("&", " and ")

    # Remove brackets but preserve their contents
    name = re.sub(r"[\(\)\[\]\{\}]", " ", name)

    # Normalize hyphens
    name = re.sub(r"[-–—]", " ", name)

    # Normalize apostrophes
    name = re.sub(r"[’`]", "'", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def remove_legal_suffix(name):
    """
    Remove common legal suffixes from the END of a business name.

    Example:
        "fh business pvt ltd"
        -> "fh business"

    We remove suffixes only at the end so we don't
    accidentally destroy meaningful words.
    """

    if not name:
        return ""

    tokens = name.split()

    changed = True

    while changed and tokens:
        changed = False

        # Try 3-word suffix
        if len(tokens) >= 3:
            last_three = " ".join(tokens[-3:])

            if last_three in LEGAL_SUFFIXES:
                tokens = tokens[:-3]
                changed = True
                continue

        # Try 2-word suffix
        if len(tokens) >= 2:
            last_two = " ".join(tokens[-2:])

            if last_two in LEGAL_SUFFIXES:
                tokens = tokens[:-2]
                changed = True
                continue

        # Try 1-word suffix
        if tokens[-1] in LEGAL_SUFFIXES:
            tokens = tokens[:-1]
            changed = True

    return " ".join(tokens)


def sorted_name_tokens(name):
    """
    Order-independent representation.

    Example:
        "desert society inc"
        -> "desert society"

        "society desert inc"
        -> "desert society"
    """

    core = remove_legal_suffix(name)

    tokens = core.split()

    return " ".join(sorted(tokens))


# ============================================================
# ADDRESS CLEANING
# ============================================================

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\brd\.\b": "road",
    r"\bdr\b": "drive",
    r"\bdr\.\b": "drive",
    r"\bave\b": "avenue",
    r"\bave\.\b": "avenue",
    r"\bav\b": "avenue",
    r"\bst\b": "street",
    r"\bst\.\b": "street",
    r"\bcir\b": "circle",
    r"\bcir\.\b": "circle",
    r"\bblvd\b": "boulevard",
    r"\bblvd\.\b": "boulevard",
    r"\bln\b": "lane",
    r"\bln\.\b": "lane",
    r"\bct\b": "court",
    r"\bct\.\b": "court",
    r"\bhwy\b": "highway",
    r"\bhwy\.\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bpkwy\.\b": "parkway",
}


def clean_address(address):
    """
    Conservative address normalization.
    """

    if pd.isna(address):
        return ""

    address = basic_clean(address)

    # Normalize common symbols
    address = address.replace("#", " ")
    address = address.replace(";", ",")

    # Normalize hyphens
    address = re.sub(r"[-–—]", "-", address)

    # Normalize common street abbreviations
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        address = re.sub(pattern, replacement, address)

    # Normalize punctuation spacing
    address = re.sub(r"\s*,\s*", ", ", address)

    # Collapse spaces
    address = re.sub(r"\s+", " ", address)

    return address.strip(" ,")


# ============================================================
# ADDRESS FEATURES
# ============================================================

def extract_address_numbers(address):
    """
    Extract numbers from an address.

    Examples:
        4038 Talmadge Road
        -> 4038

        9/1/3 Kasundia
        -> 9, 1, 3
    """

    if pd.isna(address):
        return ""

    numbers = re.findall(r"\d+(?:/\d+)*", str(address))

    return " ".join(numbers)


def address_tokens(address):
    """
    Tokenized address representation.
    """

    if not address:
        return ""

    # Remove commas for token representation
    text = address.replace(",", " ")

    tokens = text.split()

    return " ".join(tokens)


# ============================================================
# COUNTRY CLEANING
# ============================================================

def clean_country(country):
    """
    Normalize country labels without hard-coding
    US / India / France.
    """

    if pd.isna(country):
        return ""

    country = normalize_unicode(country)

    country = country.strip().lower()

    country = re.sub(r"\s+", " ", country)

    return country


# ============================================================
# MAIN DATAFRAME CLEANING
# ============================================================

def clean_dataframe(df):
    """
    Apply all cleaning transformations.

    Original columns are preserved.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Missing flags
    # --------------------------------------------------------

    df["name_missing"] = (
        df["business_name"].isna()
        | df["business_name"].astype(str).str.strip().eq("")
    )

    df["address_missing"] = (
        df["business_address"].isna()
        | df["business_address"].astype(str).str.strip().eq("")
    )

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    df["name_clean"] = df["business_name"].apply(
        clean_business_name
    )

    df["name_core"] = df["name_clean"].apply(
        remove_legal_suffix
    )

    df["name_sorted"] = df["name_clean"].apply(
        sorted_name_tokens
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    df["address_clean"] = df["business_address"].apply(
        clean_address
    )

    df["address_tokens"] = df["address_clean"].apply(
        address_tokens
    )

    df["address_numbers"] = df["address_clean"].apply(
        extract_address_numbers
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    df["country_clean"] = df["country"].apply(
        clean_country
    )

    return df


# ============================================================
# LOAD / SAVE
# ============================================================

if __name__ == "__main__":

    print("Loading datasets...")

    s1 = pd.read_csv(

        "../student_resource/dataset/train/train_source1.tsv",

        sep="\t"

    )

    s2 = pd.read_csv(

        "../student_resource/dataset/train/train_source2.tsv",

        sep="\t"

    )

    s3 = pd.read_csv(

        "../student_resource/dataset/train/train_source3.tsv",

        sep="\t"

    )

    print("S1:", s1.shape)

    print("S2:", s2.shape)

    print("S3:", s3.shape)

    print("\nCleaning S1...")

    s1 = clean_dataframe(s1)

    print("Cleaning S2...")

    s2 = clean_dataframe(s2)

    print("Cleaning S3...")

    s3 = clean_dataframe(s3)

    print("\nCleaning complete!")

    print("\nFinal shapes:")

    print("S1:", s1.shape)

    print("S2:", s2.shape)

    print("S3:", s3.shape)

    print("\nS1 sample:")

    print(

        s1[

            [

                "entity_id",

                "business_name",

                "name_clean",

                "name_core",

                "business_address",

                "address_clean",

                "country_clean"

            ]

        ].head()
    )

    os.makedirs("dataset/processed", exist_ok=True)

    print("\nSaving cleaned datasets...")

    s1.to_parquet(

        "../processed/source1_clean.parquet",

        index=False

    )

    s2.to_parquet(

        "../processed/source2_clean.parquet",

        index=False

    )

    s3.to_parquet(

        "../processed/source3_clean.parquet",

        index=False

    )

    print("Saved successfully!")