"""
PDS-24 — Stage 2: Preprocessing

Turns a raw dump of posts (from ANY source) into one clean, de-identified table:

    post_id | subreddit | cleaned_text | timestamp | metadata

The script is split in two so the data source can change without touching the cleaning:
  1. load_raw()   — reads a raw file + a small JSON "source config" that maps that file's
                    column names onto the standard raw columns (RAW_COLUMNS below).
                    New database = new source config in source_configs/, no code change.
  2. preprocess() — only ever sees the standard raw columns, so it's the same for every source.

Usage:
  py stage2_preprocessing.py --input sample_data/sample_raw_posts.csv --source-config source_configs/sample.json
"""


import argparse
import html
import json
import re
import sys
from pathlib import Path

#py -m pip install pandas
import pandas as pd


# --------------------------------------------------------------
# Standard raw columns - every source config maps onto these
# --------------------------------------------------------------
# post_id and at least one of title/body are required, the rest can be missing (filled with NA).
# "subreddit" is the community/forum name; keep the name even for non-Reddit sources (it's the agreed schema).
RAW_COLUMNS = ["post_id", "subreddit", "title", "body", "timestamp", "flair", "author", "parent_id"]


# --------------------------------------------------------------
# Cleaning settings - change here, not inside the functions
# --------------------------------------------------------------
# Body text Reddit (and its dumps) use for deleted / mod-removed posts
REMOVED_MARKERS = {"[deleted]", "[removed]", "[deleted by user]", "[removed by reddit]"}

# Accounts whose posts are dropped (lower case)
BOT_AUTHORS = {"automoderator"}

# Posts shorter than this (in words, after cleaning) are dropped as too short to carry a concern
MIN_WORDS = 5

# Only used when the source config sets "relevance_filter": true
# (e.g. a country subreddit where most posts are not about diabetes)
RELEVANCE_PATTERN = re.compile(
    r"\b(diabet\w*|t1d|t2d|insulin|metformin|a1c|hba1c|blood sugar|glucose|cgm|libre|dexcom|ozempic|glp-?1)\b",
    re.IGNORECASE,
)

# OPEN decision #10: source of diabetes type is not confirmed yet.
# Methods: "flair_then_keywords" (the proposal in the pipeline doc), "keywords", "flair", "subreddit", "none".
# Whatever is used is recorded per post in metadata.diabetes_type_source, so it can be redone later.
DIABETES_TYPE_METHOD = "flair_then_keywords"

# Only used by the "subreddit" method. Subreddits not listed get "unknown" (r/diabetes is mixed).
SUBREDDIT_TYPE_MAP = {
    "type1diabetes": "type1",
    "diabetes_t1": "type1",
    "diabetes_t2": "type2",
}

# Step 3 embedder - posts longer than EMBED_MAX_WORDPIECES are truncated by it (OPEN decision #7)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_MAX_WORDPIECES = 384


# --------------------------------------------------------------
# 1. Load: raw file + source config -> standard raw columns
# --------------------------------------------------------------
def load_source_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    if "post_id" not in config.get("columns", {}):
        raise ValueError(f"Source config {path} must map 'post_id' under 'columns'.")
    return config


def read_file(path: str, file_format: str | None) -> pd.DataFrame:
    # Format comes from the config, or else from the file extension
    file_format = file_format or Path(path).suffix.lstrip(".").lower()

    if file_format == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if file_format in ("jsonl", "ndjson"):
        return pd.read_json(path, lines=True, dtype=False)
    if file_format == "json":
        return pd.read_json(path, dtype=False)
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format '{file_format}'. Use csv, jsonl, json or parquet.")


def parse_timestamps(values: pd.Series, unit: str | None) -> pd.Series:
    # unit = "s" / "ms" for epoch numbers (Reddit's created_utc is seconds); None for date strings
    if unit:
        return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit=unit, utc=True)
    return pd.to_datetime(values, errors="coerce", utc=True)


def load_raw(input_path: str, config: dict) -> pd.DataFrame:
    df = read_file(input_path, config.get("format"))

    columns = config.get("columns", {})
    constants = config.get("constants", {})

    missing = [src for src in columns.values() if src and src not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing column(s) named in the source config: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Update 'columns' in the source config to match the data."
        )
    if not (columns.get("title") or columns.get("body") or "title" in constants or "body" in constants):
        raise ValueError("Source config must map at least one of 'title' or 'body'.")

    raw = pd.DataFrame(index=df.index)
    for col in RAW_COLUMNS:
        if columns.get(col):
            raw[col] = df[columns[col]]
        elif col in constants:
            raw[col] = constants[col]
        else:
            raw[col] = pd.NA

    # Prefix keeps IDs unique if several sources are merged later
    raw["post_id"] = config.get("id_prefix", "") + raw["post_id"].astype(str)
    raw["timestamp"] = parse_timestamps(raw["timestamp"], config.get("timestamp_unit"))
    for col in ["subreddit", "title", "body", "flair", "author", "parent_id"]:
        raw[col] = raw[col].astype("string")

    raw["source"] = config.get("name", Path(input_path).stem)
    raw["unit"] = config.get("unit", "post")  # "post" or "comment" (OPEN decision #10)
    return raw.reset_index(drop=True)


# --------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------
def drop_unusable(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    body = df["body"].fillna("").str.strip().str.lower()
    author = df["author"].fillna("").str.strip().str.lower()

    removed = body.isin(REMOVED_MARKERS)
    bot = author.isin(BOT_AUTHORS)
    no_id = df["post_id"].isin(["", "nan", "<NA>", "None"])

    report["dropped_deleted_or_removed"] = int(removed.sum())
    report["dropped_bot_posts"] = int((bot & ~removed).sum())
    report["dropped_missing_id"] = int((no_id & ~removed & ~bot).sum())
    return df[~(removed | bot | no_id)].copy()


def build_text(df: pd.DataFrame) -> pd.Series:
    # Title + body, since many posts put the question in the title and only context in the body
    title = df["title"].fillna("").str.strip()
    body = df["body"].fillna("").str.strip()
    return (title + "\n\n" + body).str.strip()


MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
REDDIT_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]+")
AT_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
# Lookarounds skip decimals (6.5) but still allow a full stop ending the sentence
PHONE_CANDIDATE = re.compile(r"(?<!\w)(?<!\d\.)\+?\(?\d[\d\s().-]{6,}\d(?!\w)(?!\.\d)")


def _phone_or_keep(match: re.Match) -> str:
    # Diabetes posts are full of numbers (readings, doses, years), so only treat a match as a phone
    # number if it has 8-15 digits, at least one group of 4+ digits, and isn't just years.
    text = match.group(0)
    groups = re.findall(r"\d+", text)
    n_digits = sum(len(g) for g in groups)
    if not 8 <= n_digits <= 15:
        return text
    if all(len(g) <= 3 for g in groups):
        return text  # e.g. "120 140 160 180" readings
    if all(len(g) == 4 and g[:2] in ("19", "20") for g in groups):
        return text  # e.g. "2019 - 2021"
    return "[PHONE]"


def deidentify(text: str) -> str:
    # NOTE: regex only removes structured identifiers. Names / places written in free text
    # ("my doctor Dr Tan at ...") are NOT removed - list as a limitation, or add NER later.
    text = html.unescape(text)
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = URL.sub("[URL]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = REDDIT_USER.sub("[USER]", text)
    text = AT_MENTION.sub("[USER]", text)
    text = PHONE_CANDIDATE.sub(_phone_or_keep, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_irrelevant(df: pd.DataFrame, relevance_filter: bool, report: dict) -> pd.DataFrame:
    n_words = df["cleaned_text"].str.split().str.len()
    too_short = n_words < MIN_WORDS
    report["dropped_too_short"] = int(too_short.sum())
    df = df[~too_short].copy()

    if relevance_filter:
        relevant = df["cleaned_text"].str.contains(RELEVANCE_PATTERN, na=False)
        report["dropped_not_diabetes_related"] = int((~relevant).sum())
        df = df[relevant].copy()
    return df


def deduplicate(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    # Done before any split (step 4) so a post can't end up in both train and test.
    # Exact duplicates only, after normalising case / punctuation / whitespace. Keeps the earliest copy.
    df = df.sort_values("timestamp", na_position="last")

    before = len(df)
    df = df.drop_duplicates(subset="post_id", keep="first")
    report["dropped_duplicate_ids"] = before - len(df)

    key = (
        df["cleaned_text"].str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    before = len(df)
    df = df[~key.duplicated(keep="first")]
    report["dropped_duplicate_text"] = before - len(df)
    return df.copy()


# --------------------------------------------------------------
# 3. Metadata: diabetes type + length in word pieces
# --------------------------------------------------------------
# "type 1.5" is LADA, so the type 1 pattern must not match it. Roman "type i" is left out
# because it also matches "type I'm ...".
TYPE_PATTERNS = {
    "type1": re.compile(r"\btype[\s-]*(1|one)(?![.\d])\b|\bt1(d|dm)?\b|\bjuvenile diabetes\b", re.IGNORECASE),
    "type2": re.compile(r"\btype[\s-]*(2|two|ii)\b|\bt2(d|dm)?\b", re.IGNORECASE),
    "lada": re.compile(r"\blada\b|\btype[\s-]*1\.5\b", re.IGNORECASE),
    "gestational": re.compile(r"\bgestational\b|\bgdm\b", re.IGNORECASE),
    "prediabetes": re.compile(r"\bpre-?diabet\w*", re.IGNORECASE),
}


def types_mentioned(text) -> list[str]:
    if pd.isna(text) or not text:
        return []
    return [t for t, pattern in TYPE_PATTERNS.items() if pattern.search(text)]


def classify_type(text) -> str:
    # "unknown" if no type is mentioned, "multiple" if more than one is (e.g. "pre-diabetic, scared of type 2")
    found = types_mentioned(text)
    if not found:
        return "unknown"
    return found[0] if len(found) == 1 else "multiple"


def diabetes_type(row: pd.Series, method: str) -> tuple[str, str]:
    """Returns (diabetes_type, where it came from)."""
    if method == "none":
        return "unknown", "none"

    if method == "subreddit":
        sub = str(row["subreddit"]).lower() if pd.notna(row["subreddit"]) else ""
        return SUBREDDIT_TYPE_MAP.get(sub, "unknown"), "subreddit"

    if method in ("flair", "flair_then_keywords"):
        flair_type = classify_type(row["flair"])
        if flair_type not in ("unknown", "multiple"):
            return flair_type, "flair"
        if method == "flair":
            return "unknown", "flair"

    if method in ("keywords", "flair_then_keywords"):
        return classify_type(row["cleaned_text"]), "keywords"

    raise ValueError(f"Unknown diabetes type method '{method}'.")


def count_wordpieces(texts: pd.Series) -> pd.Series | None:
    # Needs the step 3 tokenizer. Returns None (and the column is left empty) if transformers isn't installed.
    try:
        #py -m pip install transformers
        from transformers import AutoTokenizer
    except ImportError:
        print("WARNING: transformers not installed, skipping word-piece counts.", file=sys.stderr)
        return None

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tokenizer.model_max_length = 10**9  # only counting, silence the "too long" warning
    lengths = [len(ids) for ids in tokenizer(texts.tolist(), add_special_tokens=True)["input_ids"]]
    return pd.Series(lengths, index=texts.index)


# --------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------
def preprocess(raw: pd.DataFrame, relevance_filter: bool, type_method: str, wordpieces: bool) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": len(raw)}

    df = drop_unusable(raw, report)
    df["cleaned_text"] = build_text(df).map(deidentify)
    df = drop_irrelevant(df, relevance_filter, report)
    df = deduplicate(df, report)

    types = df.apply(lambda row: diabetes_type(row, type_method), axis=1)
    df["diabetes_type"] = [t for t, _ in types]
    df["diabetes_type_source"] = [s for _, s in types]
    df["n_words"] = df["cleaned_text"].str.split().str.len()

    n_wp = count_wordpieces(df["cleaned_text"]) if wordpieces and len(df) else None
    df["n_wordpieces"] = n_wp if n_wp is not None else pd.NA

    # Author is never written out - de-identification step 2
    meta_cols = ["source", "unit", "parent_id", "flair", "diabetes_type", "diabetes_type_source", "n_words", "n_wordpieces"]
    metadata = df[meta_cols].astype(object).where(df[meta_cols].notna(), None).to_dict(orient="records")

    out = pd.DataFrame({
        "post_id": df["post_id"],
        "subreddit": df["subreddit"],
        "cleaned_text": df["cleaned_text"],
        "timestamp": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": [json.dumps(m, ensure_ascii=False, default=int) for m in metadata],
    })

    report["output_rows"] = len(out)
    report["diabetes_type_counts"] = df["diabetes_type"].value_counts().to_dict()
    if n_wp is not None:
        over = int((n_wp > EMBED_MAX_WORDPIECES).sum())
        report["over_embed_limit"] = over
        report["share_over_embed_limit"] = round(over / len(df), 4) if len(df) else 0.0
    return out, report


def expand_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Helper for later stages: turn the metadata JSON column back into real columns."""
    meta = pd.json_normalize(df["metadata"].map(json.loads).tolist())
    meta.index = df.index
    return pd.concat([df.drop(columns="metadata"), meta], axis=1)


def main():
    parser = argparse.ArgumentParser(description="PDS-24 Stage 2: preprocess raw posts into the cleaned table.")
    parser.add_argument("--input", required=True, help="Raw data file (csv, jsonl, json or parquet)")
    parser.add_argument("--source-config", required=True, help="JSON file mapping the raw file's columns (see source_configs/)")
    parser.add_argument("--output", default="output/cleaned_posts.csv", help="Where to write the cleaned table")
    parser.add_argument("--diabetes-type-method", default=DIABETES_TYPE_METHOD,
                        choices=["flair_then_keywords", "keywords", "flair", "subreddit", "none"])
    parser.add_argument("--skip-wordpieces", action="store_true", help="Don't count word pieces (no transformers download)")
    args = parser.parse_args()

    config = load_source_config(args.source_config)
    raw = load_raw(args.input, config)
    out, report = preprocess(
        raw,
        relevance_filter=config.get("relevance_filter", False),
        type_method=args.diabetes_type_method,
        wordpieces=not args.skip_wordpieces,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    report_path = output.with_name(output.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {len(out)} rows to {output} (report: {report_path})")


if __name__ == "__main__":
    main()
