"""Match notebook preprocessing for lexical / semantic fields."""

from __future__ import annotations

import json
import re

import pandas as pd


def to_text(x) -> str:
    if isinstance(x, list):
        return " ".join(map(str, x))
    if pd.isna(x):
        return ""
    return str(x)


def clean_lexical_text(text: str) -> str:
    text = to_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(row) -> str:
    title = to_text(row.get("title", ""))
    description = to_text(row.get("description", ""))
    features = to_text(row.get("features", ""))
    return f"{title} {description} {features}".strip()


def preprocess_products_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Same filters as the notebook: usable lexical + semantic text."""
    df = raw.copy()
    for num_col in ["price", "rating_number", "average_rating"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["combined_text"] = df.apply(build_combined_text, axis=1)
    df["lexical_text"] = df["combined_text"].apply(clean_lexical_text)
    df["semantic_text"] = (
        df["combined_text"]
        .apply(lambda t: re.sub(r"\s+", " ", to_text(t)).strip())
    )

    df = df[
        (df["lexical_text"].str.strip() != "")
        & (df["semantic_text"].str.strip() != "")
    ].reset_index(drop=True)
    return df
