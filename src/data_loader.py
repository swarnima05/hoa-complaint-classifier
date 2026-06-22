"""Data loading and text preprocessing utilities.

This module owns file access and text cleanup so model training and prediction
can share one consistent preprocessing path.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "complaints.csv"


def remove_punctuation(text: str) -> str:
    """Remove punctuation so word matching focuses on complaint content."""
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces created during text cleanup."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Lowercase and clean a complaint while preserving its words."""
    lowered_text = str(text).lower()
    text_without_punctuation = remove_punctuation(lowered_text)
    return normalize_whitespace(text_without_punctuation)


def add_clean_complaint_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the data with a clean text feature column."""
    prepared_dataframe = dataframe.copy()
    prepared_dataframe["clean_complaint"] = prepared_dataframe["complaint"].apply(clean_text)
    return prepared_dataframe


def load_complaints_csv(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the labeled complaints CSV from disk."""
    return pd.read_csv(csv_path)


def load_preprocessed_data(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load complaints and add the model-ready text column."""
    complaints_dataframe = load_complaints_csv(csv_path)
    return add_clean_complaint_column(complaints_dataframe)


if __name__ == "__main__":
    data = load_preprocessed_data()
    print(data.head())
    print(f"Loaded {len(data)} complaints from {DEFAULT_DATA_PATH}")
