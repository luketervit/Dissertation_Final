"""
Data processing utilities.
Loads and preprocesses USC X 24 US Election Dataset.
"""

from typing import Optional
import pandas as pd
from pathlib import Path


def load_conversations(data_path: Path) -> pd.DataFrame:
    """Load top_conversations.csv or chunk files."""
    # TODO: Implement
    pass


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract relevant features for ABM (viewCount, timestamps, etc.)."""
    # TODO: Implement
    pass
