"""
Emotion Classification Model
Uses RoBERTa to extract aggression/emotion scores from tweets.
"""

from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionClassifier:
    """Classifies tweet emotion/aggression using RoBERTa."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-emotion"):
        """Initialize the emotion classifier."""
        # TODO: Implement initialization
        pass

    def classify(self, texts: List[str]) -> List[float]:
        """Extract aggression scores for a list of tweets."""
        # TODO: Implement classification
        pass
