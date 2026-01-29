"""
Stance Classification Model
Uses RoBERTa to classify political stance of tweets.
"""

from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StanceClassifier:
    """Classifies tweet stance using RoBERTa."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-stance"):
        """Initialize the stance classifier."""
        # TODO: Implement initialization
        pass

    def classify(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Classify stance for a list of tweets."""
        # TODO: Implement classification
        pass
