"""
Mesa ABM Model definition.
Orchestrates the simulation of political persuasion dynamics.
"""

from typing import List, Dict
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pandas as pd
import numpy as np


class PoliticalPersuasionModel(Model):
    """
    ABM simulating political persuasion on Twitter/X.

    Key features:
    - 90-9-1 Rule: Spawns ~15-20 lurkers per active agent
    - Temporal validation: 12h/12h split (Phase 1 playback, Phase 2 prediction)
    - Tracks "Ghost Shift" in lurker opinions
    """

    def __init__(
        self,
        active_agents_data: pd.DataFrame,
        num_lurkers_per_active: int = 17,  # Based on 90-9-1 rule
        bounded_confidence_threshold: float = 0.4,
        backfire_threshold: float = 0.6,
        backfire_aggression_threshold: float = 0.7,
    ):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.num_lurkers_per_active = num_lurkers_per_active
        self.bounded_confidence_threshold = bounded_confidence_threshold
        self.backfire_threshold = backfire_threshold
        self.backfire_aggression_threshold = backfire_aggression_threshold

        # Initialize data collector for tracking Ghost Shift
        self.datacollector = DataCollector(
            model_reporters={
                "mean_lurker_opinion": self._compute_mean_lurker_opinion,
                "lurker_opinion_std": self._compute_lurker_opinion_std,
                "ghost_shift": self._compute_ghost_shift,
            }
        )

        # TODO: Initialize agents from active_agents_data
        # TODO: Spawn lurkers based on 90-9-1 rule

    def _compute_mean_lurker_opinion(self) -> float:
        """Compute mean opinion of all lurker agents."""
        # TODO: Implement
        return 0.0

    def _compute_lurker_opinion_std(self) -> float:
        """Compute standard deviation of lurker opinions."""
        # TODO: Implement
        return 0.0

    def _compute_ghost_shift(self) -> float:
        """
        Compute cumulative opinion change in lurker population.
        This is the key "Ghost Shift" metric for the dissertation.
        """
        # TODO: Implement
        return 0.0

    def step(self) -> None:
        """Execute one step of the model."""
        self.datacollector.collect(self)
        self.schedule.step()
