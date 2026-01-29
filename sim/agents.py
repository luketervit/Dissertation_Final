"""
Agent definitions for the ABM.
Implements Active and Lurker agents with opinion dynamics.
"""

from typing import Optional
from mesa import Agent
import numpy as np


class ActiveAgent(Agent):
    """
    Active Twitter user who posts content.

    Attributes based on "Latent DNA":
    - stance_score: Political stance (-1 to 1, derived from RoBERTa)
    - aggression_score: Aggression level (0 to 1, derived from RoBERTa)
    """

    def __init__(
        self,
        unique_id: int,
        model,
        stance_score: float,
        aggression_score: float,
    ):
        super().__init__(unique_id, model)
        self.stance_score = stance_score
        self.aggression_score = aggression_score
        self.agent_type = "active"

    def step(self) -> None:
        """Execute one step of the agent."""
        pass


class LurkerAgent(Agent):
    """
    Lurker (silent majority) who reads but doesn't post.

    Spawned based on 90-9-1 Rule: ~15-20 lurkers per active agent.
    Tracks opinion shift without posting.
    """

    def __init__(
        self,
        unique_id: int,
        model,
        initial_opinion: float,
        stubbornness: float = 0.5,
    ):
        super().__init__(unique_id, model)
        self.opinion = initial_opinion  # Current opinion (-1 to 1)
        self.initial_opinion = initial_opinion  # For tracking "Ghost Shift"
        self.stubbornness = stubbornness  # Resistance to change
        self.agent_type = "lurker"

    def update_opinion(self, tweet_stance: float, tweet_aggression: float) -> None:
        """
        Update opinion based on Bounded Confidence and Backfire Effect.

        Rules:
        - Bounded Confidence: If distance < threshold, move toward tweet
        - Backfire Effect: If distance > 0.6 AND aggression > 0.7, move away
        """
        distance = abs(self.opinion - tweet_stance)

        # TODO: Implement bounded confidence logic
        # TODO: Implement backfire effect logic
        pass

    def step(self) -> None:
        """Execute one step of the agent."""
        pass
