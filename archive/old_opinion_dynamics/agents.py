"""
Agent classes for Historical Replay ABM.

ActiveAgent: Real users who posted in the thread
LurkerAgent: Simulated users who viewed but didn't post
"""
from mesa import Agent


class ActiveAgent(Agent):
    """
    An active user who actually posted in the thread.
    Has DNA from classification (political_label, emotion, aggression).
    """

    def __init__(self, unique_id, model, user_id, dna):
        """
        Initialize an active agent with their DNA profile.

        Args:
            unique_id: Mesa agent ID
            model: The HistoricalReplayModel instance
            user_id: Twitter user ID
            dna: Dict with DNA features from agents_for_tweet.csv
        """
        super().__init__(model)
        self.unique_id = unique_id

        # Identity
        self.user_id = user_id
        self.role = dna['role']  # 'root' or 'active'

        # Political DNA (categorical + confidence)
        self.political_label = dna['political_label']  # Left/Center/Right
        self.political_score = dna['political_score']  # Confidence [0,1]

        # Map political label to continuous opinion [0,1]
        # Left = 0.0-0.33, Center = 0.33-0.67, Right = 0.67-1.0
        self.opinion = self._label_to_opinion(self.political_label)

        # Emotional/behavioral DNA
        self.emotion_label = dna['emotion_label']
        self.emotion_score = dna['emotion_score']
        self.sentiment_label = dna['sentiment_label']
        self.sentiment_score = dna['sentiment_score']

        # Aggression (hate + offensive)
        self.hate_score = dna['hate_score']
        self.offensive_score = dna['offensive_score']
        self.aggression = self.hate_score + self.offensive_score

        # Engagement metrics
        self.tweet_count = dna['tweet_count']
        self.view_count = dna['view_count']

        # Memory state (for bounded confidence and backfire)
        self.exposure_log = []  # Recent tweets viewed
        self.emotional_residue = 0.0  # Accumulated emotional impact
        self.belief_inertia = self.political_score  # Resistance to change

        # Tracking
        self.tweets_posted = []  # List of (time, tweet_id) tuples
        self.opinion_history = [(0, self.opinion)]  # Track opinion shifts

    def _label_to_opinion(self, label):
        """Convert political label to continuous opinion [0,1]."""
        if label == 'Left':
            return 0.17  # Center of Left range [0.0-0.33]
        elif label == 'Center':
            return 0.50  # Center of Center range [0.33-0.67]
        elif label == 'Right':
            return 0.83  # Center of Right range [0.67-1.0]
        else:
            return 0.50  # Default to center

    def view_tweet(self, tweet_event, poster_agent):
        """
        Process viewing a tweet from another agent.
        Implements bounded confidence and backfire effect.

        Args:
            tweet_event: Dict with tweet data (from temporal_events)
            poster_agent: The agent who posted the tweet
        """
        # Add to exposure log (keep last K tweets)
        self.exposure_log.append({
            'time': self.model.current_time,
            'poster_opinion': poster_agent.opinion,
            'poster_aggression': poster_agent.aggression,
            'poster_label': poster_agent.political_label
        })

        # Keep only last 10 exposures
        if len(self.exposure_log) > 10:
            self.exposure_log.pop(0)

        # Calculate opinion distance
        distance = abs(self.opinion - poster_agent.opinion)

        # Bounded Confidence: If close enough, move toward tweet
        bc_threshold = self.model.bounded_confidence_threshold
        if distance < bc_threshold:
            # Aligned exposure - reinforce opinion slightly
            influence = 0.05 * (1 - self.belief_inertia)  # Less if high inertia
            self.opinion += influence * (poster_agent.opinion - self.opinion)

            # Decrease emotional residue
            self.emotional_residue *= 0.9

            # Increase belief inertia (hardening)
            self.belief_inertia = min(1.0, self.belief_inertia + 0.01)

        # Backfire Effect: If far + aggressive, move AWAY
        elif distance > self.model.backfire_threshold and poster_agent.aggression > self.model.backfire_aggression_min:
            # Oppositional exposure with high aggression
            backfire_strength = 0.03 * poster_agent.aggression
            self.opinion -= backfire_strength * (poster_agent.opinion - self.opinion)

            # Increase emotional residue
            self.emotional_residue += poster_agent.aggression * 0.1

        # Keep opinion in [0,1]
        self.opinion = max(0.0, min(1.0, self.opinion))

        # Track opinion change
        self.opinion_history.append((self.model.current_time, self.opinion))

    def post_tweet(self, tweet_id, time):
        """Record that this agent posted a tweet."""
        self.tweets_posted.append((time, tweet_id))

    def get_summary(self):
        """Get agent summary for reporting."""
        return {
            'user_id': self.user_id,
            'role': self.role,
            'political_label': self.political_label,
            'initial_opinion': self.opinion_history[0][1],
            'final_opinion': self.opinion_history[-1][1],
            'opinion_shift': self.opinion_history[-1][1] - self.opinion_history[0][1],
            'aggression': self.aggression,
            'tweets_posted': len(self.tweets_posted),
            'tweets_viewed': len(self.exposure_log)
        }


class LurkerAgent(Agent):
    """
    A lurker who views the thread but doesn't post.
    Spawned with Pew 2024 or match_active_agents distribution.
    """

    def __init__(self, unique_id, model, political_label):
        """
        Initialize a lurker agent.

        Args:
            unique_id: Mesa agent ID
            model: The HistoricalReplayModel instance
            political_label: Assigned political leaning (Left/Center/Right)
        """
        super().__init__(model)
        self.unique_id = unique_id

        # Identity
        self.user_id = None  # Lurkers don't have real user IDs
        self.role = 'lurker'

        # Political DNA (assigned, not observed)
        self.political_label = political_label
        self.political_score = 0.7  # Moderate confidence (less convicted than actives)
        self.opinion = self._label_to_opinion(political_label)

        # Behavioral params (average of population)
        self.aggression = 0.3  # Moderate aggression
        self.emotion_label = 'neutral'
        self.sentiment_label = 'neutral'

        # Memory state
        self.exposure_log = []
        self.emotional_residue = 0.0
        self.belief_inertia = self.political_score
        self.expressive_pressure = 0.0  # Could trigger posting (future work)

        # Tracking
        self.opinion_history = [(0, self.opinion)]

    def _label_to_opinion(self, label):
        """Convert political label to continuous opinion [0,1]."""
        if label == 'Left':
            return 0.17
        elif label == 'Center':
            return 0.50
        elif label == 'Right':
            return 0.83
        else:
            return 0.50

    def view_tweet(self, tweet_event, poster_agent):
        """
        Process viewing a tweet (same logic as ActiveAgent).
        """
        # Add to exposure log
        self.exposure_log.append({
            'time': self.model.current_time,
            'poster_opinion': poster_agent.opinion,
            'poster_aggression': poster_agent.aggression,
            'poster_label': poster_agent.political_label
        })

        if len(self.exposure_log) > 10:
            self.exposure_log.pop(0)

        # Calculate opinion distance
        distance = abs(self.opinion - poster_agent.opinion)

        # Bounded Confidence
        bc_threshold = self.model.bounded_confidence_threshold
        if distance < bc_threshold:
            influence = 0.05 * (1 - self.belief_inertia)
            self.opinion += influence * (poster_agent.opinion - self.opinion)
            self.emotional_residue *= 0.9
            self.belief_inertia = min(1.0, self.belief_inertia + 0.01)

        # Backfire Effect
        elif distance > self.model.backfire_threshold and poster_agent.aggression > self.model.backfire_aggression_min:
            backfire_strength = 0.03 * poster_agent.aggression
            self.opinion -= backfire_strength * (poster_agent.opinion - self.opinion)
            self.emotional_residue += poster_agent.aggression * 0.1

            # Increase expressive pressure (could convert to active)
            self.expressive_pressure += poster_agent.aggression * 0.2

        # Keep opinion in [0,1]
        self.opinion = max(0.0, min(1.0, self.opinion))

        # Track opinion change
        self.opinion_history.append((self.model.current_time, self.opinion))

    def get_summary(self):
        """Get agent summary for reporting."""
        return {
            'user_id': None,
            'role': 'lurker',
            'political_label': self.political_label,
            'initial_opinion': self.opinion_history[0][1],
            'final_opinion': self.opinion_history[-1][1],
            'opinion_shift': self.opinion_history[-1][1] - self.opinion_history[0][1],
            'aggression': self.aggression,
            'tweets_viewed': len(self.exposure_log),
            'expressive_pressure': self.expressive_pressure
        }
