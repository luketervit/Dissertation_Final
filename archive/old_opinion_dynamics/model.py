"""
Mesa ABM Model for Historical Replay simulation.
"""
from mesa import Model, DataCollector
import pandas as pd
import json
import yaml
import numpy as np

from sim.agents import ActiveAgent, LurkerAgent


class HistoricalReplayModel(Model):
    """
    Agent-Based Model for replaying a Twitter thread with bounded confidence
    and backfire effects.
    """

    def __init__(self, config_path='config/thread_config.yaml'):
        """
        Initialize the model from config file.

        Args:
            config_path: Path to thread_config.yaml
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ABM parameters from config
        self.bounded_confidence_threshold = self.config['abm']['bounded_confidence_threshold']
        self.backfire_threshold = self.config['abm']['backfire_threshold']
        self.backfire_aggression_min = self.config['abm']['backfire_aggression_min']
        self.lurker_ratio = self.config['abm']['lurker_ratio']
        self.lurker_dist_strategy = self.config['abm']['lurker_dist_strategy']

        # Temporal state
        self.current_time = 0  # Seconds since start
        self.current_step = 0

        # Load thread data
        self._load_thread_data()

        # Initialize agents (Mesa 3.x manages agents internally)
        self._initialize_agents()

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                'time': lambda m: m.current_time,
                'step': lambda m: m.current_step,
                'mean_opinion': lambda m: np.mean([a.opinion for a in m.agents]),
                'opinion_variance': lambda m: np.var([a.opinion for a in m.agents]),
                'left_opinion': lambda m: np.mean([a.opinion for a in m.agents if a.political_label == 'Left']),
                'right_opinion': lambda m: np.mean([a.opinion for a in m.agents if a.political_label == 'Right']),
                'mean_emotional_residue': lambda m: np.mean([a.emotional_residue for a in m.agents]),
                'total_exposures': lambda m: sum([len(a.exposure_log) for a in m.agents])
            },
            agent_reporters={
                'opinion': 'opinion',
                'political_label': 'political_label',
                'emotional_residue': 'emotional_residue',
                'role': 'role'
            }
        )

        print(f"✓ Model initialized:")
        print(f"  Active agents: {len(self.active_agents)}")
        print(f"  Lurker agents: {len(self.lurker_agents)}")
        print(f"  Total agents: {len(self.agents)}")
        print(f"  Temporal events: {len(self.temporal_events)}")
        print(f"  Timeline: 0 to {self.timeline_duration} seconds")

    def _load_thread_data(self):
        """Load thread metadata, tweets, and agent DNA."""
        # Load metadata
        with open(self.config['paths']['thread_metadata'], 'r') as f:
            self.metadata = json.load(f)

        self.temporal_events = self.metadata['temporal_events']
        self.timeline_duration = self.metadata['timeline']['duration_seconds']

        # Load agent DNA
        agents_df = pd.read_csv(self.config['paths']['agents_for_thread'])
        self.agents_dna = agents_df.to_dict('records')

        # Create user_id -> DNA lookup
        self.user_dna_map = {row['user_id']: row for row in self.agents_dna}

        print(f"✓ Loaded thread data:")
        print(f"  Temporal events: {len(self.temporal_events)}")
        print(f"  Agent DNA profiles: {len(self.agents_dna)}")

    def _initialize_agents(self):
        """Initialize active and lurker agents."""
        agent_id = 0

        # Initialize active agents (real users)
        self.active_agents = []
        self.user_agent_map = {}  # user_id -> agent

        for dna in self.agents_dna:
            agent = ActiveAgent(
                unique_id=agent_id,
                model=self,
                user_id=dna['user_id'],
                dna=dna
            )
            self.register_agent(agent)
            self.active_agents.append(agent)
            self.user_agent_map[dna['user_id']] = agent
            agent_id += 1

        # Initialize lurker agents (if lurker_ratio > 0)
        self.lurker_agents = []
        num_lurkers = len(self.active_agents) * self.lurker_ratio

        if num_lurkers > 0:
            # Get lurker distribution
            if self.lurker_dist_strategy == "match_active_agents":
                lurker_dist = self.config['lurker_agents_dist']
            elif self.lurker_dist_strategy == "pew_2024":
                lurker_dist = self.config['abm']['pew_2024_dist']
            else:
                raise ValueError(f"Unknown lurker_dist_strategy: {self.lurker_dist_strategy}")

            # Spawn lurkers according to distribution
            labels = list(lurker_dist.keys())
            probs = list(lurker_dist.values())

            for _ in range(num_lurkers):
                # Sample political label
                political_label = np.random.choice(labels, p=probs)

                agent = LurkerAgent(
                    unique_id=agent_id,
                    model=self,
                    political_label=political_label
                )
                self.register_agent(agent)
                self.lurker_agents.append(agent)
                agent_id += 1

    def step(self):
        """
        Advance the model by one temporal event.
        Inject the next tweet and update all agents.
        """
        if self.current_step >= len(self.temporal_events):
            self.running = False
            return

        # Get next event
        event = self.temporal_events[self.current_step]
        self.current_time = event['seconds_since_start']

        # Get the agent who posted this tweet
        poster_user_id = event['user_id']
        if poster_user_id not in self.user_agent_map:
            # User not in our DNA set (shouldn't happen with 100% coverage)
            self.current_step += 1
            return

        poster_agent = self.user_agent_map[poster_user_id]

        # Record that poster posted this tweet
        poster_agent.post_tweet(event['tweet_id'], self.current_time)

        # All other agents view this tweet
        for agent in self.agents:
            if agent.unique_id != poster_agent.unique_id:
                agent.view_tweet(event, poster_agent)

        # Collect data
        self.datacollector.collect(self)

        # Advance step
        self.current_step += 1

    def run_simulation(self):
        """Run the complete simulation (all temporal events)."""
        print(f"\n{'='*80}")
        print("RUNNING HISTORICAL REPLAY SIMULATION")
        print(f"{'='*80}")

        while self.running and self.current_step < len(self.temporal_events):
            self.step()

            # Progress reporting
            if self.current_step % 20 == 0:
                progress = self.current_step / len(self.temporal_events) * 100
                print(f"  Step {self.current_step}/{len(self.temporal_events)} ({progress:.1f}%) | "
                      f"t={self.current_time}s | "
                      f"Mean opinion: {np.mean([a.opinion for a in self.agents]):.3f}")

        print(f"\n✓ Simulation complete!")
        print(f"  Total steps: {self.current_step}")
        print(f"  Final time: {self.current_time}s")

    def get_results(self):
        """Get simulation results as DataFrames."""
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()

        return model_data, agent_data

    def get_agent_summaries(self):
        """Get summary statistics for all agents."""
        summaries = []
        for agent in self.agents:
            summaries.append(agent.get_summary())
        return pd.DataFrame(summaries)
