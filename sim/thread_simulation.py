"""
Twitter Thread Simulation using Mesa ABM.

Agents generate text replies to a base tweet and each other,
building a conversation thread with depth tracking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesa import Agent, Model, DataCollector
import pandas as pd
import json
import yaml
import numpy as np
import random

from sim.llm_generator import LLMGenerator


class ThreadAgent(Agent):
    """
    An agent that can read a thread and generate replies based on their persona.
    """

    def __init__(self, unique_id, model, user_id, dna):
        """
        Initialize agent with DNA persona.

        Args:
            unique_id: Mesa agent unique identifier
            model: The ThreadModel instance
            user_id: Twitter user ID
            dna: Dict with political_label, emotion, aggression, etc.
        """
        # Mesa 2.4.0 compat: replicate Agent.__init__ without
        # calling super() which hits object.__init__() on Python 3.9
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        if hasattr(model, 'register_agent'):
            model.register_agent(self)
        self.user_id = user_id

        # Persona from DNA
        self.political_label = dna['political_label']
        self.political_score = dna['political_score']
        self.emotion_label = dna['emotion_label']
        self.aggression = (
            dna.get('hate_score', 0) + dna.get('offensive_score', 0)
        )
        self.role = dna.get('role', 'active')

        # Staging variables
        self.pending_reply = None  # Generated in stage 1
        self.reply_target = None   # Who we're replying to

    def step(self):
        """
        Stage 1: Generate a reply based on current thread snapshot.
        Agents read thread_history and decide whether/how to respond.
        """
        # Reply probability varies by agent personality
        # Aggressive agents reply more, passive agents reply less
        base_prob = 0.05  # Base 5% chance
        aggression_boost = self.aggression * 0.15  # Up to +15% for high aggression
        reply_prob = min(0.25, base_prob + aggression_boost)  # Max 25%

        if random.random() > reply_prob:
            return

        # Read current thread
        thread = self.model.thread_history

        if len(thread) == 0:
            # No posts yet, can't reply
            return

        # Select a post to reply to (prefer recent, or controversial)
        target_post = self._select_reply_target(thread)

        if target_post is None:
            return

        # Generate reply based on persona and target
        reply_text = self._generate_reply(target_post)

        # Store for commit stage
        self.pending_reply = reply_text
        self.reply_target = target_post['post_id']

    def advance(self):
        """
        Stage 2: Commit generated reply to thread_history.
        All agents commit simultaneously after all have generated.
        """
        if self.pending_reply is not None:
            # Add to model's thread
            self.model.add_post(
                user_id=self.user_id,
                text=self.pending_reply,
                parent_id=self.reply_target,
                political_label=self.political_label,
                emotion=self.emotion_label,
                aggression=self.aggression
            )

            # Reset for next round
            self.pending_reply = None
            self.reply_target = None

    def _select_reply_target(self, thread):
        """
        Select which post to reply to.
        Prefer recent posts, controversial posts, or posts from current round.
        """
        if len(thread) == 0:
            return None

        # Only consider recent posts (last 50 or current/previous round)
        current_round = self.model.current_round
        recent_posts = [p for p in thread
                       if p['round'] >= current_round - 1 or p in thread[-50:]]

        if not recent_posts:
            recent_posts = thread[-20:]  # Fallback to last 20

        # Weight posts
        weights = []
        for post in recent_posts:
            weight = 1.0

            # Boost if from current round (ongoing conversation)
            if post['round'] == current_round:
                weight *= 3.0

            # Boost if opposing political view (controversy)
            political_distance = abs(
                self._label_to_value(self.political_label) -
                self._label_to_value(post.get('political_label', 'Center'))
            )

            # Aggressive agents seek out opposing views
            if self.aggression > 0.5 and political_distance > 0.5:  # Original baseline
                weight *= 2.5  # Original baseline controversy weight
            # Moderate agents avoid extreme disagreement
            elif self.aggression < 0.3 and political_distance > 0.6:  # Original baseline
                weight *= 0.3
            # Everyone likes some controversy
            else:
                weight *= (1 + political_distance)

            # Slight preference for shallower depth (avoid super-deep threads)
            if post['depth'] > 5:
                weight *= 0.5

            weights.append(weight)

        # Sample weighted
        total = sum(weights)
        if total == 0:
            return random.choice(recent_posts)

        probs = [w/total for w in weights]
        return np.random.choice(recent_posts, p=probs)

    def _label_to_value(self, label):
        """Convert political label to numeric for distance calc."""
        return {'Left': 0, 'Center': 0.5, 'Right': 1}.get(label, 0.5)

    def _generate_reply(self, target_post):
        """
        Generate reply text using LLM based on persona and target post.

        Passes few-shot examples and thread mean aggression from the
        model so the LLM adapts to the thread's actual tone.
        """
        # Build agent persona dict
        agent_persona = {
            'political_label': self.political_label,
            'aggression': self.aggression,
            'emotion': self.emotion_label,
        }

        # Get thread context (recent posts for context)
        thread_context = self.model.thread_history[-10:]

        # Generate using LLM with few-shot grounding + dynamic aggression
        try:
            reply = self.model.llm_generator.generate_reply(
                agent_persona=agent_persona,
                target_post=target_post,
                thread_context=thread_context,
                few_shot_examples=self.model.few_shot_examples,
                thread_mean_aggression=self.model.thread_mean_aggression,
            )
            # Live monitoring
            print(
                f"  > Agent {self.user_id} generated a reply "
                f"({len(reply)} chars)"
            )
            return reply
        except Exception as e:
            print(f"LLM generation error for user {self.user_id}: {e}")
            return "I have thoughts on this."



class ThreadModel(Model):
    """
    Mesa model for simulating a Twitter thread conversation.
    """

    def __init__(self, config_path='config/thread_config.yaml'):
        """
        Initialize the thread simulation model.

        Args:
            config_path: Path to configuration file
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load thread metadata
        with open(self.config['paths']['thread_metadata'], 'r') as f:
            metadata = json.load(f)

        # Base tweet (root of conversation)
        self.base_tweet = {
            'post_id': 0,
            'user_id': metadata['root_tweet']['user_id'],
            'text': metadata['root_tweet']['text'],
            'parent_id': None,
            'depth': 0,
            'political_label': None,
            'emotion': None,
            'aggression': 0,
            'round': 0,
        }

        # Thread history (all posts including base)
        self.thread_history = [self.base_tweet]
        self.post_id_counter = 1
        self.current_round = 0

        # ---------------------------------------------------------
        # Few-shot grounding: extract real tweet texts to anchor the
        # LLM's tone to the actual thread discourse style.
        # ---------------------------------------------------------
        self.few_shot_examples = self._build_few_shot_examples(metadata)

        # ---------------------------------------------------------
        # Thread-level mean aggression for dynamic tone scaling.
        # Computed from agents CSV so the LLM adapts to each
        # thread's hostility level (fixes r = 0.20 correlation).
        # ---------------------------------------------------------
        self.thread_mean_aggression = self._compute_thread_mean_aggression()

        # Initialize LLM generator
        print("\n✓ Initializing LLM generator...")
        self.llm_generator = LLMGenerator(self.config['llm'])
        print(f"  Provider: {self.config['llm']['provider']}")
        print(f"  Model: {self.config['llm']['model']}")
        print(
            f"  Thread mean aggression: "
            f"{self.thread_mean_aggression:.3f}"
        )
        print(f"  Few-shot examples: {len(self.few_shot_examples)}")

        # Initialize agents (manual staging, no scheduler needed)
        self.agent_list = []
        self._initialize_agents()

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                'round': lambda m: m.current_round,
                'total_posts': lambda m: len(m.thread_history),
                'max_depth': lambda m: max([p['depth'] for p in m.thread_history]),
                'left_posts': lambda m: sum(1 for p in m.thread_history if p.get('political_label') == 'Left'),
                'right_posts': lambda m: sum(1 for p in m.thread_history if p.get('political_label') == 'Right'),
                'mean_aggression': lambda m: np.mean([p['aggression'] for p in m.thread_history if p['aggression'] > 0]) if any(p['aggression'] > 0 for p in m.thread_history) else 0
            }
        )

        print(f"✓ Thread simulation initialized:")
        print(f"  Agents: {len(self.agent_list)}")
        print(f"  Base tweet: \"{self.base_tweet['text'][:80]}...\"")

    # ------------------------------------------------------------------
    # Thread-level context helpers
    # ------------------------------------------------------------------

    def _build_few_shot_examples(
        self, metadata: dict, n: int = 8
    ) -> list[str]:
        """
        Extract up to *n* real tweet texts from the thread metadata
        for few-shot grounding in the LLM user prompt.

        Supports two metadata formats:
        - ``temporal_events`` list (original single-thread pipeline)
        - Root tweet text only (batch pipeline via prepare_batch)

        The sample is cached per-run so every agent sees the same
        grounding examples within a simulation.
        """
        examples: list[str] = []

        # Try temporal_events first (has full reply texts)
        events = metadata.get('temporal_events', [])
        if events:
            # Exclude root (is_root=True), keep reply texts
            reply_texts = [
                e['text']
                for e in events
                if not e.get('is_root', False) and e.get('text')
            ]
            if reply_texts:
                sample_size = min(n, len(reply_texts))
                examples = random.sample(reply_texts, sample_size)
                return examples

        # Fallback: use the root tweet text as the sole grounding
        # example. Even one real example helps anchor the LLM to
        # the thread's topic and vocabulary.
        root_text = metadata.get('root_tweet', {}).get('text', '')
        if root_text:
            examples.append(root_text)

        return examples

    def _compute_thread_mean_aggression(self) -> float:
        """
        Compute mean aggression (hate_score + offensive_score) from the
        agents CSV for this thread.  Used for dynamic aggression
        scaling so the LLM adapts to the thread's hostility level.
        """
        agents_path = self.config['paths']['agents_for_thread']
        try:
            agents_df = pd.read_csv(agents_path)
            hate = agents_df.get('hate_score', pd.Series(dtype=float))
            offensive = agents_df.get(
                'offensive_score', pd.Series(dtype=float)
            )
            aggression = hate.fillna(0) + offensive.fillna(0)
            mean_agg = float(aggression.mean())
            return mean_agg if not np.isnan(mean_agg) else 0.3
        except Exception as e:
            print(f"  Warning: Could not compute mean aggression: {e}")
            return 0.3  # safe default

    def _initialize_agents(self):
        """Load agents from DNA profiles."""
        agents_df = pd.read_csv(self.config['paths']['agents_for_thread'])

        if 'user_id' not in agents_df.columns:
            raise ValueError(
                f"agents_for_thread.csv missing 'user_id' column. "
                f"Found columns: {list(agents_df.columns)}"
            )

        # Drop rows with missing user_id
        agents_df = agents_df.dropna(subset=['user_id'])

        for idx, row in agents_df.iterrows():
            agent = ThreadAgent(
                unique_id=self.next_id(),
                model=self,
                user_id=row['user_id'],
                dna=row.to_dict()
            )
            self.agent_list.append(agent)

    def add_post(self, user_id, text, parent_id, political_label, emotion, aggression):
        """
        Add a post to the thread history.
        Called by agents during advance stage.
        """
        # Find parent to calculate depth
        parent_post = next((p for p in self.thread_history if p['post_id'] == parent_id), None)
        depth = (parent_post['depth'] + 1) if parent_post else 0

        post = {
            'post_id': self.post_id_counter,
            'user_id': user_id,
            'text': text,
            'parent_id': parent_id,
            'depth': depth,
            'political_label': political_label,
            'emotion': emotion,
            'aggression': aggression,
            'round': self.current_round
        }

        self.thread_history.append(post)
        self.post_id_counter += 1

    def step(self):
        """
        Execute one round of the simulation with manual staging.
        Stage 1: All agents generate replies
        Stage 2: All agents commit replies simultaneously
        """
        print(f"\n  → Stage 1: Agents reading thread and generating replies...")

        # Stage 1: Generate replies (read thread, decide what to say)
        replies_generated = 0
        for i, agent in enumerate(self.agent_list):
            agent.step()

            # Count if agent generated a reply
            if agent.pending_reply is not None:
                replies_generated += 1
                # Show progress every 5 replies
                if replies_generated % 5 == 0:
                    print(f"    Generated {replies_generated} replies so far...")

        print(f"  → Stage 2: Committing {replies_generated} replies to thread...")

        # Stage 2: Commit replies (add to thread simultaneously)
        for agent in self.agent_list:
            agent.advance()

        self.current_round += 1
        self.datacollector.collect(self)

    def run(self, max_rounds=10):
        """
        Run the simulation for a fixed number of rounds.

        Args:
            max_rounds: Number of rounds to simulate
        """
        print(f"\n{'='*80}")
        print("RUNNING THREAD SIMULATION")
        print(f"{'='*80}")

        for round_num in range(max_rounds):
            print(f"\n{'─'*80}")
            print(f"ROUND {round_num + 1}/{max_rounds}")
            print(f"{'─'*80}")

            self.step()

            posts_this_round = sum(1 for p in self.thread_history if p['round'] == round_num)
            print(f"\n✓ Round {round_num + 1} complete: {posts_this_round} responses generated | "
                  f"Total Posts: {len(self.thread_history)} | "
                  f"Active Agents: {len(self.agent_list)}")
            
            # Export intermediate results for monitoring
            self.export_results()

        print(f"\n✓ Simulation complete!")
        print(f"  Total posts: {len(self.thread_history)}")
        print(f"  Max thread depth: {max([p['depth'] for p in self.thread_history])}")

    def export_results(self, output_dir='output'):
        """Export simulation results matching selected_thread_metadata.json format."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get root tweet and replies
        root = self.thread_history[0]
        replies = self.thread_history[1:]  # Everything after root

        # Sort by post_id to maintain temporal order
        sorted_thread = sorted(self.thread_history, key=lambda x: x['post_id'])

        # Build temporal_events array (matching format exactly)
        temporal_events = []
        for post in sorted_thread:
            event = {
                'tweet_id': post['post_id'],
                'user_id': post['user_id'],
                'timestamp': f"simulated_round_{post['round']}",  # Simulated, no real timestamp
                'epoch': post['round'],  # Use round as epoch
                'seconds_since_start': post['round'] * 60,  # Assume 1 minute per round
                'is_root': post['post_id'] == 0,
                'text': post['text']
            }
            temporal_events.append(event)

        # Build metadata matching selected_thread_metadata.json structure
        metadata = {
            'root_tweet': {
                'id': root['post_id'],
                'conversation_id': 0,  # Simulated
                'user_id': root['user_id'],
                'text': root['text'],
                'timestamp': 'simulated_round_0',
                'epoch': 0,
                'expected_replies': len(replies),
                'view_count': 0,  # Simulated, unknown
                'like_count': 0,
                'retweet_count': 0
            },
            'replies': {
                'actual_count': len(replies),
                'coverage_percent': 100.0,  # We have all simulated replies
                'tweet_ids': [p['post_id'] for p in replies],
                'user_ids': [p['user_id'] for p in replies],
                'timestamps': [f"simulated_round_{p['round']}" for p in replies],
                'epochs': [p['round'] for p in replies]
            },
            'timeline': {
                'start': 'simulated_round_0',
                'end': f"simulated_round_{self.current_round}",
                'start_epoch': 0,
                'end_epoch': self.current_round,
                'duration_seconds': self.current_round * 60,
                'total_events': len(temporal_events)
            },
            'temporal_events': temporal_events,
            'abm_config': {
                'active_agents': len(self.agent_list),
                'lurker_agents': 0,
                'total_agents': len(self.agent_list),
                'lurker_ratio': self.config.get('abm', {}).get('lurker_ratio', 0.0),
                'lurker_dist_strategy': self.config.get('abm', {}).get('lurker_dist_strategy', 'none'),
                'lurker_political_dist': self.config.get('lurker_agents_dist', {})
            },
            'simulation_info': {
                'total_posts': len(self.thread_history),
                'max_depth': max([p['depth'] for p in self.thread_history]),
                'rounds': self.current_round,
                'llm_provider': self.config['llm']['provider'],
                'llm_model': self.config['llm']['model']
            }
        }

        # 1. Export thread metadata (matching format)
        thread_json = output_path / 'simulated_thread_metadata.json'
        with open(thread_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved: {thread_json}")

        # 2. Export full thread history (simple format)
        thread_history_json = output_path / 'thread_history.json'
        with open(thread_history_json, 'w') as f:
            json.dump(self.thread_history, f, indent=2)
        print(f"✓ Saved: {thread_history_json}")

        # 3. Export summary statistics
        summary_txt = output_path / 'summary_stats.txt'
        with open(summary_txt, 'w') as f:
            f.write("THREAD SIMULATION SUMMARY\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total Responses: {len(replies)}\n")
            f.write(f"Max Thread Depth: {max([p['depth'] for p in self.thread_history])}\n")

            # Political breakdown
            left = sum(1 for p in self.thread_history if p.get('political_label') == 'Left')
            right = sum(1 for p in self.thread_history if p.get('political_label') == 'Right')
            center = sum(1 for p in self.thread_history if p.get('political_label') == 'Center')

            f.write(f"\nPolitical Distribution:\n")
            f.write(f"  Left: {left}\n")
            f.write(f"  Right: {right}\n")
            f.write(f"  Center: {center}\n")

            # Aggression stats
            aggressions = [p['aggression'] for p in self.thread_history if p['aggression'] > 0]
            if aggressions:
                f.write(f"\nAggression:\n")
                f.write(f"  Mean: {np.mean(aggressions):.3f}\n")
                f.write(f"  Max: {np.max(aggressions):.3f}\n")

        print(f"✓ Saved: {summary_txt}")

        # 4. Export time series data
        model_data = self.datacollector.get_model_vars_dataframe()
        model_data.to_csv(output_path / 'simulation_timeseries.csv')
        print(f"✓ Saved: {output_path / 'simulation_timeseries.csv'}")


def main():
    """Run the thread simulation."""
    print("="*80)
    print("TWITTER THREAD SIMULATION - TEXT GENERATION ABM")
    print("="*80)

    # Initialize model
    print("\nInitializing model...")
    model = ThreadModel(config_path='config/thread_config.yaml')

    # Run simulation
    model.run(max_rounds=10)

    # Export results
    print("\nExporting results...")
    model.export_results()

    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
