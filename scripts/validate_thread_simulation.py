"""
Thread Simulation Validation Script

Compares real vs simulated Twitter threads using:
- Sentiment classification (CardiffNLP Twitter-RoBERTa)
- Jensen-Shannon Divergence for distribution similarity
- Thread depth and engagement metrics
- Keyword drift analysis

Usage:
    python scripts/validate_thread_simulation.py \
        --real output/selected_thread_metadata.json \
        --simulated output/simulated_thread_metadata.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm


class ThreadValidator:
    """Validates simulated thread against real thread using sentiment analysis."""

    def __init__(self, device='auto'):
        """
        Initialize validator with sentiment model.

        Args:
            device: 'cuda', 'cpu', or 'auto' (auto-detect GPU)
        """
        print("\n" + "="*80)
        print("THREAD SIMULATION VALIDATOR")
        print("="*80)

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"\n✓ Using device: {self.device}")

        # Load sentiment model
        print("✓ Loading sentiment model (twitter-roberta-base-sentiment-latest)...")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        print("✓ Model loaded successfully\n")

    def load_thread(self, json_path):
        """
        Load thread from JSON file.

        Supports two formats:
        1. selected_thread_metadata.json (with temporal_events)
        2. simulated_thread_metadata.json (with temporal_events)
        3. thread_history.json (flat array of posts)

        Returns:
            list of dicts with 'text', 'post_id'/'tweet_id', 'parent_id'
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Format 1: metadata file with temporal_events
        if 'temporal_events' in data:
            tweets = []
            for event in data['temporal_events']:
                tweets.append({
                    'tweet_id': event.get('tweet_id', event.get('post_id')),
                    'text': event['text'],
                    'parent_id': None,  # temporal_events don't have parent_id
                    'is_root': event.get('is_root', False)
                })
            return tweets

        # Format 2: thread_history.json (flat array)
        elif isinstance(data, list):
            tweets = []
            for post in data:
                tweets.append({
                    'tweet_id': post.get('post_id', post.get('tweet_id')),
                    'text': post['text'],
                    'parent_id': post.get('parent_id'),
                    'is_root': post.get('parent_id') is None
                })
            return tweets

        else:
            raise ValueError(f"Unknown JSON format in {json_path}")

    def classify_sentiment(self, tweets):
        """
        Classify sentiment for all tweets.

        Args:
            tweets: List of tweet dicts with 'text' field

        Returns:
            List of dicts with added 'sentiment_label' and 'sentiment_score'
        """
        print(f"Classifying sentiment for {len(tweets)} tweets...")

        results = []
        for tweet in tqdm(tweets, desc="Sentiment classification"):
            text = tweet['text']

            # Tokenize and classify
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]

                # Get predicted label and confidence
                pred_idx = torch.argmax(probs).item()
                sentiment_label = self.sentiment_labels[pred_idx]
                sentiment_score = probs[pred_idx].item()

                # Add to tweet dict
                tweet_result = tweet.copy()
                tweet_result['sentiment_label'] = sentiment_label
                tweet_result['sentiment_score'] = sentiment_score
                results.append(tweet_result)

            except Exception as e:
                print(f"  Error classifying tweet {tweet.get('tweet_id')}: {e}")
                # Fallback to neutral
                tweet_result = tweet.copy()
                tweet_result['sentiment_label'] = 'Neutral'
                tweet_result['sentiment_score'] = 0.33
                results.append(tweet_result)

        return results

    def calculate_sentiment_distribution(self, classified_tweets):
        """
        Calculate sentiment distribution (Positive/Neutral/Negative percentages).

        Returns:
            dict: {'Positive': 0.3, 'Neutral': 0.5, 'Negative': 0.2}
        """
        total = len(classified_tweets)
        counts = Counter([t['sentiment_label'] for t in classified_tweets])

        distribution = {
            'Positive': counts.get('Positive', 0) / total,
            'Neutral': counts.get('Neutral', 0) / total,
            'Negative': counts.get('Negative', 0) / total
        }

        return distribution

    def calculate_jsd(self, dist1, dist2):
        """
        Calculate Jensen-Shannon Divergence between two sentiment distributions.

        Lower JSD = more similar (0 = identical, 1 = completely different)

        Returns:
            float: JSD score [0, 1]
        """
        # Ensure same order
        labels = ['Positive', 'Neutral', 'Negative']
        p = np.array([dist1[label] for label in labels])
        q = np.array([dist2[label] for label in labels])

        # JSD (scipy returns sqrt of JSD, so we square it)
        jsd = jensenshannon(p, q) ** 2
        return jsd

    def calculate_max_depth(self, tweets):
        """
        Calculate maximum thread depth by following parent_id links.

        Returns:
            int: Maximum depth (0 = root tweet only)
        """
        # Build parent map
        tweet_map = {t['tweet_id']: t for t in tweets}

        max_depth = 0
        for tweet in tweets:
            depth = 0
            current = tweet

            # Trace up to root
            while current.get('parent_id') is not None:
                parent_id = current['parent_id']
                if parent_id in tweet_map:
                    current = tweet_map[parent_id]
                    depth += 1
                else:
                    break  # Parent not in dataset

            max_depth = max(max_depth, depth)

        return max_depth

    def calculate_engagement_ratio(self, tweets):
        """
        Calculate average replies per post.

        Returns:
            float: Mean number of replies per tweet
        """
        # Count replies per parent
        reply_counts = Counter()
        for tweet in tweets:
            parent_id = tweet.get('parent_id')
            if parent_id is not None:
                reply_counts[parent_id] += 1

        # Include tweets with 0 replies
        all_tweet_ids = {t['tweet_id'] for t in tweets}
        for tweet_id in all_tweet_ids:
            if tweet_id not in reply_counts:
                reply_counts[tweet_id] = 0

        # Average
        if len(reply_counts) == 0:
            return 0.0

        return np.mean(list(reply_counts.values()))

    def extract_keywords(self, classified_tweets, sentiment_filter='Negative', top_n=5):
        """
        Extract top N keywords from tweets with specific sentiment.

        Args:
            classified_tweets: Tweets with sentiment_label
            sentiment_filter: 'Positive', 'Neutral', or 'Negative'
            top_n: Number of keywords to return

        Returns:
            list of (keyword, count) tuples
        """
        # Filter tweets
        filtered_tweets = [t for t in classified_tweets if t['sentiment_label'] == sentiment_filter]

        # Extract words (lowercase, alphanumeric only)
        words = []
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                     'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about', 'as',
                     'from', 'up', 'out', 'if', 'by', 'my', 'your', 'our'}

        for tweet in filtered_tweets:
            text = tweet['text'].lower()
            # Remove URLs, mentions, hashtags
            text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
            # Extract words (alphanumeric only, 3+ chars)
            word_list = re.findall(r'\b[a-z]{3,}\b', text)
            words.extend([w for w in word_list if w not in stopwords])

        # Count and return top N
        counts = Counter(words)
        return counts.most_common(top_n)

    def visualize_comparison(self, real_dist, sim_dist, output_path='output/sentiment_comparison.png'):
        """
        Create side-by-side bar chart comparing sentiment distributions.

        Args:
            real_dist: Real thread sentiment distribution
            sim_dist: Simulated thread sentiment distribution
            output_path: Where to save the plot
        """
        labels = ['Positive', 'Neutral', 'Negative']
        real_values = [real_dist[label] for label in labels]
        sim_values = [sim_dist[label] for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, real_values, width, label='Real Thread', alpha=0.8)
        bars2 = ax.bar(x + width/2, sim_values, width, label='Simulated Thread', alpha=0.8)

        # Formatting
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Sentiment Distribution: Real vs Simulated Thread', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"\n✓ Saved visualization: {output_path}")

    def generate_report(self, real_tweets, sim_tweets, jsd, real_depth, sim_depth,
                       real_engagement, sim_engagement, real_keywords, sim_keywords,
                       real_dist, sim_dist):
        """Print comprehensive validation report."""

        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)

        # Sentiment distributions
        print("\n📊 SENTIMENT DISTRIBUTIONS")
        print("-" * 80)
        print(f"{'Sentiment':<15} {'Real Thread':<20} {'Simulated Thread':<20}")
        print("-" * 80)
        for label in ['Positive', 'Neutral', 'Negative']:
            real_pct = real_dist[label] * 100
            sim_pct = sim_dist[label] * 100
            diff = abs(real_pct - sim_pct)
            print(f"{label:<15} {real_pct:>6.1f}%{'':<13} {sim_pct:>6.1f}%{'':<13} (Δ {diff:.1f}%)")

        # Jensen-Shannon Divergence
        print("\n📈 SIMILARITY METRICS")
        print("-" * 80)
        similarity_pct = (1 - jsd) * 100
        print(f"Jensen-Shannon Divergence: {jsd:.4f}")
        print(f"Sentiment Similarity: {similarity_pct:.1f}%")

        # Thread structure
        print("\n🧵 THREAD STRUCTURE")
        print("-" * 80)
        print(f"{'Metric':<30} {'Real':<15} {'Simulated':<15}")
        print("-" * 80)
        print(f"{'Total Tweets':<30} {len(real_tweets):<15} {len(sim_tweets):<15}")
        print(f"{'Maximum Depth':<30} {real_depth:<15} {sim_depth:<15}")
        print(f"{'Engagement Ratio (avg replies)':<30} {real_engagement:<15.2f} {sim_engagement:<15.2f}")

        # Keyword drift
        print("\n🔍 TOP NEGATIVE KEYWORDS")
        print("-" * 80)
        print(f"{'Real Thread':<40} {'Simulated Thread':<40}")
        print("-" * 80)
        max_rows = max(len(real_keywords), len(sim_keywords))
        for i in range(max_rows):
            real_kw = f"{real_keywords[i][0]} ({real_keywords[i][1]})" if i < len(real_keywords) else ""
            sim_kw = f"{sim_keywords[i][0]} ({sim_keywords[i][1]})" if i < len(sim_keywords) else ""
            print(f"{real_kw:<40} {sim_kw:<40}")

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        # Overall accuracy (weighted: 70% sentiment, 15% depth, 15% engagement)
        depth_similarity = 1 - abs(real_depth - sim_depth) / max(real_depth, sim_depth, 1)
        engagement_similarity = 1 - abs(real_engagement - sim_engagement) / max(real_engagement, sim_engagement, 1)

        overall_accuracy = (
            similarity_pct * 0.70 +
            depth_similarity * 100 * 0.15 +
            engagement_similarity * 100 * 0.15
        )

        print(f"\nThe simulation is {overall_accuracy:.1f}% accurate to the real thread.")
        print(f"  - Sentiment similarity: {similarity_pct:.1f}%")
        print(f"  - Structural similarity (depth): {depth_similarity*100:.1f}%")
        print(f"  - Engagement similarity: {engagement_similarity*100:.1f}%")

        if overall_accuracy >= 80:
            verdict = "EXCELLENT - Simulation closely matches real thread behavior"
        elif overall_accuracy >= 60:
            verdict = "GOOD - Simulation captures major patterns with some deviations"
        elif overall_accuracy >= 40:
            verdict = "FAIR - Simulation shows similarities but significant differences exist"
        else:
            verdict = "POOR - Simulation does not match real thread characteristics"

        print(f"\nAssessment: {verdict}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate simulated thread against real thread")
    parser.add_argument('--real', type=str, required=True,
                       help='Path to real thread JSON (e.g., selected_thread_metadata.json)')
    parser.add_argument('--simulated', type=str, required=True,
                       help='Path to simulated thread JSON (e.g., simulated_thread_metadata.json)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for model inference')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize validator
    validator = ThreadValidator(device=args.device)

    # Load threads
    print("Loading threads...")
    real_tweets = validator.load_thread(args.real)
    sim_tweets = validator.load_thread(args.simulated)
    print(f"  Real thread: {len(real_tweets)} tweets")
    print(f"  Simulated thread: {len(sim_tweets)} tweets\n")

    # Classify sentiment
    real_classified = validator.classify_sentiment(real_tweets)
    sim_classified = validator.classify_sentiment(sim_tweets)

    # Calculate distributions
    print("\nCalculating metrics...")
    real_dist = validator.calculate_sentiment_distribution(real_classified)
    sim_dist = validator.calculate_sentiment_distribution(sim_classified)

    # Jensen-Shannon Divergence
    jsd = validator.calculate_jsd(real_dist, sim_dist)

    # Thread structure metrics
    real_depth = validator.calculate_max_depth(real_tweets)
    sim_depth = validator.calculate_max_depth(sim_tweets)

    real_engagement = validator.calculate_engagement_ratio(real_tweets)
    sim_engagement = validator.calculate_engagement_ratio(sim_tweets)

    # Keyword extraction
    real_keywords = validator.extract_keywords(real_classified, sentiment_filter='Negative', top_n=5)
    sim_keywords = validator.extract_keywords(sim_classified, sentiment_filter='Negative', top_n=5)

    # Visualization
    viz_path = output_dir / 'sentiment_comparison.png'
    validator.visualize_comparison(real_dist, sim_dist, output_path=viz_path)

    # Generate report
    validator.generate_report(
        real_tweets=real_tweets,
        sim_tweets=sim_tweets,
        jsd=jsd,
        real_depth=real_depth,
        sim_depth=sim_depth,
        real_engagement=real_engagement,
        sim_engagement=sim_engagement,
        real_keywords=real_keywords,
        sim_keywords=sim_keywords,
        real_dist=real_dist,
        sim_dist=sim_dist
    )

    # Save detailed results
    results = {
        'sentiment_distributions': {
            'real': real_dist,
            'simulated': sim_dist
        },
        'similarity_metrics': {
            'jensen_shannon_divergence': float(jsd),
            'sentiment_similarity_percent': float((1 - jsd) * 100)
        },
        'thread_structure': {
            'real': {
                'total_tweets': len(real_tweets),
                'max_depth': real_depth,
                'engagement_ratio': float(real_engagement)
            },
            'simulated': {
                'total_tweets': len(sim_tweets),
                'max_depth': sim_depth,
                'engagement_ratio': float(sim_engagement)
            }
        },
        'negative_keywords': {
            'real': real_keywords,
            'simulated': sim_keywords
        }
    }

    results_path = output_dir / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved detailed results: {results_path}")


if __name__ == '__main__':
    main()
