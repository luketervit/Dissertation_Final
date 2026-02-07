"""
Comprehensive Thread Validation Script

Compares real vs simulated Twitter threads across ALL DNA dimensions:
- Political Leaning (Left/Center/Right)
- Emotion (joy/anger/sadness/optimism)
- Sentiment (positive/neutral/negative)
- Hate Speech (continuous 0-1)
- Offensive Language (continuous 0-1)

Uses Jensen-Shannon Divergence for distribution comparison.

Usage:
    python scripts/validate_comprehensive.py \
        --real batch_output/thread_001/thread_metadata.json \
        --simulated batch_output/thread_001/simulation_output/simulated_thread_metadata.json \
        --output-dir batch_output/thread_001/analysis
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from transformers import pipeline
import torch
from tqdm import tqdm


# Label mapping for political model (same as step1_classify_chunked.py)
POLITICAL_LABEL_MAP = {
    'LABEL_0': 'Left',
    'LABEL_1': 'Center',
    'LABEL_2': 'Right',
}


def load_models(device: str = 'auto') -> dict:
    """Load all 5 classification models."""
    if device == 'auto':
        dev = 0 if torch.cuda.is_available() else -1
    elif device == 'cuda':
        dev = 0
    else:
        dev = -1

    device_name = 'cuda' if dev == 0 else 'cpu'
    print(f"\nUsing device: {device_name}")
    print("Loading 5 classification models...")

    models = {}

    models['political'] = pipeline(
        "text-classification",
        model="matous-volf/political-leaning-politics",
        tokenizer="launch/POLITICS",
        device=dev, truncation=True, max_length=512,
    )
    print("  [1/5] Political leaning model loaded")

    models['emotion'] = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        device=dev, truncation=True, max_length=512,
    )
    print("  [2/5] Emotion model loaded")

    models['sentiment'] = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=dev, truncation=True, max_length=512,
    )
    print("  [3/5] Sentiment model loaded")

    models['hate'] = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-hate-latest",
        device=dev, truncation=True, max_length=512,
    )
    print("  [4/5] Hate speech model loaded")

    models['offensive'] = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-offensive",
        device=dev, truncation=True, max_length=512,
    )
    print("  [5/5] Offensive language model loaded")

    return models


def classify_tweet(text: str, models: dict) -> dict:
    """Run all 5 models on a single tweet. Same logic as step1_classify_chunked.py."""
    text = str(text)[:512]

    p = models['political'](text)[0]
    e = models['emotion'](text)[0]
    sn = models['sentiment'](text)[0]
    h = models['hate'](text)[0]
    o = models['offensive'](text)[0]

    return {
        'political_label': POLITICAL_LABEL_MAP.get(p['label'], p['label']),
        'political_score': p['score'],
        'emotion_label': e['label'],
        'emotion_score': e['score'],
        'sentiment_label': sn['label'],
        'sentiment_score': sn['score'],
        'hate_score': h['score'] if h['label'] == 'HATE' else 1 - h['score'],
        'offensive_score': o['score'] if o['label'] == 'OFFENSIVE' else 1 - o['score'],
    }


def load_thread(json_path: str) -> list[dict]:
    """Load thread from JSON, return list of tweet dicts with 'text' field."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'temporal_events' in data:
        return [
            {'text': event['text'], 'tweet_id': event.get('tweet_id')}
            for event in data['temporal_events']
        ]
    elif isinstance(data, list):
        return [
            {'text': post['text'], 'tweet_id': post.get('post_id', post.get('tweet_id'))}
            for post in data
        ]
    else:
        raise ValueError(f"Unknown JSON format in {json_path}")


def classify_thread(tweets: list[dict], models: dict, label: str) -> list[dict]:
    """Classify all tweets in a thread through all 5 models."""
    results = []
    for tweet in tqdm(tweets, desc=f"Classifying {label}"):
        dna = classify_tweet(tweet['text'], models)
        dna['text'] = tweet['text']
        dna['tweet_id'] = tweet.get('tweet_id')
        results.append(dna)
    return results


def compute_distribution(items: list, key: str) -> dict:
    """Compute percentage distribution for a categorical key."""
    total = len(items)
    if total == 0:
        return {}
    counts = Counter(item[key] for item in items)
    return {label: count / total for label, count in sorted(counts.items())}


def compute_jsd(dist1: dict, dist2: dict) -> float:
    """Compute JSD between two distributions, aligning labels."""
    all_labels = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    p = np.array([dist1.get(l, 0.0) for l in all_labels])
    q = np.array([dist2.get(l, 0.0) for l in all_labels])

    if p.sum() == 0 or q.sum() == 0:
        return 1.0

    jsd = jensenshannon(p, q) ** 2
    return float(jsd) if not np.isnan(jsd) else 1.0


def extract_keywords(tweets: list[dict], top_n: int = 10) -> list[tuple]:
    """Extract top keywords from tweet texts."""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about', 'as',
        'from', 'up', 'out', 'if', 'by', 'my', 'your', 'our', 'like', 'also',
        'right', 'left', 'one', 'let', 'don', 'get', 'got', 'much', 'well',
        'even', 'still', 'now', 'back', 'then', 'here', 'there', 'them',
        'him', 'her', 'its', 'his', 'their', 'been', 'way', 'going', 'know',
        'think', 'make', 'said', 'say', 'says', 'want', 'see', 'come',
    }
    words = []
    for tweet in tweets:
        text = tweet['text'].lower()
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        word_list = re.findall(r'\b[a-z]{3,}\b', text)
        words.extend(w for w in word_list if w not in stopwords)
    return Counter(words).most_common(top_n)


def print_distribution_comparison(
    name: str, real_dist: dict, sim_dist: dict, jsd: float
) -> None:
    """Print a formatted distribution comparison table."""
    similarity = (1 - jsd) * 100
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  JSD: {jsd:.4f} | Similarity: {similarity:.1f}%")
    print(f"{'─'*70}")
    all_labels = sorted(set(list(real_dist.keys()) + list(sim_dist.keys())))
    print(f"  {'Label':<20} {'Real':>10} {'Simulated':>10} {'Delta':>10}")
    print(f"  {'─'*50}")
    for label in all_labels:
        r = real_dist.get(label, 0) * 100
        s = sim_dist.get(label, 0) * 100
        delta = s - r
        arrow = "+" if delta > 0 else ""
        print(f"  {label:<20} {r:>9.1f}% {s:>9.1f}% {arrow}{delta:>8.1f}%")


def create_visualisation(results: dict, output_path: str) -> None:
    """Create multi-panel comparison chart."""
    dimensions = [
        ('political', 'Political Leaning'),
        ('emotion', 'Emotion'),
        ('sentiment', 'Sentiment'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Comprehensive Validation: Real vs Simulated Thread',
        fontsize=16, fontweight='bold', y=0.98,
    )

    for idx, (key, title) in enumerate(dimensions):
        ax = axes[idx // 2][idx % 2]
        real_dist = results['distributions'][key]['real']
        sim_dist = results['distributions'][key]['simulated']
        jsd = results['distributions'][key]['jsd']

        all_labels = sorted(set(list(real_dist.keys()) + list(sim_dist.keys())))
        x = np.arange(len(all_labels))
        width = 0.35

        real_vals = [real_dist.get(l, 0) for l in all_labels]
        sim_vals = [sim_dist.get(l, 0) for l in all_labels]

        bars1 = ax.bar(x - width / 2, real_vals, width, label='Real', alpha=0.8, color='#2196F3')
        bars2 = ax.bar(x + width / 2, sim_vals, width, label='Simulated', alpha=0.8, color='#FF9800')

        ax.set_title(f'{title} (JSD: {jsd:.4f}, Sim: {(1-jsd)*100:.1f}%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Proportion')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.0%}', ha='center', va='bottom', fontsize=7,
                    )

    # Bottom-right: Hate + Offensive scores comparison
    ax = axes[1][1]
    cont = results['continuous_scores']
    metrics = ['hate_score', 'offensive_score', 'aggression']
    x = np.arange(len(metrics))
    width = 0.35

    real_means = [cont[m]['real_mean'] for m in metrics]
    sim_means = [cont[m]['sim_mean'] for m in metrics]

    ax.bar(x - width / 2, real_means, width, label='Real', alpha=0.8, color='#2196F3')
    ax.bar(x + width / 2, sim_means, width, label='Simulated', alpha=0.8, color='#FF9800')
    ax.set_title('Continuous Scores (Mean)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hate Score', 'Offensive Score', 'Aggression\n(hate+offensive)'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean Score')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for i, (rv, sv) in enumerate(zip(real_means, sim_means)):
        ax.text(i - width / 2, rv + 0.01, f'{rv:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, sv + 0.01, f'{sv:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved visualisation: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive thread validation across all DNA dimensions")
    parser.add_argument('--real', type=str, required=True, help='Path to real thread JSON')
    parser.add_argument('--simulated', type=str, required=True, help='Path to simulated thread JSON')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────
    models = load_models(args.device)

    # ── Load threads ─────────────────────────────────────────────────────
    print("\nLoading threads...")
    real_tweets = load_thread(args.real)
    sim_tweets = load_thread(args.simulated)
    print(f"  Real thread: {len(real_tweets)} tweets")
    print(f"  Simulated thread: {len(sim_tweets)} tweets")

    # ── Classify all tweets ──────────────────────────────────────────────
    print()
    real_classified = classify_thread(real_tweets, models, "real thread")
    sim_classified = classify_thread(sim_tweets, models, "simulated thread")

    # ── Compute distributions ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE VALIDATION REPORT")
    print("=" * 70)
    print(f"  Real tweets: {len(real_classified)}  |  Simulated tweets: {len(sim_classified)}")

    results = {'distributions': {}, 'continuous_scores': {}, 'keywords': {}}

    # Categorical dimensions
    for key, name in [
        ('political_label', 'Political Leaning'),
        ('emotion_label', 'Emotion'),
        ('sentiment_label', 'Sentiment'),
    ]:
        dim_key = key.replace('_label', '')
        real_dist = compute_distribution(real_classified, key)
        sim_dist = compute_distribution(sim_classified, key)
        jsd = compute_jsd(real_dist, sim_dist)

        results['distributions'][dim_key] = {
            'real': real_dist,
            'simulated': sim_dist,
            'jsd': jsd,
            'similarity_percent': (1 - jsd) * 100,
        }

        print_distribution_comparison(name, real_dist, sim_dist, jsd)

    # Continuous scores
    for score_key in ['hate_score', 'offensive_score']:
        real_vals = [t[score_key] for t in real_classified]
        sim_vals = [t[score_key] for t in sim_classified]
        results['continuous_scores'][score_key] = {
            'real_mean': float(np.mean(real_vals)),
            'sim_mean': float(np.mean(sim_vals)),
            'real_std': float(np.std(real_vals)),
            'sim_std': float(np.std(sim_vals)),
            'real_pct_above_05': float(np.mean(np.array(real_vals) > 0.5)),
            'sim_pct_above_05': float(np.mean(np.array(sim_vals) > 0.5)),
        }

    # Aggression (hate + offensive combined)
    real_agg = [t['hate_score'] + t['offensive_score'] for t in real_classified]
    sim_agg = [t['hate_score'] + t['offensive_score'] for t in sim_classified]
    results['continuous_scores']['aggression'] = {
        'real_mean': float(np.mean(real_agg)),
        'sim_mean': float(np.mean(sim_agg)),
        'real_std': float(np.std(real_agg)),
        'sim_std': float(np.std(sim_agg)),
        'real_pct_above_05': float(np.mean(np.array(real_agg) > 0.5)),
        'sim_pct_above_05': float(np.mean(np.array(sim_agg) > 0.5)),
    }

    # Print continuous scores
    print(f"\n{'─'*70}")
    print("  Continuous Scores")
    print(f"{'─'*70}")
    print(f"  {'Metric':<20} {'Real Mean':>12} {'Sim Mean':>12} {'Real >0.5':>12} {'Sim >0.5':>12}")
    print(f"  {'─'*60}")
    for key in ['hate_score', 'offensive_score', 'aggression']:
        s = results['continuous_scores'][key]
        name = key.replace('_', ' ').title()
        print(
            f"  {name:<20} {s['real_mean']:>11.3f} {s['sim_mean']:>11.3f}"
            f" {s['real_pct_above_05']:>11.1%} {s['sim_pct_above_05']:>11.1%}"
        )

    # Keywords by political leaning
    print(f"\n{'─'*70}")
    print("  Top Keywords by Political Leaning")
    print(f"{'─'*70}")
    for pol_label in ['Left', 'Right']:
        real_pol = [t for t in real_classified if t['political_label'] == pol_label]
        sim_pol = [t for t in sim_classified if t['political_label'] == pol_label]
        real_kw = extract_keywords(real_pol, top_n=8)
        sim_kw = extract_keywords(sim_pol, top_n=8)

        results['keywords'][pol_label] = {
            'real': real_kw,
            'simulated': sim_kw,
        }

        print(f"\n  {pol_label} tweets:")
        print(f"    {'Real':<40} {'Simulated':<40}")
        print(f"    {'─'*38} {'─'*38}")
        max_rows = max(len(real_kw), len(sim_kw))
        for i in range(max_rows):
            rk = f"{real_kw[i][0]} ({real_kw[i][1]})" if i < len(real_kw) else ""
            sk = f"{sim_kw[i][0]} ({sim_kw[i][1]})" if i < len(sim_kw) else ""
            print(f"    {rk:<40} {sk:<40}")

    # Overall weighted accuracy
    pol_sim = results['distributions']['political']['similarity_percent']
    emo_sim = results['distributions']['emotion']['similarity_percent']
    sent_sim = results['distributions']['sentiment']['similarity_percent']

    overall = pol_sim * 0.30 + emo_sim * 0.20 + sent_sim * 0.30 + (
        (1 - abs(results['continuous_scores']['aggression']['real_mean']
                 - results['continuous_scores']['aggression']['sim_mean'])) * 100 * 0.20
    )
    results['overall_accuracy'] = float(overall)

    print(f"\n{'='*70}")
    print("  OVERALL ACCURACY")
    print(f"{'='*70}")
    print(f"  Political similarity (30%):  {pol_sim:.1f}%")
    print(f"  Emotion similarity (20%):    {emo_sim:.1f}%")
    print(f"  Sentiment similarity (30%):  {sent_sim:.1f}%")
    print(f"  Aggression similarity (20%): "
          f"{(1 - abs(results['continuous_scores']['aggression']['real_mean'] - results['continuous_scores']['aggression']['sim_mean'])) * 100:.1f}%")
    print(f"\n  OVERALL: {overall:.1f}%")

    if overall >= 80:
        print("  Assessment: EXCELLENT")
    elif overall >= 60:
        print("  Assessment: GOOD")
    elif overall >= 40:
        print("  Assessment: FAIR")
    else:
        print("  Assessment: POOR")
    print("=" * 70)

    # ── Visualisation ────────────────────────────────────────────────────
    viz_path = output_dir / 'comprehensive_validation.png'
    create_visualisation(results, str(viz_path))

    # ── Save JSON results ────────────────────────────────────────────────
    results_path = output_dir / 'comprehensive_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results: {results_path}")


if __name__ == '__main__':
    main()
