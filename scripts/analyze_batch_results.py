"""
Analyze Batch Simulation Results

Runs comprehensive 5-dimension validation on all completed threads
and aggregates results into summary statistics.

Usage:
    python scripts/analyze_batch_results.py \
        --sim-dir batch_output_100 \
        --real-dir batch_simulations_100 \
        --output batch_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Model loading (same as validate_comprehensive.py) ────────────────────────

MODELS = {
    'political': {
        'model': 'matous-volf/political-leaning-politics',
        'tokenizer': 'launch/POLITICS',
        'labels': {0: 'Left', 1: 'Center', 2: 'Right'},
    },
    'emotion': {
        'model': 'cardiffnlp/twitter-roberta-base-emotion',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-emotion',
        'labels': {0: 'anger', 1: 'joy', 2: 'optimism', 3: 'sadness'},
    },
    'sentiment': {
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'labels': {0: 'negative', 1: 'neutral', 2: 'positive'},
    },
    'hate': {
        'model': 'cardiffnlp/twitter-roberta-base-hate-latest',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-hate-latest',
        'labels': {0: 'not-hate', 1: 'hate'},
    },
    'offensive': {
        'model': 'cardiffnlp/twitter-roberta-base-offensive',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-offensive',
        'labels': {0: 'not-offensive', 1: 'offensive'},
    },
}


def load_models(device: str) -> dict:
    loaded = {}
    for name, info in MODELS.items():
        print(f"  Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(info['tokenizer'])
        model = AutoModelForSequenceClassification.from_pretrained(info['model'])
        model.to(device).eval()
        loaded[name] = {'tokenizer': tokenizer, 'model': model, 'labels': info['labels']}
    return loaded


def classify_texts(texts: list[str], model_info: dict, device: str) -> list[tuple]:
    """Classify a batch of texts, return list of (label, score)."""
    results = []
    tokenizer = model_info['tokenizer']
    model = model_info['model']
    labels = model_info['labels']

    for text in texts:
        try:
            inputs = tokenizer(str(text)[:512], return_tensors='pt', truncation=True,
                               max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            results.append((labels[top_idx], float(probs[top_idx])))
        except Exception:
            results.append(None)
    return results


def compute_jsd(dist_real: dict, dist_sim: dict, all_labels: list[str]) -> float:
    """Compute Jensen-Shannon Divergence between two label distributions."""
    p = np.array([dist_real.get(l, 0) for l in all_labels], dtype=float)
    q = np.array([dist_sim.get(l, 0) for l in all_labels], dtype=float)
    # Normalize
    if p.sum() > 0:
        p /= p.sum()
    if q.sum() > 0:
        q /= q.sum()
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q) ** 2)  # squared = actual JSD


def load_thread_texts(metadata_path: Path) -> list[str]:
    """Load tweet texts from a thread_metadata.json or simulated_thread_metadata.json."""
    with open(metadata_path) as f:
        data = json.load(f)

    texts = []
    if 'temporal_events' in data:
        for event in data['temporal_events']:
            text = event.get('text', '')
            if text and len(text.strip()) > 5:
                texts.append(text)
    return texts


def validate_single_thread(
    real_texts: list[str],
    sim_texts: list[str],
    models: dict,
    device: str,
) -> dict:
    """
    Run 5-dimension validation on a single thread pair.
    Returns dict with JSD scores and distribution comparisons.
    """
    result = {}

    for model_name in ['political', 'sentiment', 'emotion']:
        model_info = models[model_name]
        all_labels = list(model_info['labels'].values())

        real_classifications = classify_texts(real_texts, model_info, device)
        sim_classifications = classify_texts(sim_texts, model_info, device)

        # Build distributions
        real_dist = {}
        for c in real_classifications:
            if c is not None:
                real_dist[c[0]] = real_dist.get(c[0], 0) + 1
        total_real = sum(real_dist.values()) or 1
        real_dist = {k: v / total_real for k, v in real_dist.items()}

        sim_dist = {}
        for c in sim_classifications:
            if c is not None:
                sim_dist[c[0]] = sim_dist.get(c[0], 0) + 1
        total_sim = sum(sim_dist.values()) or 1
        sim_dist = {k: v / total_sim for k, v in sim_dist.items()}

        jsd = compute_jsd(real_dist, sim_dist, all_labels)
        result[model_name] = {
            'jsd': jsd,
            'similarity': (1 - jsd) * 100,
            'real_dist': real_dist,
            'sim_dist': sim_dist,
        }

    # Hate and offensive scores (continuous)
    for score_model in ['hate', 'offensive']:
        model_info = models[score_model]
        real_cls = classify_texts(real_texts, model_info, device)
        sim_cls = classify_texts(sim_texts, model_info, device)

        def to_score(c, positive_label):
            if c is None:
                return None
            return c[1] if c[0] == positive_label else 1 - c[1]

        pos_label = 'hate' if score_model == 'hate' else 'offensive'
        real_scores = [to_score(c, pos_label) for c in real_cls if c is not None]
        sim_scores = [to_score(c, pos_label) for c in sim_cls if c is not None]

        result[score_model] = {
            'real_mean': float(np.mean(real_scores)) if real_scores else 0,
            'sim_mean': float(np.mean(sim_scores)) if sim_scores else 0,
        }

    # Overall aggression
    real_aggression = result['hate']['real_mean'] + result['offensive']['real_mean']
    sim_aggression = result['hate']['sim_mean'] + result['offensive']['sim_mean']
    result['aggression'] = {
        'real_mean': real_aggression,
        'sim_mean': sim_aggression,
        'gap': abs(real_aggression - sim_aggression),
    }

    # Overall accuracy (weighted: 50% sentiment, 25% political, 25% emotion)
    result['overall_accuracy'] = (
        0.50 * result['sentiment']['similarity'] +
        0.25 * result['political']['similarity'] +
        0.25 * result['emotion']['similarity']
    )

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze batch simulation results')
    parser.add_argument('--sim-dir', type=str, default='batch_output_100',
                        help='Directory with simulation outputs')
    parser.add_argument('--real-dir', type=str, default='batch_simulations_100',
                        help='Directory with prepared thread inputs (real data)')
    parser.add_argument('--output', type=str, default='batch_analysis',
                        help='Output directory for analysis results')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sim_dir = project_root / args.sim_dir
    real_dir = project_root / args.real_dir
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Find completed simulations
    sim_threads = sorted(sim_dir.glob('thread_*'))
    completed = []
    for td in sim_threads:
        sim_meta = td / 'simulation_output' / 'simulated_thread_metadata.json'
        real_meta = real_dir / td.name / 'thread_metadata.json'
        if sim_meta.exists() and real_meta.exists():
            completed.append((td.name, real_meta, sim_meta))

    print(f"\nFound {len(completed)} completed thread pairs to analyze")
    if not completed:
        print("ERROR: No completed simulations found")
        sys.exit(1)

    # Load models
    print("\nLoading classification models...")
    models = load_models(device)

    # Analyze each thread
    all_results = []
    for thread_name, real_meta, sim_meta in tqdm(completed, desc="Analyzing threads"):
        real_texts = load_thread_texts(real_meta)
        sim_texts = load_thread_texts(sim_meta)

        if len(sim_texts) < 2:
            print(f"  {thread_name}: skipped (only {len(sim_texts)} sim texts)")
            continue

        result = validate_single_thread(real_texts, sim_texts, models, device)
        result['thread'] = thread_name
        result['real_tweet_count'] = len(real_texts)
        result['sim_tweet_count'] = len(sim_texts)
        all_results.append(result)

    # ── Aggregate statistics ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"AGGREGATE RESULTS ({len(all_results)} threads)")
    print(f"{'=' * 80}")

    # Build summary dataframe
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'thread': r['thread'],
            'real_tweets': r['real_tweet_count'],
            'sim_tweets': r['sim_tweet_count'],
            'sentiment_jsd': r['sentiment']['jsd'],
            'sentiment_sim': r['sentiment']['similarity'],
            'political_jsd': r['political']['jsd'],
            'political_sim': r['political']['similarity'],
            'emotion_jsd': r['emotion']['jsd'],
            'emotion_sim': r['emotion']['similarity'],
            'aggression_real': r['aggression']['real_mean'],
            'aggression_sim': r['aggression']['sim_mean'],
            'aggression_gap': r['aggression']['gap'],
            'overall_accuracy': r['overall_accuracy'],
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print aggregate stats
    for dim in ['sentiment', 'political', 'emotion']:
        jsd_col = f'{dim}_jsd'
        sim_col = f'{dim}_sim'
        print(f"\n  {dim.upper()}:")
        print(f"    Mean JSD:        {summary_df[jsd_col].mean():.4f} ± {summary_df[jsd_col].std():.4f}")
        print(f"    Mean Similarity: {summary_df[sim_col].mean():.1f}% ± {summary_df[sim_col].std():.1f}%")
        print(f"    Min Similarity:  {summary_df[sim_col].min():.1f}%")
        print(f"    Max Similarity:  {summary_df[sim_col].max():.1f}%")

    print(f"\n  AGGRESSION:")
    print(f"    Mean Real:  {summary_df['aggression_real'].mean():.3f}")
    print(f"    Mean Sim:   {summary_df['aggression_sim'].mean():.3f}")
    print(f"    Mean Gap:   {summary_df['aggression_gap'].mean():.3f}")

    print(f"\n  OVERALL ACCURACY:")
    print(f"    Mean: {summary_df['overall_accuracy'].mean():.1f}% ± {summary_df['overall_accuracy'].std():.1f}%")
    print(f"    Min:  {summary_df['overall_accuracy'].min():.1f}%")
    print(f"    Max:  {summary_df['overall_accuracy'].max():.1f}%")

    # Rating
    mean_acc = summary_df['overall_accuracy'].mean()
    if mean_acc >= 80:
        rating = "EXCELLENT"
    elif mean_acc >= 60:
        rating = "GOOD"
    elif mean_acc >= 40:
        rating = "FAIR"
    else:
        rating = "POOR"
    print(f"\n  Rating: {rating}")

    # ── Save outputs ─────────────────────────────────────────────────────
    # 1. Per-thread CSV
    summary_df.to_csv(output_dir / 'per_thread_results.csv', index=False)
    print(f"\nSaved: {output_dir / 'per_thread_results.csv'}")

    # 2. Aggregate JSON
    aggregate = {
        'n_threads': len(all_results),
        'overall_accuracy': {
            'mean': round(float(summary_df['overall_accuracy'].mean()), 2),
            'std': round(float(summary_df['overall_accuracy'].std()), 2),
            'min': round(float(summary_df['overall_accuracy'].min()), 2),
            'max': round(float(summary_df['overall_accuracy'].max()), 2),
            'rating': rating,
        },
        'sentiment': {
            'mean_jsd': round(float(summary_df['sentiment_jsd'].mean()), 4),
            'std_jsd': round(float(summary_df['sentiment_jsd'].std()), 4),
            'mean_similarity': round(float(summary_df['sentiment_sim'].mean()), 2),
        },
        'political': {
            'mean_jsd': round(float(summary_df['political_jsd'].mean()), 4),
            'std_jsd': round(float(summary_df['political_jsd'].std()), 4),
            'mean_similarity': round(float(summary_df['political_sim'].mean()), 2),
        },
        'emotion': {
            'mean_jsd': round(float(summary_df['emotion_jsd'].mean()), 4),
            'std_jsd': round(float(summary_df['emotion_jsd'].std()), 4),
            'mean_similarity': round(float(summary_df['emotion_sim'].mean()), 2),
        },
        'aggression': {
            'mean_real': round(float(summary_df['aggression_real'].mean()), 4),
            'mean_sim': round(float(summary_df['aggression_sim'].mean()), 4),
            'mean_gap': round(float(summary_df['aggression_gap'].mean()), 4),
        },
    }
    with open(output_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved: {output_dir / 'aggregate_results.json'}")

    # 3. Full per-thread results JSON
    with open(output_dir / 'all_thread_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {output_dir / 'all_thread_results.json'}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
