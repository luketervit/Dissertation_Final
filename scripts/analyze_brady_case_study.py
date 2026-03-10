"""
Analyze paired Brady case-study simulation outputs.

Expected input:
  case_study/brady_mec/manifests/thread_manifest.csv
  case_study/brady_mec/results/<batch_output_model>/thread_XXX/simulation_output/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def load_simulated_posts(sim_meta_path: Path) -> list[dict]:
    with open(sim_meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("temporal_events", [])
    posts = [e for e in events if not e.get("is_root", False)]
    return posts


def structural_metrics(sim_meta_path: Path) -> dict:
    with open(sim_meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    info = data.get("simulation_info", {})
    timeline = data.get("timeline", {})
    n_total = int(info.get("total_posts", len(data.get("temporal_events", []))))
    n_replies = max(0, n_total - 1)
    return {
        "n_replies": n_replies,
        "max_depth": int(info.get("max_depth", 0)),
        "rounds": int(info.get("rounds", 0)),
        "duration_seconds": int(timeline.get("duration_seconds", 0)),
    }


def classify_posts(posts: list[dict], models: dict, classify_tweet_full) -> pd.DataFrame:
    rows = []
    for p in posts:
        text = str(p.get("text", ""))
        if not text.strip():
            continue
        dna = classify_tweet_full(text, models)
        dna["text"] = text
        rows.append(dna)
    return pd.DataFrame(rows)


def classification_metrics(classified_df: pd.DataFrame) -> dict:
    if len(classified_df) == 0:
        return {
            "right_share": np.nan,
            "left_share": np.nan,
            "center_share": np.nan,
            "negative_share": np.nan,
            "neutral_share": np.nan,
            "positive_share": np.nan,
            "anger_share": np.nan,
            "mean_aggression": np.nan,
            "mean_sentiment_continuous": np.nan,
        }

    pol = classified_df["political_label"].value_counts(normalize=True).to_dict()
    sent = classified_df["sentiment_label"].value_counts(normalize=True).to_dict()
    emo = classified_df["emotion_label"].value_counts(normalize=True).to_dict()
    aggression = (
        classified_df["hate_score"].fillna(0) + classified_df["offensive_score"].fillna(0)
    )
    return {
        "right_share": float(pol.get("Right", 0.0)),
        "left_share": float(pol.get("Left", 0.0)),
        "center_share": float(pol.get("Center", 0.0)),
        "negative_share": float(sent.get("negative", 0.0)),
        "neutral_share": float(sent.get("neutral", 0.0)),
        "positive_share": float(sent.get("positive", 0.0)),
        "anger_share": float(emo.get("anger", 0.0)),
        "mean_aggression": float(aggression.mean()),
        "mean_sentiment_continuous": float(classified_df["sentiment_continuous"].mean()),
    }


def paired_deltas(thread_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in thread_metrics_df.columns
        if c not in {"family_id", "thread_id", "condition", "topic"}
    ]
    controls = (
        thread_metrics_df[thread_metrics_df["condition"] == "control"]
        .set_index("family_id")[metric_cols]
        .add_prefix("control_")
    )
    edited_df = thread_metrics_df[thread_metrics_df["condition"] != "control"].copy()
    outputs = []
    for condition in sorted(edited_df["condition"].unique()):
        edited = (
            edited_df[edited_df["condition"] == condition]
            .set_index("family_id")[metric_cols]
            .add_prefix("edited_")
        )
        out = controls.join(edited, how="inner").reset_index()
        out["edited_condition"] = condition
        for col in metric_cols:
            c_col = f"control_{col}"
            e_col = f"edited_{col}"
            out[f"delta_{col}"] = out[e_col] - out[c_col]
        outputs.append(out)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def summarize_deltas(delta_df: pd.DataFrame) -> dict:
    delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
    summary = {}
    for col in delta_cols:
        vals = delta_df[col].dropna()
        summary[col] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "median": float(vals.median()) if len(vals) else np.nan,
            "std": float(vals.std()) if len(vals) else np.nan,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Brady case-study outputs.")
    parser.add_argument(
        "--case-study-root",
        type=Path,
        default=Path("case_study/brady_mec"),
        help="Case-study root directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("case_study/brady_mec/results/batch_output_qwen"),
        help="Simulation results directory (thread_*/simulation_output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("case_study/brady_mec/analysis/qwen"),
        help="Where analysis outputs are written.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run 5-model tweet classification on generated replies.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for classification models when --classify is set.",
    )
    args = parser.parse_args()

    case_root = (PROJECT_ROOT / args.case_study_root).resolve()
    results_root = (PROJECT_ROOT / args.results_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = case_root / "manifests" / "thread_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"thread manifest not found: {manifest_path}")

    thread_manifest = pd.read_csv(manifest_path)

    models = None
    classify_tweet_full = None
    if args.classify:
        from validate_comprehensive import load_models
        from validate_batch_100 import classify_tweet_full as _classify_tweet_full

        models = load_models(device=args.device)
        classify_tweet_full = _classify_tweet_full

    rows = []
    classified_rows = []
    for _, row in thread_manifest.iterrows():
        thread_id = str(row["thread_id"])
        sim_meta = results_root / thread_id / "simulation_output" / "simulated_thread_metadata.json"
        if not sim_meta.exists():
            continue

        metrics = structural_metrics(sim_meta)
        record = {
            "family_id": row["family_id"],
            "thread_id": thread_id,
            "condition": row["condition"],
            "topic": row["topic"],
            **metrics,
        }

        if args.classify:
            posts = load_simulated_posts(sim_meta)
            cdf = classify_posts(posts, models, classify_tweet_full)
            cmetrics = classification_metrics(cdf)
            record.update(cmetrics)
            if len(cdf):
                cdf["thread_id"] = thread_id
                cdf["family_id"] = row["family_id"]
                cdf["condition"] = row["condition"]
                classified_rows.append(cdf)

        rows.append(record)

    if not rows:
        raise ValueError(f"No completed simulations found under {results_root}")

    thread_metrics_df = pd.DataFrame(rows)
    thread_metrics_df.to_csv(out_dir / "thread_metrics.csv", index=False)

    delta_df = paired_deltas(thread_metrics_df)
    delta_df.to_csv(out_dir / "paired_deltas.csv", index=False)

    n_pairs = int(delta_df["family_id"].nunique()) if "family_id" in delta_df.columns else 0
    summary = {
        "n_threads_with_outputs": int(len(thread_metrics_df)),
        "n_families_with_pairs": n_pairs,
        "results_root": str(results_root),
        "classify": bool(args.classify),
        "delta_summary": summarize_deltas(delta_df),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if classified_rows:
        pd.concat(classified_rows, ignore_index=True).to_csv(
            out_dir / "classified_generated_posts.csv", index=False
        )

    print(f"Analysis complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
