"""
Ablation Sweep: 20 parameter configs × 10 threads = 200 simulations.

All configurations run in 0-shot mode (no few-shot grounding).
- 15 single-parameter ablations (change 1 param from 0-shot baseline)
- 5 hail-mary multi-parameter combos

Usage:
    python scripts/run_ablation_sweep.py                     # Run all 20 configs × 10 threads
    python scripts/run_ablation_sweep.py --threads 1 2 3     # Custom threads
    python scripts/run_ablation_sweep.py --configs temp_07    # Specific configs
    python scripts/run_ablation_sweep.py --list               # List all configs
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------
# Each config specifies ONLY the overrides from the 0-shot baseline.
# The baseline is: Step 10 "new params" with use_few_shot=false.

CONFIGS: dict[str, dict] = {
    # ---- 0-shot baseline (reference, no overrides) ----
    "0shot_baseline": {},

    # ---- 15 single-parameter ablations ----
    "temp_07": {"temperature": 0.7},
    "temp_12": {"temperature": 1.2},
    "ctx_3": {"context_window": 3},
    "ctx_15": {"context_window": 15},
    "no_dynamic_scaling": {"dynamic_scaling": False},
    "old_rule6": {
        "rule6_text": (
            "Match the hostility level of the thread. "
            "Political Twitter is aggressive."
        ),
    },
    "old_4tier": {"aggression_tiers": "old"},
    "reply_prob_10": {"base_prob": 0.10},
    "reply_prob_02": {"base_prob": 0.02},
    "controversy_5": {"controversy_weight": 5.0},
    "controversy_1": {"controversy_weight": 1.0},
    "no_vocab": {"vocab_enabled": False},
    "repeat_penalty_13": {"repeat_penalty": 1.3},
    "max_tokens_80": {"max_tokens": 80},
    "rounds_20": {"max_rounds": 20},

    # ---- 5 hail-mary combos ----
    "hm_aggressive_realist": {
        "temperature": 1.1,
        "rule6_text": (
            "Match the hostility level of the thread. "
            "Political Twitter is aggressive."
        ),
        "aggression_tiers": "old",
        "controversy_weight": 5.0,
    },
    "hm_minimal_prompt": {
        "vocab_enabled": False,
        "dynamic_scaling": False,
        "context_window": 3,
        "max_tokens": 80,
    },
    "hm_high_engagement": {
        "base_prob": 0.12,
        "aggression_boost_factor": 0.25,
        "max_reply_prob": 0.40,
        "controversy_weight": 4.0,
        "max_rounds": 15,
    },
    "hm_calm_deep": {
        "temperature": 0.7,
        "base_prob": 0.03,
        "context_window": 15,
        "repeat_penalty": 1.3,
        "max_rounds": 20,
    },
    "hm_chaos": {
        "temperature": 1.4,
        "aggression_tiers": "old",
        "controversy_weight": 5.0,
        "base_prob": 0.08,
        "dynamic_scaling": False,
    },
}

# Active overrides — set before each config run, read by patched functions.
ACTIVE: dict = {}

# Default thread selection: spread across the 100 for diversity.
DEFAULT_THREADS = [1, 5, 10, 15, 20, 30, 50, 60, 75, 90]

# ---------------------------------------------------------------------------
# Original function storage
# ---------------------------------------------------------------------------
_ORIGINALS: dict = {}


def save_originals() -> None:
    """Save references to the original (unpatched) functions once."""
    if _ORIGINALS:
        return  # already saved

    import sim.llm_generator as gen
    import sim.thread_simulation as tsim

    _ORIGINALS["aggression_tone"] = gen._aggression_tone
    _ORIGINALS["build_system_prompt"] = gen.LLMGenerator._build_system_prompt
    _ORIGINALS["build_user_prompt"] = gen.LLMGenerator._build_user_prompt
    _ORIGINALS["generate_ollama"] = gen.LLMGenerator._generate_ollama
    _ORIGINALS["agent_step"] = tsim.ThreadAgent.step
    _ORIGINALS["select_reply_target"] = tsim.ThreadAgent._select_reply_target


def restore_originals() -> None:
    """Restore all functions to their original (Step 10) implementations."""
    import sim.llm_generator as gen
    import sim.thread_simulation as tsim

    gen._aggression_tone = _ORIGINALS["aggression_tone"]
    gen.LLMGenerator._build_system_prompt = _ORIGINALS["build_system_prompt"]
    gen.LLMGenerator._build_user_prompt = _ORIGINALS["build_user_prompt"]
    gen.LLMGenerator._generate_ollama = _ORIGINALS["generate_ollama"]
    tsim.ThreadAgent.step = _ORIGINALS["agent_step"]
    tsim.ThreadAgent._select_reply_target = _ORIGINALS["select_reply_target"]


# ---------------------------------------------------------------------------
# Old 4-tier aggression function (pre-Step-10)
# ---------------------------------------------------------------------------

def _old_aggression_tone(aggression: float) -> str:
    if aggression >= 0.7:
        return (
            "extremely hostile and confrontational — you insult "
            "opponents directly, use profanity, and show zero respect"
        )
    if aggression >= 0.5:
        return (
            "aggressive and combative — you attack opponents' "
            "arguments harshly and dismiss them with contempt"
        )
    if aggression >= 0.3:
        return (
            "assertive and sharp — you push back firmly, use "
            "pointed sarcasm, and don't hold back"
        )
    return "pointed but measured — you disagree firmly and use dry wit"


# ---------------------------------------------------------------------------
# Parametric replacement functions (read from ACTIVE dict)
# ---------------------------------------------------------------------------

def _patched_build_system_prompt(
    self, agent_persona: dict, thread_mean_aggression: float = 0.3
) -> str:
    """Replacement _build_system_prompt that reads overrides from ACTIVE."""
    import sim.llm_generator as gen

    political = agent_persona.get("political_label", "Center")
    aggression = agent_persona.get("aggression", 0.3)
    emotion = agent_persona.get("emotion", "anger")

    # Political orientation
    if political == "Left":
        ideology = (
            "You hold progressive political views. You support "
            "social justice, government accountability, and are "
            "critical of conservative/Republican policies."
        )
        if ACTIVE.get("vocab_enabled", True):
            vocab = ", ".join(gen.LEFT_VOCAB[:8])
        else:
            vocab = ""
    elif political == "Right":
        ideology = (
            "You hold conservative political views. You support "
            "limited government, traditional values, and are "
            "critical of liberal/Democratic policies."
        )
        if ACTIVE.get("vocab_enabled", True):
            vocab = ", ".join(gen.RIGHT_VOCAB[:8])
        else:
            vocab = ""
    else:
        ideology = (
            "You hold moderate political views. You criticise "
            "extremes on both sides and value pragmatic solutions."
        )
        vocab = "both sides, common sense, pragmatic, compromise"

    # Aggression tone
    use_old_tiers = ACTIVE.get("aggression_tiers") == "old"
    tone_fn = _old_aggression_tone if use_old_tiers else gen._aggression_tone

    if ACTIVE.get("dynamic_scaling", True):
        # Dynamic scaling relative to thread mean
        relative_agg = aggression - thread_mean_aggression
        if relative_agg > 0.2:
            tone = tone_fn(aggression)
        elif relative_agg > -0.1:
            tone = tone_fn(max(0.0, aggression - 0.15))
        else:
            tone = tone_fn(max(0.0, aggression - 0.25))
    else:
        # Absolute thresholds (no scaling)
        tone = tone_fn(aggression)

    emotion_desc = gen.EMOTION_BEHAVIOUR.get(
        emotion.lower(),
        "passionate and opinionated, engaging forcefully in debate",
    )

    # Rule 6 text
    rule6 = ACTIVE.get(
        "rule6_text",
        "Not every reply is an attack. Sometimes you agree with someone, "
        "crack a joke, share a fact, or express genuine concern. "
        "Vary your tone naturally.",
    )

    # Vocabulary line
    vocab_line = f"\nVOCABULARY: Use words like: {vocab}" if vocab else ""

    return f"""You write short tweets in a political argument on Twitter/X.

POLITICAL VIEWS: {ideology}
TONE: You are {tone}.
EMOTIONAL STYLE: You are {emotion_desc}.{vocab_line}

RULES — follow these EXACTLY:
1. Write ONLY the tweet text. No quotation marks around it.
2. NEVER reveal you are an AI, a bot, a language model, or playing a role.
3. NEVER start with "As a..." or mention your political leaning explicitly.
4. NEVER use words like "dialogue", "unity", "together", "both sides" unless you are Center.
5. Be specific — reference the topic being discussed, attack specific policies or people.
6. {rule6}
7. Maximum 280 characters. No hashtags unless relevant."""


def _patched_build_user_prompt(
    self,
    target_post: dict,
    thread_context: list,
    few_shot_examples: list[str] | None = None,
) -> str:
    """Replacement _build_user_prompt that reads context_window from ACTIVE."""
    ctx_size = ACTIVE.get("context_window", 8)
    parts: list[str] = []

    # Few-shot grounding — always disabled for ablation sweep
    # (controlled via config.yaml use_few_shot=false, but also skip here)
    if few_shot_examples:
        example_lines = "\n".join(f'- "{ex}"' for ex in few_shot_examples)
        parts.append(
            "Here's how people are actually talking in this thread:\n"
            f"{example_lines}\n\n"
            "Match this tone and style. Some tweets attack, some "
            "agree, some joke."
        )

    # Context window (configurable size)
    context_lines = []
    for p in thread_context[-ctx_size:]:
        text = p.get("text", "")
        if text:
            context_lines.append(f'- "{text}"')
    context_str = "\n".join(context_lines) if context_lines else "(none)"
    parts.append(f"Recent posts in the thread:\n{context_str}")

    target_text = target_post.get("text", "")
    parts.append(
        f'You are replying to this tweet:\n"{target_text}"\n\n'
        "Write your reply tweet:"
    )
    return "\n\n".join(parts)


def _patched_generate_ollama(
    self, system_prompt: str, user_prompt: str
) -> str:
    """Replacement _generate_ollama that reads repeat_penalty from ACTIVE."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

    rp = ACTIVE.get("repeat_penalty", 1.1)

    def _do_request() -> str:
        resp = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 2048,
                    "repeat_penalty": rp,
                },
            },
            timeout=(30, 120),
        )
        if resp.status_code != 200:
            return "[Error generating response]"
        message = resp.json().get("message", {})
        return message.get("content", "").strip()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_request)
            result = future.result(timeout=90)
            return result if result else "[Error generating response]"
    except FutTimeout:
        print("Ollama 90s hard timeout — skipping agent")
        return "[Timeout]"
    except Exception as e:
        print(f"Ollama error: {e}")
        return "[Error generating response]"


def _patched_agent_step(self):
    """Replacement ThreadAgent.step() that reads reply prob from ACTIVE."""
    base_prob = ACTIVE.get("base_prob", 0.05)
    boost_factor = ACTIVE.get("aggression_boost_factor", 0.15)
    max_prob = ACTIVE.get("max_reply_prob", 0.25)

    aggression_boost = self.aggression * boost_factor
    reply_prob = min(max_prob, base_prob + aggression_boost)

    if random.random() > reply_prob:
        return

    thread = self.model.thread_history
    if len(thread) == 0:
        return

    target_post = self._select_reply_target(thread)
    if target_post is None:
        return

    reply_text = self._generate_reply(target_post)
    self.pending_reply = reply_text
    self.reply_target = target_post["post_id"]


def _patched_select_reply_target(self, thread):
    """Replacement _select_reply_target with configurable weights."""
    if len(thread) == 0:
        return None

    controversy_w = ACTIVE.get("controversy_weight", 2.5)
    avoidance_w = ACTIVE.get("avoidance_weight", 0.3)

    current_round = self.model.current_round
    recent_posts = [
        p for p in thread
        if p["round"] >= current_round - 1 or p in thread[-50:]
    ]
    if not recent_posts:
        recent_posts = thread[-20:]

    weights = []
    for post in recent_posts:
        weight = 1.0

        if post["round"] == current_round:
            weight *= 3.0

        political_distance = abs(
            self._label_to_value(self.political_label)
            - self._label_to_value(post.get("political_label", "Center"))
        )

        if self.aggression > 0.5 and political_distance > 0.5:
            weight *= controversy_w
        elif self.aggression < 0.3 and political_distance > 0.6:
            weight *= avoidance_w
        else:
            weight *= 1 + political_distance

        if post["depth"] > 5:
            weight *= 0.5

        weights.append(weight)

    total = sum(weights)
    if total == 0:
        return random.choice(recent_posts)

    probs = [w / total for w in weights]
    return np.random.choice(recent_posts, p=probs)


# ---------------------------------------------------------------------------
# Apply / reset configuration
# ---------------------------------------------------------------------------

def apply_config(overrides: dict) -> None:
    """Set ACTIVE overrides and install monkey-patches."""
    import sim.llm_generator as gen
    import sim.thread_simulation as tsim

    ACTIVE.clear()
    ACTIVE.update(overrides)

    # Always patch to parametric versions that read ACTIVE
    gen.LLMGenerator._build_system_prompt = _patched_build_system_prompt
    gen.LLMGenerator._build_user_prompt = _patched_build_user_prompt
    gen.LLMGenerator._generate_ollama = _patched_generate_ollama
    tsim.ThreadAgent.step = _patched_agent_step
    tsim.ThreadAgent._select_reply_target = _patched_select_reply_target

    # _aggression_tone only needs replacing if using old tiers
    if overrides.get("aggression_tiers") == "old":
        gen._aggression_tone = _old_aggression_tone


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------

def write_temp_config(
    thread_dir: Path,
    overrides: dict,
    output_dir: Path,
    *,
    provider_override: str | None = None,
) -> Path:
    """
    Read thread's config.yaml, apply overrides, write to output dir.

    Always sets use_few_shot=false for 0-shot ablation.
    Rewrites paths to point at the actual *thread_dir* so configs
    created on one machine (e.g. GCP) work on another (local Mac).
    """
    src_config_path = thread_dir / "config.yaml"
    with open(src_config_path) as f:
        config = yaml.safe_load(f)

    # --- Fix paths to match current machine ---
    config["paths"]["thread_metadata"] = str(
        thread_dir / "thread_metadata.json"
    )
    config["paths"]["agents_for_thread"] = str(
        thread_dir / "agents_for_thread.csv"
    )

    # Force 0-shot
    config.setdefault("simulation", {})["use_few_shot"] = False

    # Apply YAML-level overrides
    if "temperature" in overrides:
        config["llm"]["temperature"] = overrides["temperature"]
    if "max_tokens" in overrides:
        config["llm"]["max_tokens"] = overrides["max_tokens"]
    if "max_rounds" in overrides:
        config["simulation"]["max_rounds"] = overrides["max_rounds"]

    # Optional provider override (e.g. --dry-run uses "mock")
    if provider_override:
        config["llm"]["provider"] = provider_override

    # Write to output dir so each run has its own config
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "config.yaml"
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return dest


# ---------------------------------------------------------------------------
# Simulation runner (adapted from run_dual_batch.py)
# ---------------------------------------------------------------------------

def run_single_thread(
    thread_dir: Path,
    output_dir: Path,
    config_path: Path,
    max_rounds_override: int | None = None,
) -> dict:
    """Run one simulation and export results."""
    from sim.thread_simulation import ThreadModel

    sim_output = output_dir / "simulation_output"
    sim_output.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        model = ThreadModel(config_path=str(config_path))

        with open(config_path) as f:
            config = yaml.safe_load(f)
        max_rounds = (
            max_rounds_override
            or config.get("simulation", {}).get("max_rounds", 10)
        )

        model.run(max_rounds=max_rounds)
        model.export_results(output_dir=str(sim_output))

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "total_posts": len(model.thread_history),
            "elapsed_seconds": round(elapsed, 1),
            "agents": len(model.agent_list),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def save_progress(
    progress_path: Path,
    results: list[dict],
    total: int,
    start_time: float,
) -> None:
    completed = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped_done")

    progress = {
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "results": results,
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation sweep: 20 configs × 10 threads (0-shot)"
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=DEFAULT_THREADS,
        help="Thread numbers to run (default: %(default)s)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Run only these config names (default: all 20)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="batch_simulations_reconstructed",
        help="Input directory with prepared threads",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all config names and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate everything without Ollama: rewrites configs, "
            "patches functions, runs 1-round mock simulations, "
            "then exits. Use locally before deploying to GCP."
        ),
    )
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name, overrides in CONFIGS.items():
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
            print(f"  {name:30s} {desc}")
        return

    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    ablation_root = project_root / "ablation_output"

    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} not found.")
        print("Run scripts/reconstruct_batch_inputs.py first.")
        sys.exit(1)

    # Validate thread dirs exist
    thread_dirs: list[Path] = []
    for t in args.threads:
        td = input_dir / f"thread_{t:03d}"
        if td.exists():
            thread_dirs.append(td)
        else:
            print(f"WARNING: {td} not found, skipping")

    if not thread_dirs:
        print("ERROR: No valid thread directories found.")
        sys.exit(1)

    # Filter configs
    if args.configs:
        configs_to_run = {
            k: v for k, v in CONFIGS.items() if k in args.configs
        }
        unknown = set(args.configs) - set(CONFIGS.keys())
        if unknown:
            print(f"WARNING: Unknown configs ignored: {unknown}")
    else:
        configs_to_run = CONFIGS

    # Save originals
    save_originals()

    total_sims = len(configs_to_run) * len(thread_dirs)
    all_results: list[dict] = []
    start_time = time.time()
    progress_path = ablation_root / "sweep_progress.json"
    ablation_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n*** DRY-RUN MODE: using mock LLM, 1 round per sim ***\n")

    # Save manifest
    manifest = {
        "configs": {
            name: overrides for name, overrides in configs_to_run.items()
        },
        "threads": [td.name for td in thread_dirs],
        "total_simulations": total_sims,
        "all_0shot": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ablation_root / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sim_count = 0
    for config_name, overrides in configs_to_run.items():
        print(f"\n{'=' * 80}")
        desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
        print(f"CONFIG: {config_name}  [{desc}]")
        print(f"{'=' * 80}")

        # Reset to original then apply this config's patches
        restore_originals()
        apply_config(overrides)

        for thread_dir in thread_dirs:
            thread_name = thread_dir.name
            output_dir = ablation_root / config_name / thread_name
            sim_count += 1

            # Skip if already completed
            done_marker = (
                output_dir / "simulation_output"
                / "simulated_thread_metadata.json"
            )
            if done_marker.exists():
                print(
                    f"\n[{sim_count}/{total_sims}] "
                    f"{config_name}/{thread_name}: DONE, skipping"
                )
                all_results.append({
                    "config": config_name,
                    "thread": thread_name,
                    "status": "skipped_done",
                })
                continue

            print(f"\n{'─' * 60}")
            print(
                f"[{sim_count}/{total_sims}] "
                f"{config_name}/{thread_name}..."
            )

            # Write temp config with YAML-level overrides
            config_path = write_temp_config(
                thread_dir,
                overrides,
                output_dir,
                provider_override="mock" if args.dry_run else None,
            )

            # In dry-run, force 1 round regardless of config
            result = run_single_thread(
                thread_dir,
                output_dir,
                config_path,
                max_rounds_override=1 if args.dry_run else None,
            )
            result["config"] = config_name
            result["thread"] = thread_name
            all_results.append(result)

            if result["status"] == "success":
                print(
                    f"  OK: {result['total_posts']} posts, "
                    f"{result['agents']} agents, "
                    f"{result['elapsed_seconds']}s"
                )
            else:
                print(f"  FAILED: {result.get('error', 'unknown')}")

            # Save progress
            save_progress(progress_path, all_results, total_sims, start_time)

    # Restore originals when done
    restore_originals()

    # Final summary
    elapsed = time.time() - start_time
    successes = sum(1 for r in all_results if r["status"] == "success")
    failures = sum(1 for r in all_results if r["status"] == "error")
    skipped = sum(1 for r in all_results if r["status"] == "skipped_done")

    print(f"\n{'=' * 80}")
    print("ABLATION SWEEP COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Configs:   {len(configs_to_run)}")
    print(f"  Threads:   {len(thread_dirs)}")
    print(f"  Success:   {successes}/{total_sims}")
    print(f"  Failed:    {failures}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Output:    {ablation_root}/")
    print(f"{'=' * 80}")

    # Per-config summary
    print("\nPer-config results:")
    for config_name in configs_to_run:
        config_results = [
            r for r in all_results if r["config"] == config_name
        ]
        ok = sum(1 for r in config_results if r["status"] == "success")
        err = sum(1 for r in config_results if r["status"] == "error")
        skip = sum(1 for r in config_results if r["status"] == "skipped_done")
        print(f"  {config_name:30s}  ok={ok}  err={err}  skip={skip}")


if __name__ == "__main__":
    main()
