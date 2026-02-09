"""
Run Dual Batch Simulations: OLD params vs NEW params.

Runs the same 100 threads twice:
  1. OLD parameters (pre-Step-10: no few-shot, 4 aggression tiers, aggressive Rule 6)
  2. NEW parameters (Step-10 fixes: few-shot, 5 tiers, softened Rule 6, dynamic scaling)

Saves to:
  - batch_output_old/thread_NNN/simulation_output/
  - batch_output_new/thread_NNN/simulation_output/

Usage:
    python scripts/run_dual_batch.py --run old   # Run only old params
    python scripts/run_dual_batch.py --run new   # Run only new params
    python scripts/run_dual_batch.py --run both  # Run both sequentially
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Monkey-patch helpers to switch between OLD and NEW parameter sets
# ---------------------------------------------------------------------------

def patch_old_parameters():
    """
    Revert llm_generator to OLD (pre-Step-10) behaviour:
    - No few-shot grounding (ignore few_shot_examples)
    - 4 aggression tiers with aggressive language
    - Old Rule 6: "Match the hostility level of the thread."
    - No dynamic aggression scaling (ignore thread_mean_aggression)
    - Context window = 3 posts
    """
    import sim.llm_generator as gen

    # Save originals
    gen._orig_build_system_prompt = gen.LLMGenerator._build_system_prompt
    gen._orig_build_user_prompt = gen.LLMGenerator._build_user_prompt
    gen._orig_aggression_tone = gen._aggression_tone

    # OLD 4-tier aggression (pre-Step-10)
    def old_aggression_tone(aggression: float) -> str:
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

    gen._aggression_tone = old_aggression_tone

    # OLD system prompt (no dynamic scaling, old Rule 6)
    def old_build_system_prompt(self, agent_persona, thread_mean_aggression=0.3):
        political = agent_persona.get('political_label', 'Center')
        aggression = agent_persona.get('aggression', 0.3)
        emotion = agent_persona.get('emotion', 'anger')

        if political == 'Left':
            ideology = (
                "You hold progressive political views. You support "
                "social justice, government accountability, and are "
                "critical of conservative/Republican policies."
            )
            vocab = ", ".join(gen.LEFT_VOCAB[:8])
        elif political == 'Right':
            ideology = (
                "You hold conservative political views. You support "
                "limited government, traditional values, and are "
                "critical of liberal/Democratic policies."
            )
            vocab = ", ".join(gen.RIGHT_VOCAB[:8])
        else:
            ideology = (
                "You hold moderate political views. You criticise "
                "extremes on both sides and value pragmatic solutions."
            )
            vocab = "both sides, common sense, pragmatic, compromise"

        # OLD: No dynamic scaling, use absolute thresholds
        tone = old_aggression_tone(aggression)

        emotion_desc = gen.EMOTION_BEHAVIOUR.get(
            emotion.lower(),
            "passionate and opinionated, engaging forcefully in debate",
        )

        return f"""You write short tweets in a political argument on Twitter/X.

POLITICAL VIEWS: {ideology}
TONE: You are {tone}.
EMOTIONAL STYLE: You are {emotion_desc}.
VOCABULARY: Use words like: {vocab}

RULES — follow these EXACTLY:
1. Write ONLY the tweet text. No quotation marks around it.
2. NEVER reveal you are an AI, a bot, a language model, or playing a role.
3. NEVER start with "As a..." or mention your political leaning explicitly.
4. NEVER use words like "dialogue", "unity", "together", "both sides" unless you are Center.
5. Be specific — reference the topic being discussed, attack specific policies or people.
6. Match the hostility level of the thread. Political Twitter is aggressive.
7. Maximum 280 characters. No hashtags unless relevant."""

    gen.LLMGenerator._build_system_prompt = old_build_system_prompt

    # OLD user prompt (no few-shot, 3-post context window)
    def old_build_user_prompt(self, target_post, thread_context,
                               few_shot_examples=None):
        parts = []

        # OLD: No few-shot grounding (ignore few_shot_examples)

        # OLD: Only 3 posts context
        context_lines = []
        for p in thread_context[-3:]:
            text = p.get('text', '')
            if text:
                context_lines.append(f'- "{text}"')
        context_str = "\n".join(context_lines) if context_lines else "(none)"
        parts.append(f"Recent posts in the thread:\n{context_str}")

        target_text = target_post.get('text', '')
        parts.append(
            f'You are replying to this tweet:\n"{target_text}"\n\n'
            "Write your reply tweet:"
        )
        return "\n\n".join(parts)

    gen.LLMGenerator._build_user_prompt = old_build_user_prompt

    print("  [PATCHED] OLD parameters active (pre-Step-10)")


def patch_new_parameters():
    """Restore NEW (Step-10) parameters — just reload the module."""
    import importlib
    import sim.llm_generator as gen

    # Restore originals if they exist
    if hasattr(gen, '_orig_build_system_prompt'):
        gen.LLMGenerator._build_system_prompt = gen._orig_build_system_prompt
        gen.LLMGenerator._build_user_prompt = gen._orig_build_user_prompt
        gen._aggression_tone = gen._orig_aggression_tone
    else:
        # Fresh import — current code IS new params
        importlib.reload(gen)

    print("  [PATCHED] NEW parameters active (Step-10 fixes)")


# ---------------------------------------------------------------------------
# Simulation runner (same as run_batch_simulations.py)
# ---------------------------------------------------------------------------

def run_single_thread(thread_dir: Path, output_dir: Path) -> dict:
    from sim.thread_simulation import ThreadModel
    import yaml

    config_path = thread_dir / 'config.yaml'
    if not config_path.exists():
        return {'status': 'skipped', 'reason': 'no config.yaml'}

    sim_output = output_dir / 'simulation_output'
    sim_output.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        model = ThreadModel(config_path=str(config_path))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        max_rounds = config.get('simulation', {}).get('max_rounds', 10)

        model.run(max_rounds=max_rounds)
        model.export_results(output_dir=str(sim_output))

        elapsed = time.time() - start_time
        return {
            'status': 'success',
            'total_posts': len(model.thread_history),
            'elapsed_seconds': round(elapsed, 1),
            'agents': len(model.agent_list),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed_seconds': round(elapsed, 1),
        }


def run_batch(input_dir: Path, output_dir: Path, label: str,
              start: int = 1, end: int = 999) -> None:
    thread_dirs = sorted(input_dir.glob('thread_*'))
    thread_dirs = [
        d for d in thread_dirs
        if start <= int(d.name.split('_')[1]) <= end
    ]

    if not thread_dirs:
        print(f"ERROR: No thread directories found in {input_dir}")
        return

    print(f"\n{'=' * 80}")
    print(f"BATCH SIMULATION [{label}]: {len(thread_dirs)} threads")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total_start = time.time()

    for i, thread_dir in enumerate(thread_dirs):
        thread_name = thread_dir.name
        thread_output = output_dir / thread_name

        # Skip if already completed
        done_marker = (
            thread_output / 'simulation_output'
            / 'simulated_thread_metadata.json'
        )
        if done_marker.exists():
            print(
                f"\n[{i+1}/{len(thread_dirs)}] {thread_name}: "
                "ALREADY DONE, skipping"
            )
            results.append({'thread': thread_name, 'status': 'skipped_done'})
            continue

        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(thread_dirs)}] Running {thread_name} [{label}]...")

        result = run_single_thread(thread_dir, thread_output)
        result['thread'] = thread_name
        results.append(result)

        if result['status'] == 'success':
            print(
                f"  OK: {result['total_posts']} posts, "
                f"{result['agents']} agents, "
                f"{result['elapsed_seconds']}s"
            )
        else:
            print(f"  FAILED: {result.get('error', 'unknown')}")

        # Save progress
        progress = {
            'label': label,
            'completed': sum(
                1 for r in results if r['status'] == 'success'
            ),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'total_elapsed': round(time.time() - total_start, 1),
            'results': results,
        }
        with open(output_dir / 'batch_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

    total_elapsed = time.time() - total_start
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{'=' * 80}")
    print(f"BATCH [{label}] COMPLETE")
    print(f"  Success: {successes}/{len(thread_dirs)}")
    print(f"  Failed: {failures}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dual batch: old vs new parameters"
    )
    parser.add_argument(
        "--run",
        choices=["old", "new", "both"],
        default="both",
        help="Which parameter set to run",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="batch_simulations_reconstructed",
        help="Input directory with prepared threads",
    )
    parser.add_argument(
        "--start", type=int, default=1, help="Start thread number"
    )
    parser.add_argument(
        "--end", type=int, default=999, help="End thread number"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input

    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} not found.")
        print("Run scripts/reconstruct_batch_inputs.py first.")
        sys.exit(1)

    if args.run in ("old", "both"):
        patch_old_parameters()
        run_batch(
            input_dir,
            project_root / "batch_output_old",
            label="OLD PARAMS",
            start=args.start,
            end=args.end,
        )

    if args.run in ("new", "both"):
        patch_new_parameters()
        run_batch(
            input_dir,
            project_root / "batch_output_new",
            label="NEW PARAMS",
            start=args.start,
            end=args.end,
        )

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    if args.run == "both":
        print("  Old params: batch_output_old/")
        print("  New params: batch_output_new/")
    print("Pull results locally and run validation.")


if __name__ == "__main__":
    main()
