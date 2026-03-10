"""
Run batch thread simulations for model comparison (one model at a time).

Designed for GCP/nohup runs on the same prepared 100-thread set used by
the baseline run.

Examples:
  python scripts/run_model_comparison_batch.py --model llama
  python scripts/run_model_comparison_batch.py --model qwen
  python scripts/run_model_comparison_batch.py --model deepseek_r1
  python scripts/run_model_comparison_batch.py --model deepseek_llm
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


MODEL_PRESETS = {
    "llama": {
        "provider": "ollama",
        "model": "dolphin-llama3:8b",
        "api_key_env": None,
        "temperature": 0.9,
        "max_tokens": 150,
    },
    # Local Ollama models suitable for T4.
    "qwen": {
        "provider": "ollama",
        "model": "qwen2.5:14b",
        "api_key_env": None,
        "temperature": 0.9,
        "max_tokens": 150,
    },
    "deepseek_r1": {
        "provider": "ollama",
        "model": "deepseek-r1:14b",
        "api_key_env": None,
        "temperature": 0.9,
        "max_tokens": 150,
    },
    # DeepSeek API (OpenAI-compatible endpoint) using DEEPSEEK_API_KEY.
    "deepseek_llm": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.9,
        "max_tokens": 150,
    },
}

MODEL_OUTPUT_DIR = {
    "llama": "batch_output_llama",
    "qwen": "batch_output_qwen",
    "deepseek_r1": "batch_output_deepseek_r1",
    "deepseek_llm": "batch_output_deepseek_llm",
}


def _thread_num(thread_dir: Path) -> int:
    return int(thread_dir.name.split("_")[1])


def discover_threads(input_dir: Path, start: int, end: int) -> list[Path]:
    thread_dirs = sorted(
        [d for d in input_dir.glob("thread_*") if d.is_dir()],
        key=_thread_num,
    )
    return [d for d in thread_dirs if start <= _thread_num(d) <= end]


def assert_same_threads(
    input_threads: list[Path],
    reference_dir: Path | None,
) -> None:
    if reference_dir is None:
        return
    if not reference_dir.exists():
        return

    ref_threads = sorted(
        [d.name for d in reference_dir.glob("thread_*") if d.is_dir()]
    )
    in_threads = sorted([d.name for d in input_threads])

    if in_threads != ref_threads:
        missing_in_input = sorted(set(ref_threads) - set(in_threads))
        missing_in_ref = sorted(set(in_threads) - set(ref_threads))
        raise ValueError(
            "Input thread set does not match baseline reference set.\n"
            f"  Missing in input: {missing_in_input[:5]}\n"
            f"  Missing in reference: {missing_in_ref[:5]}"
        )


def build_runtime_config(
    base_config: dict,
    model_key: str,
    thread_dir: Path,
    max_rounds_override: int | None,
) -> dict:
    cfg = copy.deepcopy(base_config)
    preset = MODEL_PRESETS[model_key]

    cfg.setdefault("llm", {})
    cfg["llm"]["provider"] = preset["provider"]
    cfg["llm"]["model"] = preset["model"]
    cfg["llm"]["api_key_env"] = preset["api_key_env"]
    cfg["llm"]["temperature"] = preset["temperature"]
    cfg["llm"]["max_tokens"] = preset["max_tokens"]

    if preset["provider"] == "deepseek":
        cfg["llm"]["base_url"] = preset["base_url"]
    else:
        cfg["llm"].pop("base_url", None)
        cfg["llm"].pop("base_url_env", None)

    if max_rounds_override is not None:
        cfg.setdefault("simulation", {})
        cfg["simulation"]["max_rounds"] = max_rounds_override

    # Normalize path fields to local thread dir if absolute path doesn't exist.
    cfg.setdefault("paths", {})
    for field in ("thread_metadata", "agents_for_thread"):
        original = cfg["paths"].get(field)
        if not original:
            continue
        original_path = Path(str(original))
        if original_path.exists():
            continue
        fallback = thread_dir / original_path.name
        if fallback.exists():
            cfg["paths"][field] = str(fallback)
        else:
            raise FileNotFoundError(
                f"Could not resolve path '{field}': {original}"
            )

    return cfg


def run_single_thread(
    thread_dir: Path,
    thread_output_dir: Path,
    runtime_config: dict,
) -> dict:
    from sim.thread_simulation import ThreadModel

    thread_output_dir.mkdir(parents=True, exist_ok=True)
    sim_output = thread_output_dir / "simulation_output"
    sim_output.mkdir(parents=True, exist_ok=True)

    runtime_config_path = thread_output_dir / "runtime_config.yaml"
    with open(runtime_config_path, "w") as f:
        yaml.safe_dump(runtime_config, f, sort_keys=False)

    start = time.time()
    try:
        model = ThreadModel(config_path=str(runtime_config_path))
        max_rounds = (
            runtime_config.get("simulation", {}).get("max_rounds", 10)
        )
        model.run(max_rounds=max_rounds)
        model.export_results(output_dir=str(sim_output))

        return {
            "status": "success",
            "elapsed_seconds": round(time.time() - start, 1),
            "total_posts": len(model.thread_history),
            "agents": len(model.agent_list),
            "max_rounds": max_rounds,
        }
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_seconds": round(time.time() - start, 1),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def save_progress(
    output_dir: Path,
    model_key: str,
    provider: str,
    model_name: str,
    results: list[dict],
    started_at: float,
) -> None:
    completed = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped_done")
    payload = {
        "model_key": model_key,
        "provider": provider,
        "model": model_name,
        "completed": completed,
        "failed": failed,
        "skipped_done": skipped,
        "total": len(results),
        "elapsed_seconds": round(time.time() - started_at, 1),
        "updated_at_epoch": time.time(),
        "results": results,
    }
    with open(output_dir / "batch_progress.json", "w") as f:
        json.dump(payload, f, indent=2)


def run_model_batch(
    project_root: Path,
    input_dir: Path,
    output_dir: Path,
    model_key: str,
    start: int,
    end: int,
    max_rounds_override: int | None,
    force: bool,
    reference_dir: Path | None,
    expected_threads: int | None,
    dry_run: bool,
) -> None:
    threads = discover_threads(input_dir, start, end)
    if not threads:
        raise ValueError(f"No thread directories found in {input_dir}")
    if expected_threads is not None and len(threads) != expected_threads:
        raise ValueError(
            f"Expected {expected_threads} threads, found {len(threads)} in "
            f"range {start}-{end}"
        )

    assert_same_threads(threads, reference_dir)

    preset = MODEL_PRESETS[model_key]
    print("=" * 80)
    print(f"MODEL BATCH: {model_key}")
    print(f"  provider: {preset['provider']}")
    print(f"  model:    {preset['model']}")
    print(f"  input:    {input_dir}")
    print(f"  output:   {output_dir}")
    print(f"  threads:  {len(threads)} ({threads[0].name} .. {threads[-1].name})")
    print("=" * 80)

    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    results: list[dict] = []

    for idx, thread_dir in enumerate(threads, start=1):
        thread_name = thread_dir.name
        thread_output_dir = output_dir / thread_name
        done_marker = (
            thread_output_dir
            / "simulation_output"
            / "simulated_thread_metadata.json"
        )

        print(f"\n[{idx}/{len(threads)}] {thread_name}")

        if done_marker.exists() and not force:
            print("  already complete; skipping")
            result = {"thread": thread_name, "status": "skipped_done"}
            results.append(result)
            save_progress(
                output_dir,
                model_key,
                preset["provider"],
                preset["model"],
                results,
                started_at,
            )
            continue

        config_path = thread_dir / "config.yaml"
        if not config_path.exists():
            print("  missing config.yaml; skipping")
            result = {
                "thread": thread_name,
                "status": "error",
                "error": "missing config.yaml",
            }
            results.append(result)
            save_progress(
                output_dir,
                model_key,
                preset["provider"],
                preset["model"],
                results,
                started_at,
            )
            continue

        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        runtime_config = build_runtime_config(
            base_config=base_config,
            model_key=model_key,
            thread_dir=thread_dir,
            max_rounds_override=max_rounds_override,
        )

        result = run_single_thread(
            thread_dir=thread_dir,
            thread_output_dir=thread_output_dir,
            runtime_config=runtime_config,
        )
        result["thread"] = thread_name
        results.append(result)

        if result["status"] == "success":
            print(
                "  success: "
                f"{result['total_posts']} posts, "
                f"{result['agents']} agents, "
                f"{result['elapsed_seconds']}s"
            )
        else:
            print(f"  error: {result.get('error', 'unknown')}")

        save_progress(
            output_dir,
            model_key,
            preset["provider"],
            preset["model"],
            results,
            started_at,
        )

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped_done")
    elapsed = round(time.time() - started_at, 1)

    summary = {
        "model_key": model_key,
        "provider": preset["provider"],
        "model": preset["model"],
        "threads_total": len(threads),
        "success": succeeded,
        "failed": failed,
        "skipped_done": skipped,
        "elapsed_seconds": elapsed,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"COMPLETE: {model_key}")
    print(
        f"  success={succeeded}, failed={failed}, "
        f"skipped={skipped}, elapsed={elapsed}s"
    )
    print(f"  output={output_dir}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model-comparison batch simulations (one model at a time)."
    )
    parser.add_argument(
        "--model",
        choices=["llama", "qwen", "deepseek_r1", "deepseek_llm", "all"],
        required=True,
        help="Model preset to run.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="batch_simulations_reconstructed",
        help="Prepared thread input directory.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="batch_output_new",
        help=(
            "Reference output directory used to verify the same thread set. "
            "Set to '' to disable."
        ),
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument(
        "--expected-threads",
        type=int,
        default=100,
        help="Expected number of threads in selected range.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override simulation.max_rounds in thread configs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Root directory for output folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a thread output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate thread set and print plan without running.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = (project_root / args.input).resolve()
    output_root = (project_root / args.output_root).resolve()
    reference_dir = (
        None
        if args.reference.strip() == ""
        else (project_root / args.reference).resolve()
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model_sequence = (
        ["llama", "qwen", "deepseek_r1", "deepseek_llm"]
        if args.model == "all"
        else [args.model]
    )

    for model_key in model_sequence:
        run_model_batch(
            project_root=project_root,
            input_dir=input_dir,
            output_dir=output_root / MODEL_OUTPUT_DIR[model_key],
            model_key=model_key,
            start=args.start,
            end=args.end,
            max_rounds_override=args.max_rounds,
            force=args.force,
            reference_dir=reference_dir,
            expected_threads=args.expected_threads,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
