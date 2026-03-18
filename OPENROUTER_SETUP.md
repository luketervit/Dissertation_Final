# OpenRouter Simulation Runtime

This repo now includes a standalone OpenRouter runtime that does not depend on the older locked local/Ollama modules.

## Files

- `.env.openrouter`: OpenRouter API key and default model.
- `openrouter_runtime/client.py`: OpenRouter client with retries and a shared 20 RPM rate limiter.
- `openrouter_runtime/simulation.py`: Minimal thread simulator with parallel reply generation.
- `scripts/check_openrouter_limits.py`: Inspect current key status.
- `scripts/run_openrouter_simulation.py`: Run a single simulation.
- `scripts/run_openrouter_batch.py`: Run multiple simulations in parallel while sharing one rate budget.

## Single run

```bash
python3 scripts/run_openrouter_simulation.py
```

## Batch run

```bash
python3 scripts/run_openrouter_batch.py --workers 4 --rpm 20
```

## Notes

- The new runtime automatically loads `.env.openrouter`.
- It now defaults to `meta-llama/llama-3.3-70b-instruct`.
- `cognitivecomputations/dolphin-llama-3-70b` is not currently in the live OpenRouter catalog, so the runtime preflights the model ID and fails fast if a requested model is unavailable.
- Parallelism is bounded by the shared OpenRouter rate limiter, so you can raise `--workers` without exceeding the configured request budget.
