# GCP Batch Thread Simulation

This directory contains scripts to run 100 thread simulations on GCP and analyze sentiment comparisons between simulated and actual Twitter threads.

## Quick Start

### 1. Prepare Simulations (Local or GCP)
```bash
# Prepare all threads from august_thread_depths.csv
python scripts/prepare_batch_simulations.py

# Or limit to a specific number
python scripts/prepare_batch_simulations.py --limit 10
```

This creates `batch_simulations/thread_001/`, `thread_002/`, etc. with:
- `config.yaml` - Thread-specific configuration
- `thread_metadata.json` - Root tweet and thread info
- `thread_tweets.csv` - All tweets in the thread
- `agents_for_thread.csv` - Agent DNA profiles

### 2. Run Simulations on GCP

**Option A: Run All at Once (Parallel)**
```bash
# Run all simulations in parallel with 4 workers
python scripts/gcp_batch_runner.py --mode parallel --workers 4 --rounds 10
```

**Option B: Run Sequentially**
```bash
python scripts/gcp_batch_runner.py --mode sequential --rounds 10
```

**Option C: Run Specific Range**
```bash
# Run threads 1-20 only
python scripts/gcp_batch_runner.py --start 1 --end 20 --mode parallel
```

**Option D: Use Master Pipeline Script**
```bash
./scripts/run_batch_pipeline.sh --workers 4
```

### 3. Aggregate Results
```bash
python scripts/aggregate_results.py
```

## Output Files

After running all simulations:

```
batch_output/
├── thread_001/
│   ├── simulation_output/
│   │   ├── simulated_thread_metadata.json
│   │   ├── thread_history.json
│   │   └── summary_stats.txt
│   ├── validation_output/
│   │   ├── validation_results.json
│   │   └── sentiment_comparison.png
│   └── summary.json
├── thread_002/
│   └── ...
├── aggregated_results.csv      # One row per thread with all metrics
├── aggregated_statistics.json  # Cross-thread statistics
└── batch_results.json          # Batch execution summary
```

### Key Output: `aggregated_results.csv`

| Column | Description |
|--------|-------------|
| `thread_num` | Thread number (1-100) |
| `root_id` | Original tweet ID |
| `success` | Whether simulation succeeded |
| `real_positive/neutral/negative` | Real thread sentiment distribution |
| `sim_positive/neutral/negative` | Simulated thread sentiment distribution |
| `jensen_shannon_divergence` | Sentiment similarity (lower = more similar) |
| `sentiment_similarity` | Sentiment similarity % |
| `overall_accuracy` | Overall simulation accuracy % |

## Requirements

Ensure these are installed on your GCP VM:
```bash
pip install pandas numpy pyyaml mesa transformers torch scipy matplotlib tqdm
pip install anthropic  # If using Anthropic for LLM
```

## LLM Configuration

Edit the config in `batch_simulations/thread_*/config.yaml`:

**For Ollama (local):**
```yaml
llm:
  provider: ollama
  model: dolphin-llama3:8b
```

**For Anthropic (cloud):**
```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Cost Estimation

| Provider | Cost per Simulation | Total (100 threads) |
|----------|---------------------|---------------------|
| Ollama | Free | Free |
| Claude Haiku | ~$0.25 | ~$25 |
| GPT-3.5 | ~$0.15 | ~$15 |

## Troubleshooting

**Simulation fails with "No agents found":**
- Check that `processed_agents/` directory has agent files
- Ensure agent user IDs match thread participants

**Validation fails with "No sentiment data":**
- Ensure `transformers` and `torch` are installed
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

**Timeout errors:**
- Increase timeout in `gcp_batch_runner.py` (default: 30 min per thread)
- Use fewer rounds: `--rounds 5`
