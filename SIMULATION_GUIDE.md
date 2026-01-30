# LLM-Powered Thread Simulation Guide

## Setup

### 1. Create `.env` file
Copy the example and add your Anthropic API key:
```bash
cp .env.example .env
```

Edit `.env` and add:
```
ANTHROPIC_API_KEY=your-actual-api-key-here
```

### 2. Configuration
Edit `config/thread_config.yaml` if needed:
```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001  # Fast and cheap
  temperature: 0.8                   # Higher = more creative
  max_tokens: 150                    # Max reply length
```

## Running the Simulation

```bash
python sim/thread_simulation.py
```

## How It Works

### Agent Behavior
- **Reply probability:** 5-25% per round (based on aggression)
- **Target selection:** Prefers recent, controversial posts
- **Response generation:** Claude Haiku generates replies based on:
  - Agent's political leaning (Left/Right/Center)
  - Aggression level (0-1)
  - Emotion (joy, anger, etc.)
  - Thread context (last 10 posts)
  - Post being replied to

### Staging System
1. **Stage 1 (step):** All agents read thread, generate replies
2. **Stage 2 (advance):** All replies committed simultaneously

### Output Files
- `thread_output.json` - Full conversation with metadata
- `summary_stats.txt` - Aggregated statistics
- `simulation_timeseries.csv` - Round-by-round metrics

## Cost Estimation
- **Claude Haiku:** ~$0.25 per million input tokens, ~$1.25 per million output tokens
- **10 rounds, 50 posts/round:** ~500 LLM calls
- **Estimated cost:** $0.10-0.30 per 10-round simulation

## Adjusting Parameters

### Lower Response Rate
```python
# In thread_simulation.py, ThreadAgent.step():
base_prob = 0.02  # Down from 0.05 (2% base chance)
```

### Change Model
```yaml
# Faster/cheaper:
model: claude-haiku-4-5-20251001

# Smarter/slower:
model: claude-3-5-sonnet-20241022
```

### Adjust Temperature
```yaml
temperature: 0.7  # More consistent
temperature: 1.0  # More creative/varied
```
