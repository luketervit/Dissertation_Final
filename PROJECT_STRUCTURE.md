# Project Structure & Pipeline Guide

## Complete Pipeline (From Raw Data → Validated Simulation)

### Step 1: Agent Classification
```bash
python scripts/step1_classify_chunked.py [chunk_number]
```
**Input:** `data/may_july_chunk_N.csv` (raw tweets)
**Output:** `output/processed_agents_raw_N_political.csv` (user DNA profiles)
**What it does:** Classifies each user's political leaning, emotion, sentiment, hate/offensive scores

---

### Step 2: Thread Selection & Extraction
```bash
python scripts/run_thread_pipeline.py
```
**Input:**
- `config/thread_config.yaml` (thread selection criteria)
- `output/processed_agents_raw_1_political.csv` (agent DNA)
- `data/may_july_chunk_1.csv` (raw tweets)

**Output:**
- `output/selected_thread_metadata.json` (real thread with temporal events)
- `output/selected_thread_tweets.csv` (thread tweets)
- `output/agents_for_tweet.csv` (DNA for agents in this thread)

**What it does:**
- Finds thread with most replies
- Extracts all tweets in conversation
- Matches users to their DNA profiles
- Creates temporal event timeline

---

### Step 3: Thread Simulation
```bash
python sim/thread_simulation.py
```
**Input:**
- `config/thread_config.yaml` (LLM config, ABM parameters)
- `output/agents_for_tweet.csv` (agent DNA)
- `output/selected_thread_metadata.json` (root tweet)

**Output:**
- `output/simulated_thread_metadata.json` (simulated thread)
- `output/thread_history.json` (simple format)
- `output/summary_stats.txt` (statistics)
- `output/simulation_timeseries.csv` (round-by-round metrics)

**What it does:**
- Initializes 169 agents with real DNA
- Runs 10 rounds of LLM-powered conversation
- Agents reply based on personality and thread context
- Generates realistic Twitter discourse

---

### Step 4: Validation
```bash
python scripts/validate_thread_simulation.py \
    --real output/selected_thread_metadata.json \
    --simulated output/simulated_thread_metadata.json
```
**Output:**
- `output/sentiment_comparison.png` (visualization)
- `output/validation_results.json` (metrics)
- Console report with JSD, accuracy, keyword analysis

**What it does:**
- Classifies sentiment of all tweets (real + simulated)
- Calculates Jensen-Shannon Divergence
- Compares keyword usage
- Generates accuracy score

---

## Directory Structure

```
Dissertation_Final/
├── config/
│   └── thread_config.yaml          # Central configuration
├── data/
│   ├── may_july_chunk_1.csv        # Raw USC X-24 tweets
│   └── may_july_chunk_2.csv
├── scripts/
│   ├── step1_classify_chunked.py   # Agent DNA classification
│   ├── find_best_thread.py         # Find high-engagement threads
│   ├── extract_thread.py           # Extract conversation tree
│   ├── create_agents_for_thread.py # Match users to DNA
│   ├── run_thread_pipeline.py      # One-command pipeline
│   └── validate_thread_simulation.py # Validation framework
├── sim/
│   ├── thread_simulation.py        # Main ABM (ThreadAgent, ThreadModel)
│   └── llm_generator.py            # LLM wrapper (Ollama/Anthropic/OpenAI)
├── output/
│   ├── processed_agents_raw_1_political.csv  # User DNA
│   ├── selected_thread_metadata.json         # Real thread
│   ├── simulated_thread_metadata.json        # Simulated thread
│   ├── sentiment_comparison.png              # Validation viz
│   └── validation_results.json               # Metrics
├── archive/
│   ├── old_opinion_dynamics/       # Deprecated ABM code
│   └── utils/                      # One-off scripts
├── .env                            # API keys (not committed)
├── .env.example                    # Template
├── Implementation.md               # Detailed implementation notes
├── implementation_plan.md          # High-level plan
├── PARAMETER_INVENTORY.md          # All tunable parameters
├── PROJECT_STRUCTURE.md            # This file
└── SIMULATION_GUIDE.md             # User guide
```

---

## Configuration Files

### config/thread_config.yaml
**Controls:**
- Which thread to simulate
- LLM provider/model/parameters
- ABM parameters (reply probability, etc.)
- Lurker distribution strategy (future)

**Current LLM:**
- Provider: `ollama`
- Model: `dolphin-llama3:8b`
- Temperature: `1.1`
- Max tokens: `150`

---

## Key Files

### Data Files (Not Committed)
- `data/may_july_chunk_*.csv` - Raw USC X-24 tweets
- `output/processed_agents_raw_*.csv` - User DNA profiles
- `output/*.json` - Thread data and results

### Documentation
- `Implementation.md` - Step-by-step implementation decisions
- `implementation_plan.md` - High-level architecture plan
- `PARAMETER_INVENTORY.md` - All tunable parameters with ranges
- `SIMULATION_GUIDE.md` - How to run simulations
- `CLAUDE.md` - Project instructions for Claude AI

### Code
- **Classification:** `scripts/step1_classify_chunked.py`
- **Pipeline:** `scripts/run_thread_pipeline.py`
- **Simulation:** `sim/thread_simulation.py`
- **Validation:** `scripts/validate_thread_simulation.py`
- **LLM:** `sim/llm_generator.py`

---

## Required Dependencies

```bash
# Core
pip install pandas numpy mesa transformers torch

# Validation
pip install scipy matplotlib

# LLM
pip install anthropic openai python-dotenv

# Local LLM (optional)
brew install ollama
ollama pull dolphin-llama3:8b
```

---

## Quick Start

```bash
# 1. Setup environment
cp .env.example .env
# Add API keys to .env

# 2. Install Ollama + model
brew install ollama
brew services start ollama
ollama pull dolphin-llama3:8b

# 3. Run full pipeline
python scripts/run_thread_pipeline.py  # Extract thread
python sim/thread_simulation.py        # Simulate
python scripts/validate_thread_simulation.py \
    --real output/selected_thread_metadata.json \
    --simulated output/simulated_thread_metadata.json
```

---

## Current Validation Results

**Thread:** Pelosi/MTG Jan 6 recordings debate (ID: 1801016461601001478)

**Model:** Dolphin-Llama3 8B (local, uncensored)

**Results:**
- Overall Accuracy: 93.5% (EXCELLENT)
- Sentiment Similarity: 90.7% (JSD = 0.093)
- Sentiment Distribution:
  - Real: 78.8% Negative, 15.8% Neutral, 5.4% Positive
  - Simulated: 39.8% Negative, 29.0% Neutral, 31.2% Positive

**Gap to Close:** Need +39% more negative sentiment to match real thread

**⚠️ Limitation:** Only tested on ONE thread - generalization not proven

---

## Archived Code

**Location:** `archive/`

**What's there:**
- `old_opinion_dynamics/` - Deprecated bounded confidence ABM (backwards causality issue)
- `utils/` - One-off utility scripts not part of pipeline

**Note:** Archived code may be useful for Phase 2 (lurker opinion integration)

---

## Next Steps (Future Work)

1. **Close Sentiment Gap** (Priority 1)
   - Test Dolphin 70B model
   - Adjust system prompt for more aggression
   - Lower aggression threshold

2. **Cross-Thread Validation** (Priority 2)
   - Test on 5-10 diverse threads
   - Calculate mean ± std across threads
   - Prove model generalizes

3. **Lurker Integration** (Phase 2)
   - Combine thread simulation with opinion dynamics
   - Track "Ghost Shift" in silent majority

4. **Temporal Realism** (Phase 3)
   - Map simulation rounds to real timestamps
   - Model time-dependent engagement patterns

---

## Git Workflow

```bash
# Before committing
git status                  # Check changes
git add <files>             # Stage files
git commit -m "message"     # Commit with message
git push origin main        # Push to GitHub

# What's tracked:
✓ All code (scripts/, sim/)
✓ Configuration (config/)
✓ Documentation (.md files)

# What's ignored (.gitignore):
✗ Data files (data/, output/)
✗ Environment (.env)
✗ Cache (__pycache__)
✗ Archive (archive/)
```
