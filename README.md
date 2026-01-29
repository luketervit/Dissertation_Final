# Political Persuasion ABM - Dissertation Project

Agent-Based Model simulating political persuasion on Twitter/X to predict "Latent Opinion Shift" in the silent majority (lurkers).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Classify agents and extract latent DNA
python step1_classify.py

# Step 2: Run the ABM simulation
python lurker_sim.py

# Run tests
pytest
```

## Project Structure

```
├── data/              # Raw USC X 24 US Election Dataset
├── models/            # RoBERTa stance & emotion classifiers
├── sim/               # Mesa ABM (agents, model, schedule)
├── scripts/           # Data processing & DNA extraction
├── output/            # Results & "Ghost Shift" visualizations
├── tests/             # Pytest test suite
├── step1_classify.py  # Entry point for classification
└── lurker_sim.py      # Entry point for simulation
```

## Key Concepts

- **90-9-1 Rule**: ~15-20 lurker agents per active agent
- **Bounded Confidence**: Opinion shifts toward similar content
- **Backfire Effect**: Opinion shifts away from extreme opposing content
- **Ghost Shift**: Cumulative opinion change in non-posting lurkers
- **12h/12h Split**: Phase 1 playback → Phase 2 prediction validation

## Deadline

Final submission: **March 9th, 2026**
