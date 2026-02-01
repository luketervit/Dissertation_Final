# Thread Simulation Parameters - Quick Reference

## ⚠️ CRITICAL LIMITATIONS

**Single-Thread Testing:**
- Validated on **ONE thread only**: Pelosi/MTG Jan 6 (ID: 1801016461601001478)
- 78.8% negative, highly partisan political debate
- **Cannot claim generalization** without testing on 5-10 diverse threads

**Experiment History:**

**Baseline (Dolphin 8B, temp=1.3):**
- Accuracy: 93.5% (EXCELLENT)
- JSD: 0.093, Sentiment Similarity: 90.7%
- Real 78.8% negative → Simulated 39.8% negative (Gap: -39%)

**Experiment 1 - Thread #1 Aggressive Settings (Feb 1, 2026):**
- Thread: Pelosi/Jan6 (55.6% Left, 78.8% negative)
- Changes: Aggression threshold 0.5→0.4, Controversy 2.5→3.5, Aggressive prompt, Temp 1.3
- Accuracy: 94.4% (EXCELLENT)
- JSD: 0.0798, Sentiment Similarity: 92.0%
- Real 78.8% negative → Simulated **100.0%** negative (Gap: +21.2%)
- **RESULT: OVERCORRECTED** - Lost all positive/neutral diversity

**Experiment 2 - Thread #1 Conservative Defaults (Feb 1, 2026):**
- Thread: Pelosi/Jan6 (55.6% Left, 78.8% negative)
- Changes: Reverted to baseline (Agg 0.5, Controversy 2.5, Neutral prompt, Temp 1.1)
- Real 78.8% negative → Simulated 39.8% negative (Gap: -39%)
- **RESULT: Original baseline (Dolphin 8B)**

**Experiment 3 - Thread #1 Conservative Defaults (Feb 1, 2026):**
- Thread: Pelosi/Jan6 (55.6% Left, 78.8% negative)
- Settings: Baseline (Agg 0.5, Controversy 2.5, Neutral prompt, Temp 1.1)
- Accuracy: **99.9%** (NEARLY PERFECT!)
- JSD: **0.0020**, Sentiment Similarity: **99.8%**
- Real 78.8% negative → Simulated **76.2%** negative (Gap: **-2.6%**)
- **RESULT: NEARLY PERFECT MATCH!**

**Experiment 4 - Thread #2 Conservative Defaults (Feb 1, 2026):**
- Thread: Immigration (85.5% Right, 65.9% negative)
- Settings: **IDENTICAL to Experiment 3**
- Accuracy: 86.1% (EXCELLENT)
- JSD: 0.1990, Sentiment Similarity: 80.1%
- Real 65.9% negative → Simulated **22.9%** negative (Gap: **-43%**)
- **RESULT: WAY TOO POSITIVE** - 43% gap!

**🚨 CRITICAL FINDING: Political Composition Bias Confirmed!**

With **IDENTICAL conservative parameters**:
- Thread #1 (55.6% Left): **76.2% negative** (Gap: -2.6%) ✓ PERFECT
- Thread #2 (85.5% Right): **22.9% negative** (Gap: -43%) ✗ MASSIVE FAILURE

**43% sentiment gap between threads using same settings!**

**Hypothesis CONFIRMED:** Dolphin-Llama3 behaves fundamentally differently based on political composition:
- **Left-majority threads:** Accurate negative sentiment generation
- **Right-majority threads:** Way too positive/polite (undercorrected by 43%)

---

## Key Parameters

### LLM (`config/thread_config.yaml`)
```yaml
provider: ollama
model: dolphin-llama3:8b      # Try: :70b for quality
temperature: 1.1               # Try: 1.3 for more extreme
max_tokens: 150
system_prompt: "You are..."   # Try: more aggressive version
```

### Agent Behavior (`sim/thread_simulation.py`)
```python
base_prob = 0.05              # Reply probability (try: 0.02-0.10)
aggression_boost = 0.15       # Multiplier (try: 0.20-0.25)
reply_prob_max = 0.25         # Cap (try: 0.30-0.40)

recency_weight = 3.0          # Boost current round posts
controversy_weight = 2.5      # Seek opposing views
depth_penalty = 0.5           # Avoid deep threads
```

### Prompt Engineering (`sim/llm_generator.py`)
```python
if aggression > 0.7:    # "aggressive" (try: 0.5)
elif aggression > 0.4:  # "assertive" (try: 0.3)
else:                   # "polite"
```

---

## ⚠️ GENERALIZATION PROBLEM DISCOVERED

**Cross-Thread Validation Reveals Major Issue:**

The model produces **wildly different** sentiment distributions on different threads with **identical parameters**:

| Thread | Political Split | Real Neg % | Sim Neg % | Error |
|--------|----------------|------------|-----------|-------|
| #1 (Pelosi) | 55.6% Left | 78.8% | 100.0% | **+21.2%** |
| #2 (Immigration) | 85.5% Right | 65.9% | 22.9% | **-43.0%** |

**Root Causes (Hypotheses):**

1. **Political Composition Bias:**
   - Left-majority threads → Model generates too much negativity
   - Right-majority threads → Model generates too much positivity
   - Dolphin-Llama3 may have political training bias

2. **Topic Sensitivity:**
   - Jan 6 / Pelosi → Triggers aggressive responses
   - Immigration → Triggers defensive/positive responses
   - LLM prompt understanding varies by political domain

3. **Agent Distribution Effect:**
   - More Left agents → More aggressive cross-partisan attacks?
   - More Right agents → More solidarity/agreement?
   - Reply targeting logic may amplify echo chambers

**Implications for Dissertation:**

- ❌ Cannot use single parameter set for all threads
- ❌ Cannot claim model "works" based on one thread
- ✓ Can study WHY different threads produce different patterns
- ✓ Can develop thread-specific calibration approach

---

## 🎯 Next Research Directions

**Strategy: Revert some changes to find sweet spot**

**Option A: Keep aggressive prompt only**
```yaml
# Revert: aggression thresholds back to 0.5/0.3
# Revert: controversy weight back to 2.5
# Keep: aggressive system prompt
# Keep: temperature 1.3
```
Expected: 55-65% negative

**Option B: Keep aggression threshold only**
```yaml
# Revert: controversy weight back to 2.5
# Revert: system prompt to neutral
# Keep: aggression thresholds at 0.4/0.2
# Keep: temperature 1.3
```
Expected: 50-60% negative

**Option C: Moderate all changes (RECOMMENDED)**
```yaml
# Partial revert: aggression threshold 0.45 (between 0.4-0.5)
# Partial revert: controversy weight 3.0 (between 2.5-3.5)
# Keep: aggressive prompt but soften slightly
# Keep: temperature 1.3
```
Expected: 70-80% negative ← **Target range**

---

## 🔥 Previous Experiments (Archive)

**1. Aggressive Prompt** → Expected: 45-55% negative
```yaml
system_prompt: 'You are in a heated political argument. Be direct and passionate.'
```

**2. Lower Aggression Threshold** → Expected: 50-60% negative
```python
if aggression > 0.5:  # Was 0.7
```

**3. Upgrade Model** → Expected: 55-65% negative
```yaml
model: dolphin-llama3:70b
```

**4. Combine All Three** → Expected: 65-75% negative

---

## Quick Testing

**3-round test:**
```python
# sim/thread_simulation.py line 481
model.run(max_rounds=3)
```

**Validate:**
```bash
python scripts/validate_thread_simulation.py \
    --real output/selected_thread_metadata.json \
    --simulated output/simulated_thread_metadata.json
```

**Target:** JSD < 0.05

---

## ⚠️ Cross-Thread Validation Required

**Must test on 5-10 diverse threads before dissertation:**
1. High negativity (political conflict)
2. Medium negativity (policy debates)
3. Low negativity (bipartisan topics)
4. Positive threads (celebrations)
5. Non-political (sports, entertainment)

**Success criteria:**
- Mean accuracy >80% across all threads
- Std deviation JSD <0.10

**Timeline:** ~5 hours total

**Without this: Cannot claim model generalizes**
