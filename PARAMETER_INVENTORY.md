# Thread Simulation Parameter Inventory

## Current Results Summary

⚠️ **CRITICAL LIMITATION: Single-Thread Validation**
- All results based on **ONE tweet only**: Pelosi/MTG Jan 6 thread (ID: 1801016461601001478)
- Topic: Politically polarizing (Supreme Court ethics, MAGA, Democrats)
- Sample size: 183 real replies, 169 unique users
- **Generalization NOT tested** - parameters may be overfit to this specific thread
- **Required:** Test on 5-10 different threads with varying topics/sentiment to prove model generalizes

**Validation Score:** 93.5% accuracy (EXCELLENT) **on this one thread**
**Sentiment Similarity:** 90.7% (JSD = 0.093)
**Sentiment Distribution:**
- Real Thread: 78.8% Negative, 15.8% Neutral, 5.4% Positive
- Simulated (Dolphin): 39.8% Negative, 29.0% Neutral, 31.2% Positive
- **Gap to close:** +39% more negative sentiment needed

---

## 1. LLM Parameters (config/thread_config.yaml)

| Parameter | Current Value | Location | Effect | Suggested Range | Impact on Results |
|-----------|--------------|----------|--------|----------------|-------------------|
| `provider` | `ollama` | config | LLM backend (ollama/anthropic/openai) | ollama, anthropic, openai | Dolphin = 93.5% accuracy, Claude = 74.1% |
| `model` | `dolphin-llama3:8b` | config | Which model to use | dolphin-llama3:8b, :70b, claude-haiku | 8B = 39.8% neg, Claude = 12% neg |
| `temperature` | `1.1` | config | Response randomness/creativity | 0.7-1.5 | Higher → more varied/edgy, Lower → more consistent |
| `max_tokens` | `150` | config | Max reply length | 50-280 | Longer allows more complex arguments |
| `system_prompt` | "You are a Twitter user..." | config | Base personality instruction | See below | Current = too polite, needs more aggression |

**System Prompt Tuning Options:**
- **Current:** "Stay in character... Be authentic and conversational"
- **More Aggressive:** "Be direct, blunt, and emotionally charged. Use raw, unfiltered language..."
- **Most Aggressive:** "Argue passionately. Don't hold back. Twitter is combative, not polite..."

---

## 2. Agent Behavior Parameters (sim/thread_simulation.py)

### Reply Probability (Lines 56-58)

| Parameter | Current Value | Location | Effect | Suggested Range | Impact |
|-----------|--------------|----------|--------|----------------|--------|
| `base_prob` | `0.05` (5%) | thread_simulation.py:56 | Baseline reply chance for all agents | 0.02-0.15 | Lower → fewer posts, Higher → more active thread |
| `aggression_boost` | `aggression * 0.15` | thread_simulation.py:57 | How much aggression increases reply rate | 0.10-0.25 | Higher → aggressive agents dominate |
| `reply_prob` (max) | `0.25` (25%) | thread_simulation.py:58 | Maximum reply probability cap | 0.15-0.40 | Current caps most aggressive agents at 25% |

**Formula:** `reply_prob = min(0.25, 0.05 + aggression * 0.15)`

**Current Results:** ~22 posts per round (13% of 169 agents replying)

**Sensitivity Analysis:**
- `base_prob = 0.02` → ~10 posts/round (sparse conversation)
- `base_prob = 0.10` → ~40 posts/round (very active)
- `aggression_boost = 0.25` → aggressive agents reply ~50% of time

---

### Target Selection Weights (Lines 121-146)

| Parameter | Current Value | Location | Effect | Suggested Range | Notes |
|-----------|--------------|----------|--------|----------------|-------|
| **Recency Weight** | `3.0x` | line 126 | Boost for current round posts | 2.0-5.0 | Higher → agents focus on latest posts |
| **Controversy Weight (high agg)** | `2.5x` | line 136 | Aggressive agents seek opposing views | 2.0-4.0 | Higher → more cross-partisan conflict |
| **Controversy Weight (low agg)** | `0.3x` | line 139 | Moderate agents avoid conflict | 0.1-0.5 | Lower → moderates stay silent |
| **Depth Penalty** | `0.5x` (depth > 5) | line 146 | Discourage deep thread branches | 0.3-0.8 | Lower → flatter threads |
| **Political Distance** | `1 + distance` | line 142 | General controversy attraction | 1.0-2.0 | Current = linear, could be exponential |

**Current Effect:** Agents mostly reply to recent posts from opposing political side

---

### Context Window (Lines 174, 108)

| Parameter | Current Value | Location | Effect | Suggested Range | Impact |
|-----------|--------------|----------|--------|----------------|--------|
| `thread_context` | Last 10 posts | thread_simulation.py:174 | How much history LLM sees | 5-20 posts | More context → better coherence, slower inference |
| `recent_posts` | Last 50 posts | thread_simulation.py:114 | Pool for target selection | 20-100 posts | Larger → more variety, smaller → focus on recent |
| `fallback` | Last 20 posts | thread_simulation.py:117 | Minimum if filtering fails | 10-50 posts | Rarely triggered |

**Current LLM Prompt Context:** Last 10 posts (only last 3 shown in prompt - see llm_generator.py:108)

**Cost vs Quality:**
- 3 posts context = fast, may miss nuance
- 10 posts context = balanced
- 20 posts context = better coherence, 2x slower

---

## 3. Simulation Duration (sim/thread_simulation.py)

| Parameter | Current Value | Location | Effect | Suggested Range | Impact |
|-----------|--------------|----------|--------|----------------|--------|
| `max_rounds` | `10` | thread_simulation.py:481 | Number of simulation rounds | 5-50 | Current: 10 rounds = ~200 posts total |

**Round Time:**
- 1 round = ~60 seconds simulated time (timestamp increment)
- 10 rounds = 10 minutes simulated (vs real thread: 2.5 hours)

**Temporal Realism Issue:** Need to map rounds to actual elapsed time in real thread

---

## 4. Prompt Engineering Parameters (sim/llm_generator.py)

| Parameter | Current Value | Location | Effect | Suggested Range | Notes |
|-----------|--------------|----------|--------|----------------|-------|
| **Tone (high agg)** | "aggressive and confrontational" | llm_generator.py:97 | Personality description for aggression > 0.7 | Could add more intense descriptors | Currently might be too mild |
| **Tone (medium)** | "assertive and direct" | llm_generator.py:99 | aggression 0.4-0.7 | - | Balanced |
| **Tone (low)** | "polite and measured" | llm_generator.py:101 | aggression < 0.4 | - | May be why we have too much positivity |
| **Context shown** | Last 3 posts | llm_generator.py:108 | Thread context in prompt | 2-10 posts | More = better context, longer prompts |

**Aggression Thresholds:**
- Current: 0.7 (high), 0.4 (medium)
- Could lower to 0.6, 0.3 to make more agents "aggressive"

---

## 5. Agent DNA Distribution (Fixed - from real data)

| Attribute | Distribution | Source | Notes |
|-----------|-------------|--------|-------|
| Political Leaning | 55.6% Left, 43.2% Right, 1.2% Center | Real thread users | Fixed from actual data |
| Total Agents | 169 | Real thread users | Fixed (could subsample for faster testing) |
| Aggression Range | 0.0 - 2.0 (hate_score + offensive_score) | RoBERTa classification | Fixed from user DNA |

---

## 6. Unused ABM Parameters (Not currently applied)

These are in config but **not implemented** in thread simulation:

| Parameter | Current Value | Status | Notes |
|-----------|--------------|--------|-------|
| `lurker_ratio` | 0 | NOT USED | Future: spawn lurkers who don't post but track opinions |
| `bounded_confidence_threshold` | 0.3 | NOT USED | Future: opinion dynamics |
| `backfire_threshold` | 0.6 | NOT USED | Future: opinion dynamics |
| `backfire_aggression_min` | 0.7 | NOT USED | Future: opinion dynamics |

---

## 7. Priority Parameters to Tune (Ordered by Impact)

### 🔥 HIGH IMPACT - Tune These First

1. **System Prompt** (biggest lever)
   - **Current:** "Be authentic and conversational"
   - **Try:** "Be direct, blunt, and passionate. This is Twitter - argue your point forcefully"
   - **Expected:** +10-20% negative sentiment

2. **Model Size**
   - **Current:** dolphin-llama3:8b
   - **Try:** dolphin-llama3:70b
   - **Expected:** Better quality, more realistic tone (+5-10% negative)

3. **Temperature**
   - **Current:** 1.1
   - **Try:** 1.3 (more extreme), 0.9 (more consistent)
   - **Expected:** Higher temp → more varied/edgy responses

4. **Aggression Tone Threshold**
   - **Current:** "aggressive" if > 0.7
   - **Try:** Lower to > 0.5 or > 0.6
   - **Expected:** More agents use aggressive tone (+5-10% negative)

### 🟡 MEDIUM IMPACT

5. **Base Reply Probability**
   - **Current:** 5%
   - **Try:** 3% (sparser), 8% (denser)
   - **Expected:** Changes thread activity level

6. **Controversy Weight**
   - **Current:** 2.5x for aggressive seeking opposing views
   - **Try:** 3.5x or 4.0x
   - **Expected:** More cross-partisan conflict

7. **Context Window**
   - **Current:** Last 10 posts (but only 3 shown to LLM)
   - **Try:** Show 5-7 posts to LLM
   - **Expected:** Better coherence but slower

### 🟢 LOW IMPACT (Fine-tuning)

8. **Max Tokens**
   - **Current:** 150
   - **Try:** 100 (shorter), 200 (longer rants)

9. **Depth Penalty**
   - **Current:** 0.5x for depth > 5
   - **Try:** 0.3x (flatter threads)

10. **Max Rounds**
    - **Current:** 10
    - **Try:** 3 (quick test), 20 (longer simulation)

---

## Recommended Tuning Sequence

### Experiment 1: More Aggressive Prompt
```yaml
system_prompt: 'You are a Twitter user in a heated political argument. Be direct,
blunt, and emotionally charged. Don'\''t hold back - this is Twitter, not a civil
debate. Use strong language and argue your point passionately.'
```
**Expected:** 45-55% negative sentiment (vs current 39.8%)

### Experiment 2: Lower Aggression Threshold
```python
# In llm_generator.py line 96-101
if aggression > 0.5:  # Was 0.7
    tone = "aggressive and confrontational"
elif aggression > 0.3:  # Was 0.4
    tone = "assertive and direct"
```
**Expected:** More agents use aggressive tone → 50-60% negative

### Experiment 3: Upgrade to 70B Model
```yaml
model: dolphin-llama3:70b
```
**Expected:** Higher quality, more nuanced negativity → 55-65% negative

### Experiment 4: Combine All Three
- Aggressive prompt + Lower threshold + 70B model
**Expected:** 65-75% negative (close to real 78.8%)

---

## Current Parameter Settings (Baseline)

```yaml
# config/thread_config.yaml
llm:
  provider: ollama
  model: dolphin-llama3:8b
  temperature: 1.1
  max_tokens: 150
  system_prompt: "You are a Twitter user engaging in political discourse..."
```

```python
# sim/thread_simulation.py
base_prob = 0.05
aggression_boost = aggression * 0.15
reply_prob_max = 0.25
max_rounds = 10
thread_context_size = 10
recent_posts_pool = 50

# Target selection weights
recency_weight = 3.0
controversy_weight_high_agg = 2.5
controversy_weight_low_agg = 0.3
depth_penalty_threshold = 5
depth_penalty_weight = 0.5
```

```python
# sim/llm_generator.py
high_aggression_threshold = 0.7
medium_aggression_threshold = 0.4
context_posts_shown = 3  # Last 3 posts
```

**These settings produced:**
- 93.5% overall accuracy (EXCELLENT)
- 90.7% sentiment similarity
- 39.8% negative (vs target 78.8%)
- ~22 posts per round
- 186 total posts in 10 rounds

---

## Quick Test Script

Want to test parameter changes quickly? Run 3-round simulations:

```python
# sim/thread_simulation.py line 481
model.run(max_rounds=3)  # Instead of 10
```

Then validate:
```bash
python scripts/validate_thread_simulation.py \
    --real output/selected_thread_metadata.json \
    --simulated output/simulated_thread_metadata.json
```

Track JSD score - target is < 0.05 for excellent match.

---

## ⚠️ CRITICAL: Cross-Thread Generalization Testing

### Current Status: NOT GENERALIZABLE

**Problem:** All tuning and validation done on **one specific tweet**
- Thread: Pelosi/MTG Jan 6 recordings debate
- Characteristics: Highly partisan, 78.8% negative, political elites topic
- Risk: Parameters may be **overfit** to this specific thread

### What This Means:

**Parameters might fail on:**
- Less polarizing topics (e.g., policy discussions)
- More positive threads (e.g., celebration tweets)
- Non-political threads (e.g., sports, entertainment)
- Different political framings (economic vs social issues)

### Required for Dissertation Validity:

**Minimum: Test on 5-10 diverse threads**

**Recommended Thread Selection:**
1. **High Negativity** (like current): Trump indictment, election fraud claims
2. **Medium Negativity**: Infrastructure bill debate, Supreme Court ruling
3. **Low Negativity**: Bipartisan legislation, local politics
4. **Positive Thread**: Election victory celebration, policy win
5. **Non-Political**: Celebrity news, sports controversy

**For Each Thread:**
- Run simulation with current parameters
- Calculate JSD, sentiment similarity, overall accuracy
- Track which parameters need adjustment
- Compute mean ± std deviation across all threads

### Generalization Metrics:

**Success Criteria:**
- Mean accuracy across 10 threads: >80%
- Std deviation of JSD: <0.10 (consistent performance)
- Works on both positive and negative threads
- Parameter changes needed: <20% from baseline

**If Parameters Don't Generalize:**
- Need thread-type detection (positive/negative/neutral)
- Dynamic parameter adjustment based on root tweet sentiment
- Separate prompt templates for different thread types

### Next Steps:

1. **Select 10 diverse threads** from USC dataset using `scripts/find_best_thread.py`
2. **Run pipeline on each:**
   ```bash
   # For each thread
   python scripts/run_thread_pipeline.py  # Extract thread
   python sim/thread_simulation.py        # Simulate
   python scripts/validate_thread_simulation.py  # Validate
   ```
3. **Aggregate results** into cross-validation table
4. **Adjust parameters** if mean accuracy < 80%
5. **Document findings** in Implementation.md

### Expected Timeline:

- Thread selection: 1 hour
- 10 simulations @ 15 min each: 2.5 hours
- Analysis: 1 hour
- **Total: ~5 hours for full cross-validation**

### This Is Required Before:

- ❌ Claiming model "works" in general
- ❌ Publishing/defending dissertation
- ❌ Drawing conclusions about Twitter discourse patterns
- ✅ Can only claim: "Works on Pelosi/MTG thread with these specific characteristics"
