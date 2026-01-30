Implementation Plan: Historical Replay & Cross-Model Validation

## 1. Data Selection & Anchoring (COMPLETED)

Instead of arbitrary time splits, we anchor the simulation to a specific High-Density Event.

### Target Selection: **COMPLETED** (`scripts/find_best_thread.py`)

**Selected Thread:**
- **Root Tweet ID:** 1801027119356526704
- **Conversation ID:** 1801027119356526704
- **Root User ID:** 133028836
- **Timestamp:** 2024-06-12 23:01:41
- **Topic:** Senate Republicans blocking Supreme Court ethics rules
- **Expected Replies:** 8,095
- **View Count:** 450,132
- **Actual Replies in Dataset:** 4 (0.05% coverage)

**Why This Thread:**
- Highest engagement in chunk 1 (8,095 expected replies)
- Politically polarizing topic (Supreme Court, MAGA, ethics)
- Posted at 23:01, we observe first 34 minutes of replies (23:07-23:41)

### Timeline Extraction: **COMPLETED**

Thread timeline spans 34 minutes:
- Root tweet: 2024-06-12 23:01:41
- First reply: 2024-06-12 23:07:27 (+6 min)
- Last reply: 2024-06-12 23:41:38 (+40 min)

All tweets sorted by epoch (timestamp) in `output/best_thread_tweets.csv`

### Agent Identification: **COMPLETED**

**Actives (Users Who Actually Replied):**
- 4 active users with user_ids extracted
- DNA available in `processed_agents_raw_1_political.csv` for classification matching

**Lurkers:**
- N = (root_view_count - actual_replies - 1)
- N = (450,132 - 4 - 1) = **450,127 lurker agents**
- Will spawn using Pew 2024 political distributions

**Total Agents:** 450,132 (1 root author + 4 actives + 450,127 lurkers)

---

## Config-Driven Pipeline System (COMPLETED)

**What:** Automated the entire thread extraction pipeline with YAML configuration.

**Files:**
- `config/thread_config.yaml` - Editable parameters
- `scripts/run_thread_pipeline.py` - Single-command execution

**Usage:**
```bash
# Edit config to select thread
vim config/thread_config.yaml

# Run complete pipeline
python scripts/run_thread_pipeline.py

# Output files auto-generated:
# - selected_thread_metadata.json (temporal events)
# - selected_thread_tweets.csv
# - agents_for_tweet.csv (DNA profiles)
```

**Key Features:**
1. **Auto-selection:** Can find best thread automatically
2. **Temporal events:** Sorted timeline for minute-by-minute replay
3. **Lurker distribution:** Configurable (match_active_agents or pew_2024)
4. **Reproducibility:** Config tracks all parameters + auto-updates with metadata

**Current Thread (Final Selection):**
- Tweet: 1801016461601001478 (Pelosi/MTG Jan 6 debate)
- Replies: 183 actual (was 4 in Warren tweet)
- Users: 169 unique (100% DNA coverage)
- Duration: 2h 29min (8,968 seconds)
- Political: 55.6% Left, 43.2% Right, 1.2% Center
- Lurker strategy: match_active_agents

---

## LLM-Powered Thread Simulation (COMPLETED)

**What:** Implemented Mesa-based ABM where agents generate authentic text responses using Claude Haiku API, simulating realistic conversation dynamics within a Twitter thread.

**Why This Approach:**
- Initial opinion dynamics ABM had backwards causality issue (agents updating opinions after posting)
- Template-based responses were too generic and repetitive
- LLM integration enables context-aware, persona-driven conversational realism
- Aligns with dissertation goal: understand how political discourse unfolds in real-time

**Architecture:**

### Two-Stage Activation System
Mesa 3.x doesn't support `StagedActivation`, so implemented manual staging:

**Stage 1 (step):** All agents read thread, decide if/how to reply, generate text
**Stage 2 (advance):** All agents commit replies simultaneously to thread history

This prevents race conditions where early agents see later agents' posts from same round.

### Agent Behavior Logic

**ThreadAgent DNA:**
- `political_label` (Left/Center/Right)
- `political_score` (confidence)
- `emotion_label` (joy, anger, etc.)
- `aggression` = `hate_score + offensive_score`

**Reply Probability (Per Round):**
```python
base_prob = 0.05  # 5% baseline
aggression_boost = self.aggression * 0.15
reply_prob = min(0.25, base_prob + aggression_boost)  # Max 25%
```

Aggressive agents reply more frequently, moderate agents are selective.

**Target Selection (Weighted Sampling):**
- **Recency:** Posts from current/previous round get 3x weight
- **Controversy:** Opposing political views boost weight (especially for aggressive agents)
- **Depth Penalty:** Posts at depth >5 get 0.5x weight to avoid super-deep threads

### LLM Integration

**Provider:** Anthropic Claude Haiku (`claude-haiku-4-5-20251001`)

**Rationale:**
- Fast: ~1-2s per generation
- Cheap: $0.25/$1.25 per million tokens (in/out)
- High-quality: Contextually aware, persona-driven responses
- Cost estimate: $0.10-0.30 per 10-round simulation

**Configuration (`config/thread_config.yaml`):**
```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY
  temperature: 0.8  # Balance creativity vs consistency
  max_tokens: 150   # Twitter-length replies
```

**API Key Management:**
- Stored in `.env` file at project root (gitignored)
- Loaded via `python-dotenv` in `sim/llm_generator.py`
- Never hardcoded in source code

### Prompt Engineering

**System Prompt:**
```
You are a Twitter user engaging in political discourse. You have your own
political views and emotional tendencies. Generate realistic, concise replies
that reflect your persona while staying on-topic and authentic to how real
people argue online.
```

**User Prompt Structure:**
```
Your persona:
- Political leaning: {Left/Right/Center}
- Aggression level: {0.0-1.0}
- Current emotion: {joy/anger/etc.}

Thread context (last 10 posts):
[Recent conversation history...]

You are replying to:
"{target_post_text}"

Generate a short, authentic Twitter reply (1-3 sentences) that reflects
your political views and emotional state. Be opinionated but realistic.
```

**Why This Works:**
- Combines agent personality with thread context
- Short context window (10 posts) keeps costs low
- Explicit persona description guides tone and content
- Encourages realistic disagreement without toxic templates

### Output Format Matching

**Goal:** Simulation output matches `selected_thread_metadata.json` structure for interoperability.

**Key Sections:**
1. `root_tweet` - Base tweet that starts the thread
2. `replies` - All generated responses with IDs, user IDs, timestamps
3. `temporal_events` - Chronologically sorted array with:
   - `seconds_since_start` (round * 60)
   - `is_root` flag
   - Full text content
4. `simulation_info` - LLM provider, model, agent counts, max depth

**Files Generated:**
- `output/simulated_thread_metadata.json` - Full structured output
- `output/thread_history.json` - Simple post array for debugging
- `output/summary_stats.txt` - Political distribution, aggression stats
- `output/simulation_timeseries.csv` - Round-by-round metrics

### Implementation Files

**Core Modules:**
- `sim/thread_simulation.py` - ThreadAgent, ThreadModel, staging logic
- `sim/llm_generator.py` - Multi-provider LLM wrapper (Anthropic/OpenAI/Ollama)
- `config/thread_config.yaml` - Central configuration
- `.env` - API keys (not committed)

**Usage:**
```bash
# Setup
cp .env.example .env
# Add ANTHROPIC_API_KEY=your-key-here to .env

# Run simulation
python sim/thread_simulation.py

# Output appears in output/
```

### Mesa 3.x Compatibility Fixes

**Problem 1:** `ModuleNotFoundError: No module named 'mesa.time'`
**Fix:** Removed `BaseScheduler`, use `Model.agents` directly + manual staging

**Problem 2:** `Agent.__init__()` signature changed
**Fix:** Changed from `super().__init__(unique_id, model)` to `super().__init__(model)`, set `self.unique_id` manually

**Problem 3:** No `StagedActivation` in Mesa 3.x
**Fix:** Implemented manual two-stage loop in `ThreadModel.step()`:
```python
def step(self):
    for agent in self.agent_list:
        agent.step()  # Stage 1: Generate
    for agent in self.agent_list:
        agent.advance()  # Stage 2: Commit
```

### Validation Approach

**Political Distribution Check:**
- Compare simulated Left/Right ratio to input agent distribution
- Verify reply political balance matches active agent demographics

**Engagement Metrics:**
- Track replies per round, max depth, political polarization over time
- Compare to real thread patterns (if available)

**Content Quality (Manual):**
- Sample 20-30 generated replies
- Check for: on-topic responses, persona consistency, conversational coherence
- Flag any generic/nonsensical outputs

**Cost Tracking:**
- Log total API calls, estimated token usage
- Verify cost estimates align with actual usage

### Known Limitations

1. **No Real Timestamps:** Uses `round * 60` seconds as synthetic timeline (future: map rounds to actual minutes)
2. **Fixed Reply Probability:** 5-25% may not match real engagement curves (future: dynamic probability based on thread virality)
3. **Limited Context Window:** Only last 10 posts provided to LLM (tradeoff: cost vs. accuracy)
4. **No Lurker Activation:** Current implementation only simulates active agent conversations (future: integrate with lurker opinion dynamics ABM)
5. **Single-Thread Focus:** Doesn't model cross-thread influence or user history persistence
6. **API Rate Limits:** Anthropic has rate limits; large simulations (500+ agents, 50+ rounds) may need batching
7. **Model Determinism:** Temperature=0.8 means repeated runs produce different outputs (reproducibility requires seed control)

### Next Steps for Integration

**Phase 1: Standalone Thread Simulation** (COMPLETED)
- ✓ LLM-powered agent responses
- ✓ Staged activation preventing race conditions
- ✓ Output format matching real thread metadata

**Phase 2: Opinion Dynamics Integration** (FUTURE)
- Combine thread simulation with lurker opinion shift model
- Active agents post LLM-generated text → Lurkers update latent opinions
- Track "Ghost Shift" in silent majority based on simulated discourse

**Phase 3: Historical Replay Validation** (FUTURE)
- Use real thread as Phase 1 playback
- Simulate Phase 2 continuation with LLM agents
- Compare simulated activity patterns to real Phase 2 data

**Phase 4: Cross-Model Validation** (FUTURE)
- Extract thread signatures from simulated output
- Train classifier on real vs. simulated discourse
- Target: >80% "real thread" classification confidence

### IMPORTANT: Current Status & Future Refinements

**Current Implementation Status:** PROTOTYPE/MVP for testing LLM integration approach

**Known Limitations Requiring Redesign:**

1. **Timeline Mismatch:**
   - Current: Simulates in abstract "rounds" (10 rounds = 10 minutes simulated time)
   - Needed: Real 24-hour timeline matching actual thread duration
   - Issue: Reply probability and agent activation need temporal calibration

2. **Missing Lurker Integration:**
   - Current: Only active agents participate (no lurker opinion dynamics)
   - Needed: 90-9-1 Rule implementation with lurker agents tracking latent opinion shift
   - Issue: This is the core dissertation question (Ghost Shift) but not yet implemented

3. **Model Selection Not Finalized:**
   - Current: Using Claude Haiku (Anthropic API, paid)
   - Needed: Test local models (Llama 3, Mistral, etc.) for cost/quality tradeoff
   - Issue: API costs scale with simulation size; local inference may be required for 100k+ lurker simulations

4. **Agent Behavior Needs Calibration:**
   - Current: 5-25% reply probability is arbitrary
   - Needed: Calibrate against real engagement patterns in USC dataset
   - Issue: May need dynamic probability based on thread virality, time decay, etc.

**Next Implementation Phase:**
- Integrate lurker opinion dynamics from `sim/agents.py` with thread simulation
- Implement 24-hour timeline replay matching real thread timestamps
- Test local LLM alternatives (Ollama + Llama 3.3 70B or Mistral Large)
- Calibrate agent parameters against real thread engagement metrics
- Run sensitivity analysis on reply probability, aggression thresholds, context window size

---

## 2. Categorical Agent Architecture

We will not use a single float. We will use raw categorical labels and confidence scores to drive behavior.

Active Agent DNA

Political Identity: political_label (e.g., Left, Right, Neutral)

Certainty: political_score (0.0–1.0)

Style:

emotion_label

aggression_score = hate_score + offensive_score

Lurker Agent DNA

Assigned Identity
Seeded using Pew 2024 distributions (e.g., 52% Democrat label).

Silence Buffer
A threshold value: expressive_pressure

2.5. Ephemeral Agent Memory (Single-Simulation Only)

Agents maintain short-term, decaying memory that exists only within a single historical replay.
All memory is reset between simulations and does not persist across threads.

Each agent maintains the following internal state:

Memory State Variables

exposure_log
A rolling window of the last 
𝐾
K viewed tweets, storing:

tweet.political_label

tweet.sentiment_label

tweet.hate_score

emotional_residue
A scalar representing accumulated emotional impact:

Increases with high hate_score or anger

Decays over time

belief_inertia
A resistance term derived from political_score:

High certainty → slower belief response

Low certainty → faster reinforcement or backfire

Memory Update Rules (Per Viewed Tweet)

Aligned Exposure
If agent.label == tweet.label:

Decrease emotional_residue

Slightly increase belief_inertia (hardening)

Oppositional Exposure
If agent.label != tweet.label:

Increase emotional_residue proportional to tweet.hate_score

Increase expressive_pressure faster if recent exposure_log is opposition-heavy

Memory Decay

At each timestep:

emotional_residue *= decay_factor (e.g., 0.9)

Drop exposure_log entries older than 
𝐾
K events

This ensures agents respond to recent discourse patterns, not full-history accumulation.

Role in Activation & Reply Generation

expressive_pressure is computed from:

current tweet features

emotional_residue

alignment balance in exposure_log

When a Lurker converts to Active:

Their synthetic reply is conditioned on:

dominant emotion in exposure_log

peak emotional_residue at conversion time

3. The “Historical Replay” Simulation Loop

The simulation steps through the exact timestamps of the original thread.

T-Step (Minute by Minute)

If the CSV has a real tweet at this minute:

Inject that tweet into the simulation “Wall”

All Lurkers and “Future” Actives view this tweet

Interaction Rules (Categorical)

Alignment
If agent.label == tweet.label
→ reinforce opinion (Certainty +5%)

Backfire
If agent.label != tweet.label AND tweet.hate_score > 0.7
→ increase expressive_pressure

The “Conversion” Moment

If a Lurker’s expressive_pressure > 10:

Change status to Active

Generate a Synthetic Reply based on their DNA and memory state

4. Comparison & Validation Model

This is the second classification layer used to validate simulation success.

Feature Extraction

Compute a Thread Signature for both datasets:

S1 (Real Thread)
[Mean Sentiment, % Anger, % Offensive, Total Volume, Stance Variance]

S2 (Simulated Thread)
[Mean Sentiment, % Anger, % Offensive, Total Volume, Stance Variance]

Cross-Model Classifier

Use Random Forest or XGBoost

Perform a behavioral “Turing Test”

Training
Train the classifier to distinguish between different real threads from the USC-X-24 dataset.

Testing
Feed the classifier the Simulated Thread signature.

Success Metric
If the classifier labels the Simulated Thread as
“Real Election Discourse” with >80% confidence, the simulation logic is validated.