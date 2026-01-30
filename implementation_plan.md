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
2. Categorical Agent Architecture

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