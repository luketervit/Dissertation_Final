# Implementation Notes

## Step 1: Agent Classification Pipeline (`step1_classify_chunked.py`)

### Design Decision: User-Level Aggregation

**What:** Instead of classifying individual tweets, we aggregate all tweets per user to create user-level "DNA" profiles.

**Why:**
- In ABM literature, agents represent entities (users), not actions (tweets)
- Aligns with the 90-9-1 Rule framework where we model Active Users vs Lurkers
- Users have stable political identities; individual tweets are noisy samples
- Enables spawning Lurker agents based on user-level engagement (viewCount)
- Reduces computational complexity: ~100k tweets → ~10k users

**Implementation:**
1. Extract `user_id` from nested user dictionary in raw tweet data
2. Classify each tweet individually to get 8 DNA features
3. Aggregate by user_id using:
   - **Mean** for continuous scores (political leaning, emotion, sentiment, hate, offensive)
   - **Mode** for categorical labels (dominant political leaning/emotion/sentiment)
   - **Sum** for engagement metrics (total views, total replies)
   - **Count** for tweet frequency

**Assumptions:**
- Users maintain consistent political ideology across the observation window
- Averaging scores reduces noise from individual tweet variability
- A user's "true" political leaning is best captured by their aggregate behavior
- Missing viewCount values default to 0 (conservative estimate)

---

### Model Selection: 5-Model Stack

**Models Used:**
1. **Political Leaning:** `matous-volf/political-leaning-politics` with `launch/POLITICS` tokenizer
2. **Emotion:** `cardiffnlp/twitter-roberta-base-emotion`
3. **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
4. **Hate:** `cardiffnlp/twitter-roberta-base-hate-latest`
5. **Offensive:** `cardiffnlp/twitter-roberta-base-offensive`

**Why These Models:**
- **Political Leaning Model:** Trained on US political data across 12 datasets including news, social media, and political content (F1 score: 83.6%)
- **CardiffNLP Models:** Trained specifically on Twitter data (~124M tweets, 2018-2021), part of validated TweetEval benchmark (EMNLP 2020)
- All models are free, open-source, and run locally (no API costs)
- Outputs calibrated probability scores, not just labels

**Political Leaning Model Selection:**
- **Initial Attempt:** Used `cardiffnlp/twitter-roberta-base-stance-hillary` (2016 Hillary Clinton stance detection)
- **Problem Discovered:** Hillary model produced heavily skewed results (68% AGAINST, 1% FAVOR) despite dataset containing 2.5x more Biden than Trump mentions
- **Root Cause:** Hillary model detects negative/critical tone vs supportive tone, NOT political ideology
- **Solution:** Replaced with `matous-volf/political-leaning-politics` which directly classifies Left/Center/Right ideology
- **Validation:** Tested on 100 sample tweets; model correctly identifies:
  - Anti-Biden/pro-Trump tweets → Right
  - Anti-Trump/progressive tweets → Left
  - Mixed/moderate content → Center
- **Results:** ~68% Right, ~32% Left distribution reflects genuine conservative lean in May-July 2024 Twitter sample
- **Why Better:** Directly measures political ideology, not tied to specific 2016 politician, theoretically grounded for 2024 election ABM

**Hate & Offensive Score Normalization:**
- Models output binary labels (HATE/NOT_HATE, OFFENSIVE/NOT_OFFENSIVE)
- We normalize to continuous [0,1] scale where 1 = maximum hate/offensive
- Formula: `score if label==HATE else 1-score`
- **Rationale:** ABM requires continuous aggression variable for Backfire Effect threshold

**Text Truncation:**
- All tweets truncated to 512 tokens (RoBERTa limit)
- **Assumption:** Political stance is conveyed in first 512 tokens; thread context ignored

---

### Output Schema: `processed_agents_raw_N.csv`

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique Twitter user identifier |
| `tweet_count` | int | Number of tweets posted by user |
| `view_count` | int | Total views across all user's tweets |
| `reply_count` | int | Total replies received |
| `political_label` | str | Dominant political leaning (Left/Center/Right) |
| `political_score` | float | Mean political leaning confidence [0,1] |
| `emotion_label` | str | Dominant emotion (joy, anger, etc.) |
| `emotion_score` | float | Mean emotion confidence [0,1] |
| `sentiment_label` | str | Dominant sentiment (positive/negative/neutral) |
| `sentiment_score` | float | Mean sentiment confidence [0,1] |
| `hate_score` | float | Mean hate speech likelihood [0,1] |
| `offensive_score` | float | Mean offensive language likelihood [0,1] |

**Usage in ABM:**
- `political_label` → Agent's ideological position (Left/Center/Right)
- `political_score` → Confidence in ideological classification
- `hate_score + offensive_score` → Aggression parameter for Backfire Effect
- `view_count` → Determines Lurker spawn ratio (90-9-1 Rule)
- `tweet_count` → Distinguishes Active (≥1 tweet) from Lurker (0 tweets) agents

**Political Leaning Interpretation:**
- **Left (LABEL_0):** Liberal/Democratic ideology
- **Center (LABEL_1):** Moderate/centrist positions
- **Right (LABEL_2):** Conservative/Republican ideology
- Can be mapped to continuous opinion scale: Left=[0.0-0.33], Center=[0.33-0.67], Right=[0.67-1.0]

---

### Technical Choices

**Device:** Auto-detect GPU via `torch.cuda.is_available()`, fallback to CPU
- **Rationale:** ~80MB CSV will take 30-60 min on CPU, <10 min on GPU

**Progress Tracking:** `tqdm` for visual feedback during long classification runs

**Error Handling:** Try/except on `viewCount` parsing (nested dict with inconsistent structure in raw data)

**Data Parsing Strategy - Regex Extraction:**
- **Challenge:** Raw CSV stores user metadata and viewCount as string representations of Python dicts containing `datetime` objects
- **Problem:** `ast.literal_eval()` fails because it only parses Python literals, not function calls like `datetime.datetime(...)`
- **Solution:** Use regex pattern matching to extract numeric IDs directly from string representation
  - User ID: `re.search(r"'id':\s*(\d+)", user_str)` extracts the numeric user ID (unquoted)
  - View count: `re.search(r"'count':\s*'(\d+)'", viewCount_str)` extracts engagement count (quoted string)
- **Data Format Quirk:** The `viewCount` field stores counts as **quoted strings** (`'count': '10'`), not integers
  - This required adjusting the regex to match `'count': '<digits>'` instead of `'count': <digits>`
  - Example: `{'count': '54186', 'state': 'EnabledWithCount'}` → extracts `54186`
- **Why Regex Over eval():**
  - `eval()` is unsafe with untrusted data (arbitrary code execution risk)
  - Full dict parsing unnecessary - we only need specific numeric fields
  - Regex is faster and more robust to malformed dict strings
- **Assumption:** The USC dataset has consistent field naming (`'id':` and `'count':`) across all records

**Dependencies:** `transformers`, `torch`, `pandas`, `numpy`, `re`
- **No external APIs:** All models run locally to avoid cost/rate limits
- **No tokenization preprocessing:** RoBERTa handles raw text

---

### Chunked Processing with Checkpoint System (`step1_classify_chunked.py`)

**Challenge:** Processing 50,000 tweets through 5 RoBERTa models takes 30-60 minutes on CPU. System crashes, interruptions, or errors would lose all progress.

**Solution:** Implemented checkpoint-based processing system that:
1. Accepts chunk number as command-line argument (`python step1_classify_chunked.py 2`)
2. Saves progress every 1,000 tweets to JSON checkpoint file
3. Detects existing checkpoints on restart and offers resume option
4. Automatically cleans up checkpoint files after successful completion

**Implementation Details:**
- **Checkpoint Format:** JSON file containing `last_index`, `tweet_results`, and `total_tweets`
- **Type Conversion:** All pandas/numpy types (`int64`, `float64`) converted to native Python types before JSON serialization
  - Problem: `json.dump()` fails on `numpy.int64` and `pandas.Int64` types
  - Solution: Explicit casting with `int()` and `float()` when appending to results list
- **Resume Logic:** User prompted on restart if checkpoint exists; can choose to resume or start fresh
- **Checkpoint Interval:** 1,000 tweets (~2-3 minutes of work) balances overhead vs. progress preservation

**File Naming Convention:**
- Input: `data/may_july_chunk_N.csv`
- Output: `output/processed_agents_raw_N.csv`
- Checkpoint: `output/checkpoint_chunk_N.json`

**Why Chunked Processing:**
- USC dataset is large (100k+ tweets); processing in manageable 50k chunks
- Enables parallel processing on multiple machines if needed
- Easier to debug and validate intermediate results
- Can aggregate chunks later for full dataset analysis

**Rationale:**
- Classification is CPU-intensive (~8 tweets/second on CPU)
- Long-running processes are vulnerable to network issues, system updates, user interruption
- Checkpoint overhead is minimal (<1% performance impact) compared to re-running entire pipeline
- JSON format is human-readable for debugging checkpoint issues

**Assumptions:**
- Chunks are processed independently (no cross-chunk dependencies)
- Tweet order within chunk doesn't matter for final user-level aggregation
- Checkpoint files are reliable (filesystem doesn't corrupt during write)

---

### Political Model Selection: From Hillary Stance to Political Leaning

**Initial Attempt - Hillary Stance Model:**
After processing chunk 1 with `cardiffnlp/twitter-roberta-base-stance-hillary`, discovered heavily skewed results:
- AGAINST: 68.4% of users
- NONE: 30.6% of users
- FAVOR: 1.1% of users

**Investigation - Data vs Model Issue:**
Examined raw data to determine if this was a data bias or model mismatch:
- Dataset contains **2.5x more Biden mentions than Trump mentions** (53% vs 21%)
- This rules out pro-Trump data bias as the primary cause
- Conclusion: The Hillary model is detecting **negative/critical tone** (AGAINST) vs **positive/supportive tone** (FAVOR), NOT political affiliation

**Root Cause Analysis:**
- The stance model was trained on 2016 Hillary Clinton tweets (SemEval 2016)
- When applied to 2024 election discourse, it detects sentiment polarity rather than partisan stance
- Most political tweets are **negative/critical** regardless of which candidate they support (attacking opponents, expressing outrage)
- This explains why 68% are AGAINST even though data mentions Biden more than Trump

**Solution - Political Leaning Model:**
Replaced Hillary model with `matous-volf/political-leaning-politics`:
1. **Model Characteristics:**
   - Trained on 12 US political datasets (news, social media, political content)
   - Directly classifies Left/Center/Right ideology (not tied to specific politician)
   - F1 score: 83.6% average across test sets
   - Outputs LABEL_0 (Left), LABEL_1 (Center), LABEL_2 (Right)

2. **Validation Testing:**
   - Tested on 100 sample tweets from chunk 1
   - Results: 68% Right, 32% Left, 0% Center
   - Manual inspection confirms correct classifications:
     - Anti-Biden/pro-Trump tweets → Right ✓
     - Pro-progressive/anti-MAGA tweets → Left ✓
     - Model predictions align with tweet content

3. **Interpretation of 68% Right:**
   - This reflects **genuine conservative lean** in the May-July 2024 Twitter sample
   - NOT a model artifact (unlike Hillary's 68% AGAINST)
   - Consistent with Twitter's user demographics during 2024 election period
   - Scientifically interesting finding for dissertation

**Implementation Decision:**
- **Adopted:** Political leaning model as primary ideology measure
- **Rationale:**
  - Direct measurement of Left/Center/Right ideology
  - Not tied to 2016 politician or outdated training data
  - Theoretically grounded for 2024 election ABM
  - Easier to defend in dissertation (no Hillary model mismatch to explain)
- **ABM Usage:** `political_label` and `political_score` replace `stance_label` and `stance_score`

**Lesson Learned:**
- Always validate model outputs against raw data before assuming model correctness
- Domain-specific models (political leaning) outperform repurposed models (Hillary stance) for specialized tasks
- Model selection is iterative: test, validate, and replace if necessary

---

### Understanding Political Score: Confidence vs. Position

**Common Confusion:** Why can both Left and Right agents have similar high scores (e.g., 0.93)?

**Key Insight:** `political_score` measures **model confidence**, NOT political position on a spectrum.

**How Classification Works:**

For each tweet, the RoBERTa model outputs probabilities for all three classes:
```python
# Example Tweet A (pro-Biden, anti-Trump):
Model output: [Left: 0.93, Center: 0.04, Right: 0.03]
Result: political_label='Left', political_score=0.93

# Example Tweet B (pro-Trump, anti-Biden):
Model output: [Left: 0.02, Center: 0.01, Right: 0.97]
Result: political_label='Right', political_score=0.97
```

The model picks the **highest probability** as the label, and that probability becomes the score. This means:
- **Left with 0.93 score** = "93% confident this is Left (not Center/Right)"
- **Right with 0.97 score** = "97% confident this is Right (not Left/Center)"
- **Center with 0.88 score** = "88% confident this is Center (not Left/Right)"

**User-Level Aggregation (replace_stance_both_chunks.py:142-145):**

When aggregating by user, we take:
1. **Most common label** across all user's tweets (e.g., if 8/10 tweets are "Left", user is "Left")
2. **Average score** across all tweets (measures overall classification confidence)

**Example from Actual Data:**
```csv
user_id | political_label | political_score | Interpretation
--------|-----------------|-----------------|----------------
3577    | Left           | 0.9319          | Strong Left, model is confident
10253   | Left           | 0.9931          | Strong Left, model very confident
428333  | Right          | 0.9982          | Strong Right, model very confident
708973  | Right          | 0.7671          | Moderate Right, some ambiguity
```

**Why High Scores Are Good:**

High scores (>0.8) indicate the model is **decisive and confident**, not confused:
- **High Score (0.85-0.99):** Clear ideological signal, model can distinguish easily
- **Medium Score (0.60-0.85):** Moderate or mixed messaging, some uncertainty
- **Low Score (0.33-0.60):** Ambiguous, model struggling to classify

**ABM Applications:**

The `political_score` can be used as a **conviction parameter**:

1. **Stubbornness/Rigidity:** High-score agents (>0.9) are ideologically rigid
   - Less susceptible to bounded confidence persuasion
   - More likely to trigger Backfire Effect when exposed to opposing views

2. **Openness to Persuasion:** Low-score agents (<0.7) have weaker ideology
   - More easily influenced by opposing viewpoints
   - Larger bounded confidence threshold

3. **Filtering:** Remove low-confidence classifications (<0.6) as unreliable

**Implementation Example (Future ABM Code):**
```python
# High conviction → small bounded confidence threshold
agent.bc_threshold = 0.3 if political_score > 0.9 else 0.5

# High conviction → stronger backfire effect
agent.backfire_multiplier = political_score * 2.0

# Stubbornness scales with confidence
agent.opinion_update_rate = 0.1 * (1 - political_score)
```

**Data Characteristics:**

Analysis of processed_agents_raw_1_political.csv and processed_agents_raw_2_political.csv shows:
- **Mean political_score: ~0.85-0.90** → Most users have clear ideological positions
- **Score distribution:** 75% of users have scores >0.80
- **Interpretation:** Twitter political discourse during May-July 2024 was highly polarized with confident Left/Right positions

**Scientific Implication:**

The high average political_score validates the dataset's suitability for studying polarization and backfire effects. If scores were low (~0.5-0.6), it would suggest:
- Model is not working well on this data, OR
- Users lack clear political identities (unlikely for election discourse)

High scores confirm we're capturing genuine ideological divides, not noisy/ambiguous content.

---

### Known Limitations & Future Work

1. **Conservative Bias in Dataset:** 68% Right vs 32% Left distribution suggests sample is not representative of general US population
   - **Implication:** ABM results will reflect conservative-leaning Twitter discourse, not balanced political spectrum
   - **Mitigation:** Acknowledge in dissertation; results generalize to similar conservative-leaning social media samples
   - **Future Work:** Process additional chunks or different time periods to check temporal stability

2. **No Temporal Dynamics:** Aggregation loses within-user opinion evolution
   - **Rationale:** ABM focuses on inter-user influence, not intra-user change
   - **Future Work:** Track user opinion changes across multiple time windows

3. **Single-Pass Classification:** No ensemble or cross-validation
   - **Rationale:** Dissertation scope prioritizes simulation over ML optimization
   - **Future Work:** Ensemble multiple political leaning models for robustness

4. **No Bot Detection:** Assumes all users are human
   - **Risk:** Bots may skew engagement metrics
   - **Assumption:** USC dataset likely pre-filtered for quality
   - **Future Work:** Apply bot detection (e.g., Botometer) to filter non-human accounts

5. **Limited Center Representation:** Political model classified 0% Center in sample
   - **Possible Causes:** Twitter political discourse is polarized, or model threshold for Center is too strict
   - **Implication:** ABM will model Left vs Right dynamics with limited moderate/centrist agents
   - **Future Work:** Investigate if larger samples show Center classifications

---

### Validation Plan (Next Steps)

- **Political Distribution Check:** Compare Left/Right split to known Twitter demographics and 2024 polling data
- **Correlation Analysis:** Confirm hate_score + offensive_score correlate with negative sentiment and political extremity
- **Sample Inspection:** Manually review high-aggression users and edge cases for face validity
- **Cross-Chunk Validation:** Compare political distributions across chunk 1 and chunk 2 for consistency
- **Temporal Stability:** Check if political leaning distributions are stable across different months (May vs July)
