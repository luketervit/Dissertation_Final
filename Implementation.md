# Implementation Notes

## Step 1: Agent Classification Pipeline (`step1_classify.py`)

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
   - **Mean** for continuous scores (stance, emotion, sentiment, hate, offensive)
   - **Mode** for categorical labels (dominant stance/emotion/sentiment)
   - **Sum** for engagement metrics (total views, total replies)
   - **Count** for tweet frequency

**Assumptions:**
- Users maintain consistent political stance across the observation window
- Averaging scores reduces noise from individual tweet variability
- A user's "true" stance is best captured by their aggregate behavior
- Missing viewCount values default to 0 (conservative estimate)

---

### Model Selection: 5-Model RoBERTa Stack

**Models Used:**
1. **Stance:** `cardiffnlp/twitter-roberta-base-stance-hillary`
2. **Emotion:** `cardiffnlp/twitter-roberta-base-emotion`
3. **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
4. **Hate:** `cardiffnlp/twitter-roberta-base-hate-latest`
5. **Offensive:** `cardiffnlp/twitter-roberta-base-offensive`

**Why CardiffNLP Models:**
- Trained specifically on Twitter data (~124M tweets, 2018-2021)
- Part of validated TweetEval benchmark (EMNLP 2020)
- Free, open-source, runs locally (no API costs)
- Widely cited in computational social science literature
- Outputs calibrated probability scores, not just labels

**Stance Model Rationale:**
- **Challenge:** No general "election stance" model exists in CardiffNLP suite
- **Solution:** Use `stance-hillary` as proxy for political stance detection
- **Justification:**
  - Detects FAVOR/AGAINST/NEUTRAL toward political figures
  - Captures partisan alignment better than generic sentiment
  - Validated on political Twitter data (SemEval 2016)
- **Alternative Considered:** Remove stance entirely and rely only on sentiment
- **Trade-off:** Hillary-specific model may not perfectly generalize to 2024 Trump/Biden discourse, but political stance patterns are relatively stable across election cycles

**Hate & Offensive Score Normalization:**
- Models output binary labels (HATE/NOT_HATE, OFFENSIVE/NOT_OFFENSIVE)
- We normalize to continuous [0,1] scale where 1 = maximum hate/offensive
- Formula: `score if label==HATE else 1-score`
- **Rationale:** ABM requires continuous aggression variable for Backfire Effect threshold

**Text Truncation:**
- All tweets truncated to 512 tokens (RoBERTa limit)
- **Assumption:** Political stance is conveyed in first 512 tokens; thread context ignored

---

### Output Schema: `processed_agents_raw.csv`

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique Twitter user identifier |
| `tweet_count` | int | Number of tweets posted by user |
| `view_count` | int | Total views across all user's tweets |
| `reply_count` | int | Total replies received |
| `stance_label` | str | Dominant stance (FAVOR/AGAINST/NEUTRAL) |
| `stance_score` | float | Mean stance confidence [0,1] |
| `emotion_label` | str | Dominant emotion (joy, anger, etc.) |
| `emotion_score` | float | Mean emotion confidence [0,1] |
| `sentiment_label` | str | Dominant sentiment (positive/negative/neutral) |
| `sentiment_score` | float | Mean sentiment confidence [0,1] |
| `hate_score` | float | Mean hate speech likelihood [0,1] |
| `offensive_score` | float | Mean offensive language likelihood [0,1] |

**Usage in ABM:**
- `stance_score` → Agent's latent opinion position
- `hate_score + offensive_score` → Aggression parameter for Backfire Effect
- `view_count` → Determines Lurker spawn ratio (90-9-1 Rule)
- `tweet_count` → Distinguishes Active (≥1 tweet) from Lurker (0 tweets) agents

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

### Known Limitations & Future Work

1. **Stance Model Mismatch:** Hillary-specific model may not capture Trump/Biden stances perfectly
   - **Mitigation:** Could fine-tune on 2024 election data if accuracy is poor
2. **No Temporal Dynamics:** Aggregation loses within-user opinion evolution
   - **Rationale:** ABM focuses on inter-user influence, not intra-user change
3. **Single-Pass Classification:** No ensemble or cross-validation
   - **Rationale:** Dissertation scope prioritizes simulation over ML optimization
4. **No Bot Detection:** Assumes all users are human
   - **Risk:** Bots may skew engagement metrics
   - **Assumption:** USC dataset likely pre-filtered for quality

---

### Validation Plan (Next Steps)

- **Distribution Check:** Verify stance_label distribution matches 2024 election polling data
- **Correlation Analysis:** Confirm hate_score + offensive_score correlate with negative sentiment
- **Sample Inspection:** Manually review high-aggression users for face validity
