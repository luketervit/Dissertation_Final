import pandas as pd
import numpy as np
import re
import torch
from transformers import pipeline
from tqdm import tqdm

# 1. SETUP MODELS (Local & Free)
device = 0 if torch.cuda.is_available() else -1
print(f"Using: {'GPU' if device == 0 else 'CPU'}")

# Loading the Distinction-level 5-model stack
# Note: These will download to your local cache (~2GB total)
stance_p  = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-stance-hillary", device=device)
emo_p     = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", device=device)
sent_p    = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)
hate_p    = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest", device=device)
offen_p   = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", device=device)

def get_agent_dna(text):
    """Extract stance, emotion, sentiment, hate, and offensive scores from text."""
    text = str(text)[:512]  # Transformers limit

    # Raw Model Calls
    s = stance_p(text)[0]
    e = emo_p(text)[0]
    sn = sent_p(text)[0]
    h = hate_p(text)[0]
    o = offen_p(text)[0]

    return {
        'stance_label': s['label'],   # AGAINST, NEUTRAL, FAVOR
        'stance_score': s['score'],   # Probability
        'emotion_label': e['label'],  # anger, joy, etc.
        'emotion_score': e['score'],
        'sentiment_label': sn['label'], # positive, neutral, negative
        'sentiment_score': sn['score'],
        'hate_score': h['score'] if h['label'] == 'HATE' else 1 - h['score'],
        'offensive_score': o['score'] if o['label'] == 'OFFENSIVE' else 1 - o['score']
    }

# 2. LOAD DATA
print("Loading tweets...")
df = pd.read_csv('data/may_july_chunk_1.csv')

# 3. EXTRACT USER IDs
print("Extracting user IDs...")
def extract_user_id(user_str):
    """Parse user dict and extract user_id using regex (handles datetime objects)."""
    try:
        # Extract 'id': <number> from the string using regex
        # Handles cases where ast.literal_eval fails due to datetime objects
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        if match:
            return int(match.group(1))
        return None
    except:
        return None

df['user_id'] = df['user'].apply(extract_user_id)
df = df.dropna(subset=['user_id'])  # Remove rows without valid user_id
df['user_id'] = df['user_id'].astype(int)

print(f"Processing {len(df)} tweets from {df['user_id'].nunique()} unique users...")

# 4. CLASSIFY EACH TWEET
tweet_results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying tweets"):
    # Get DNA for this tweet
    dna = get_agent_dna(row['text'])

    # Extract viewCount
    try:
        # Use regex to extract 'count': '<number>' from viewCount string
        # Note: count value is stored as a string, so we need to match the quotes
        match = re.search(r"'count':\s*'(\d+)'", str(row['viewCount']))
        view_count = int(match.group(1)) if match else 0
    except:
        view_count = 0

    # Store tweet-level results
    tweet_results.append({
        'user_id': row['user_id'],
        'tweet_id': row['id'],
        'view_count': view_count,
        'reply_count': row['replyCount'],
        **dna
    })

# Convert to DataFrame
tweets_df = pd.DataFrame(tweet_results)

# 5. AGGREGATE BY USER
print("\nAggregating to user-level profiles...")

def most_common(series):
    """Return the most common value in a series."""
    return series.mode()[0] if len(series.mode()) > 0 else series.iloc[0]

user_profiles = tweets_df.groupby('user_id').agg({
    # Count tweets per user
    'tweet_id': 'count',

    # Sum engagement metrics
    'view_count': 'sum',
    'reply_count': 'sum',

    # Average continuous scores
    'stance_score': 'mean',
    'emotion_score': 'mean',
    'sentiment_score': 'mean',
    'hate_score': 'mean',
    'offensive_score': 'mean',

    # Most common labels (user's dominant stance/emotion/sentiment)
    'stance_label': most_common,
    'emotion_label': most_common,
    'sentiment_label': most_common
}).reset_index()

# Rename tweet_id count to tweet_count
user_profiles.rename(columns={'tweet_id': 'tweet_count'}, inplace=True)

# Reorder columns for clarity
user_profiles = user_profiles[[
    'user_id', 'tweet_count', 'view_count', 'reply_count',
    'stance_label', 'stance_score',
    'emotion_label', 'emotion_score',
    'sentiment_label', 'sentiment_score',
    'hate_score', 'offensive_score'
]]

# 6. SAVE RESULTS
user_profiles.to_csv('output/processed_agents_raw.csv', index=False)

print(f"\n Sprint 1 Complete!")
print(f"   Processed: {len(df)} tweets")
print(f"   Unique users: {len(user_profiles)}")
print(f"   Output: 'output/processed_agents_raw.csv'")
print(f"   Avg tweets/user: {user_profiles['tweet_count'].mean():.1f}")
