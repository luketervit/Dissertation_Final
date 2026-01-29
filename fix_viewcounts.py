"""
Quick fix script to update view_count in processed_agents_raw.csv
without re-running the entire classification pipeline.
"""
import pandas as pd
import re

print("Loading raw tweet data...")
df = pd.read_csv('data/may_july_chunk_1.csv')

print("Extracting user IDs and view counts...")
def extract_user_id(user_str):
    """Extract user_id using regex."""
    try:
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        if match:
            return int(match.group(1))
        return None
    except:
        return None

def extract_view_count(vc_str):
    """Extract view count using regex (handles quoted strings)."""
    try:
        match = re.search(r"'count':\s*'(\d+)'", str(vc_str))
        return int(match.group(1)) if match else 0
    except:
        return 0

# Extract user_id and view_count
df['user_id'] = df['user'].apply(extract_user_id)
df['view_count'] = df['viewCount'].apply(extract_view_count)

# Remove rows without valid user_id
df = df.dropna(subset=['user_id'])
df['user_id'] = df['user_id'].astype(int)

print(f"Processing {len(df)} tweets from {df['user_id'].nunique()} unique users...")

# Aggregate view counts by user
print("Aggregating view counts by user...")
view_counts = df.groupby('user_id')['view_count'].sum().reset_index()

print(f"\nView count statistics:")
print(view_counts['view_count'].describe())

# Load existing processed agents file
print("\nLoading existing processed_agents_raw.csv...")
agents = pd.read_csv('output/processed_agents_raw.csv')

print(f"Before merge: {(agents['view_count'] > 0).sum()} users with non-zero view counts")

# Merge view counts (update existing view_count column)
agents = agents.drop(columns=['view_count'])  # Remove old column
agents = agents.merge(view_counts, on='user_id', how='left')
agents['view_count'] = agents['view_count'].fillna(0).astype(int)  # Fill missing with 0

print(f"After merge: {(agents['view_count'] > 0).sum()} users with non-zero view counts")

# Reorder columns to match original format
agents = agents[[
    'user_id', 'tweet_count', 'view_count', 'reply_count',
    'stance_label', 'stance_score',
    'emotion_label', 'emotion_score',
    'sentiment_label', 'sentiment_score',
    'hate_score', 'offensive_score'
]]

# Save updated file
print("\nSaving updated file...")
agents.to_csv('output/processed_agents_raw.csv', index=False)

print(f"\n✓ View counts updated successfully!")
print(f"  Total users: {len(agents)}")
print(f"  Users with views: {(agents['view_count'] > 0).sum()}")
print(f"  Total views: {agents['view_count'].sum():,}")
print(f"  Avg views/user: {agents['view_count'].mean():.1f}")
print(f"  Max views: {agents['view_count'].max():,}")
