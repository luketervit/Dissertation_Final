"""
Extract a specific conversation thread for ABM Historical Replay.
Extracts the thread with the most actual replies in our dataset.
"""
import pandas as pd
import json
import re
from datetime import datetime

# Target tweet with most replies
TARGET_TWEET_ID = 1801016461601001478

print("="*80)
print("EXTRACTING THREAD WITH MOST ACTUAL REPLIES")
print("="*80)

# Load data
print("\nLoading chunk 1 data...")
df = pd.read_csv('data/may_july_chunk_1.csv')
df['datetime'] = pd.to_datetime(df['epoch'], unit='s')

# Extract view count helper
def extract_view_count(viewCount_str):
    try:
        match = re.search(r"'count':\s*'(\d+)'", str(viewCount_str))
        return int(match.group(1)) if match else 0
    except:
        return 0

df['views'] = df['viewCount'].apply(extract_view_count)

# Extract user ID helper
def extract_user_id(user_str):
    try:
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        return int(match.group(1)) if match else None
    except:
        return None

# Get root tweet
root_tweet = df[df['id'] == TARGET_TWEET_ID].iloc[0]
conversation_id = root_tweet['conversationId']

print(f"✓ Found root tweet: {TARGET_TWEET_ID}")
print(f"✓ Conversation ID: {conversation_id}")

# Get all tweets in this conversation
thread_tweets = df[df['conversationId'] == conversation_id].copy()
thread_tweets = thread_tweets.sort_values('datetime')

# Separate root from replies
actual_replies = thread_tweets[thread_tweets['id'] != root_tweet['id']].copy()

print(f"✓ Total tweets in thread: {len(thread_tweets)}")
print(f"✓ Actual replies: {len(actual_replies)}")

# Extract user IDs
thread_tweets['user_id'] = thread_tweets['user'].apply(extract_user_id)
actual_replies['user_id'] = actual_replies['user'].apply(extract_user_id)
root_user_id = extract_user_id(root_tweet['user'])

# Clean up data
thread_tweets = thread_tweets.dropna(subset=['user_id'])
actual_replies = actual_replies.dropna(subset=['user_id'])
thread_tweets['user_id'] = thread_tweets['user_id'].astype(int)
actual_replies['user_id'] = actual_replies['user_id'].astype(int)

print("\n" + "="*80)
print("SELECTED ROOT TWEET")
print("="*80)
print(f"Tweet ID: {root_tweet['id']}")
print(f"User ID: {root_user_id}")
print(f"Posted at: {root_tweet['datetime']}")
print(f"Expected replies (metadata): {int(root_tweet['replyCount']):,}")
print(f"Actual replies in dataset: {len(actual_replies):,}")
print(f"View count: {root_tweet['views']:,}")
print(f"\nTweet text:\n  \"{root_tweet['text']}\"")

# Timeline stats
print("\n" + "="*80)
print("THREAD TIMELINE")
print("="*80)
print(f"First tweet: {thread_tweets['datetime'].min()}")
print(f"Last tweet: {thread_tweets['datetime'].max()}")
duration = thread_tweets['datetime'].max() - thread_tweets['datetime'].min()
print(f"Duration: {duration}")
print(f"Unique users: {thread_tweets['user_id'].nunique()}")

# ABM initialization
root_views = root_tweet['views']
lurker_count = max(0, root_views - len(actual_replies) - 1)

print("\n" + "="*80)
print("ABM AGENT INITIALIZATION")
print("="*80)
print(f"Root author: 1")
print(f"Active agents (replied): {len(actual_replies):,}")
print(f"Lurker agents (views - replies): {lurker_count:,}")
print(f"Total agents: {len(actual_replies) + lurker_count + 1:,}")

# Coverage
expected = int(root_tweet['replyCount']) if root_tweet['replyCount'] > 0 else len(actual_replies)
coverage = (len(actual_replies) / expected * 100) if expected > 0 else 100
print(f"\nCoverage: {len(actual_replies):,} / {expected:,} = {coverage:.1f}%")

# Prepare output data
thread_data = {
    'root_tweet': {
        'id': int(root_tweet['id']),
        'conversation_id': int(root_tweet['conversationId']),
        'user_id': int(root_user_id) if root_user_id else None,
        'text': str(root_tweet['text']),
        'timestamp': root_tweet['datetime'].isoformat(),
        'epoch': int(root_tweet['epoch']),
        'expected_replies': int(root_tweet['replyCount']),
        'view_count': int(root_tweet['views']),
        'like_count': int(root_tweet['likeCount']) if pd.notna(root_tweet['likeCount']) else 0,
        'retweet_count': int(root_tweet['retweetCount']) if pd.notna(root_tweet['retweetCount']) else 0
    },
    'replies': {
        'actual_count': int(len(actual_replies)),
        'coverage_percent': float(coverage),
        'tweet_ids': [int(x) for x in actual_replies['id'].tolist()],
        'user_ids': [int(x) for x in actual_replies['user_id'].tolist()],
        'timestamps': actual_replies['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'epochs': [int(x) for x in actual_replies['epoch'].tolist()]
    },
    'timeline': {
        'start': thread_tweets['datetime'].min().isoformat(),
        'end': thread_tweets['datetime'].max().isoformat(),
        'duration_seconds': int((thread_tweets['datetime'].max() - thread_tweets['datetime'].min()).total_seconds())
    },
    'abm_config': {
        'active_agents': int(len(actual_replies)),
        'lurker_agents': int(lurker_count),
        'total_agents': int(len(actual_replies) + lurker_count + 1)
    }
}

# Save metadata JSON
output_json = 'output/selected_thread_metadata.json'
with open(output_json, 'w') as f:
    json.dump(thread_data, f, indent=2)
print(f"\n✓ Thread metadata saved to: {output_json}")

# Save full thread tweets CSV
output_csv = 'output/selected_thread_tweets.csv'
thread_tweets.to_csv(output_csv, index=False)
print(f"✓ Thread tweets saved to: {output_csv}")

# Save just the reply user IDs for easy DNA matching
output_users = 'output/selected_thread_user_ids.txt'
with open(output_users, 'w') as f:
    f.write(f"# Thread {TARGET_TWEET_ID} - User IDs\n")
    f.write(f"# Root user: {root_user_id}\n")
    f.write(f"# Reply users ({len(actual_replies)} total):\n\n")
    for user_id in sorted(actual_replies['user_id'].unique()):
        f.write(f"{user_id}\n")
print(f"✓ User IDs saved to: {output_users}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Match user_ids with processed_agents_raw_1_political.csv")
print("2. Check DNA coverage (how many reply users have classifications)")
print("3. Initialize Mesa ABM with Active + Lurker agents")
print("="*80)
