"""
Find the tweet with the most replies in chunk 1 for Historical Replay ABM.

This script identifies the best conversation thread to use for simulation by:
1. Finding the tweet with the highest reply count
2. Extracting all replies that exist in our dataset
3. Comparing expected vs actual reply coverage
4. Outputting thread metadata for ABM initialization

Output: JSON file with root tweet info, reply IDs, user IDs, and coverage stats.
"""
import pandas as pd
import json
from datetime import datetime

print("="*80)
print("FINDING BEST CONVERSATION THREAD FOR HISTORICAL REPLAY")
print("="*80)

# Load chunk 1 data
print("\nLoading chunk 1 data...")
df = pd.read_csv('data/may_july_chunk_1.csv')
df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
print(f"✓ Loaded {len(df):,} tweets")

# Extract viewCount from dict string
def extract_view_count(viewCount_str):
    """Extract 'count' from viewCount dict string."""
    try:
        import re
        match = re.search(r"'count':\s*'(\d+)'", str(viewCount_str))
        return int(match.group(1)) if match else 0
    except:
        return 0

df['views'] = df['viewCount'].apply(extract_view_count)

# Find tweets with replies (potential root tweets)
print("\nAnalyzing conversation threads...")
df_with_replies = df[df['replyCount'] > 0].copy()
print(f"  Found {len(df_with_replies):,} tweets with at least 1 reply")

# Sort by reply count to find most-replied tweet
df_sorted = df_with_replies.sort_values('replyCount', ascending=False)

print("\nTop 10 most-replied tweets:")
print("-"*80)
for idx, row in df_sorted.head(10).iterrows():
    print(f"  Tweet {row['id']}: {int(row['replyCount']):,} replies | "
          f"{row['views']:,} views | "
          f"{row['datetime']}")

# Get the most-replied tweet (root tweet for our simulation)
root_tweet = df_sorted.iloc[0]

print("\n" + "="*80)
print("SELECTED ROOT TWEET FOR SIMULATION")
print("="*80)
print(f"Tweet ID: {root_tweet['id']}")
print(f"Conversation ID: {root_tweet['conversationId']}")
print(f"Posted at: {root_tweet['datetime']}")
print(f"Expected replies (from metadata): {int(root_tweet['replyCount']):,}")
print(f"View count: {root_tweet['views']:,}")
print(f"\nTweet text:\n  \"{root_tweet['text'][:200]}...\"")

# Find all replies to this tweet that exist in our dataset
conversation_id = root_tweet['conversationId']
thread_tweets = df[df['conversationId'] == conversation_id].copy()
thread_tweets = thread_tweets.sort_values('datetime')

# Separate root from replies
actual_replies = thread_tweets[thread_tweets['id'] != root_tweet['id']]

print("\n" + "="*80)
print("THREAD COVERAGE ANALYSIS")
print("="*80)
print(f"Expected total replies: {int(root_tweet['replyCount']):,}")
print(f"Actual replies in dataset: {len(actual_replies):,}")
coverage = (len(actual_replies) / root_tweet['replyCount'] * 100) if root_tweet['replyCount'] > 0 else 0
print(f"Coverage: {coverage:.1f}%")

if len(actual_replies) > 0:
    print(f"\nReply timeline:")
    print(f"  First reply: {actual_replies['datetime'].min()}")
    print(f"  Last reply: {actual_replies['datetime'].max()}")
    print(f"  Duration: {actual_replies['datetime'].max() - actual_replies['datetime'].min()}")

    print(f"\nUnique users who replied: {actual_replies['user'].nunique()}")

# Extract user IDs from replies
print("\nExtracting user IDs from replies...")
import re

def extract_user_id(user_str):
    """Extract user_id from user dict string."""
    try:
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        return int(match.group(1)) if match else None
    except:
        return None

actual_replies = actual_replies.copy()
actual_replies['user_id'] = actual_replies['user'].apply(extract_user_id)
actual_replies = actual_replies.dropna(subset=['user_id'])
actual_replies['user_id'] = actual_replies['user_id'].astype(int)

root_user_id = extract_user_id(root_tweet['user'])

print(f"✓ Extracted {len(actual_replies)} reply user IDs")
print(f"✓ Root tweet user ID: {root_user_id}")

# Calculate lurker spawn count (90-9-1 rule)
root_views = root_tweet['views']
lurker_count = max(0, root_views - len(actual_replies) - 1)  # -1 for root tweet author

print("\n" + "="*80)
print("ABM AGENT INITIALIZATION")
print("="*80)
print(f"Active agents (users who replied): {len(actual_replies):,}")
print(f"Lurker agents to spawn (views - replies): {lurker_count:,}")
print(f"Total agents: {len(actual_replies) + lurker_count + 1:,}")  # +1 for root author

# Prepare output data (convert all numpy/pandas types to native Python)
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

# Save to JSON
output_file = 'output/best_thread_metadata.json'
with open(output_file, 'w') as f:
    json.dump(thread_data, f, indent=2)

print(f"\n✓ Thread metadata saved to: {output_file}")

# Also save the full thread tweets to CSV for ABM
thread_csv = 'output/best_thread_tweets.csv'
thread_tweets.to_csv(thread_csv, index=False)
print(f"✓ Thread tweets saved to: {thread_csv}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review the selected thread in best_thread_tweets.csv")
print("2. Match reply user_ids with processed_agents_chunk_1.csv for DNA")
print("3. Initialize ABM with Active agents (have DNA) + Lurkers (spawn from Pew)")
print("4. Run Historical Replay simulation using the timeline")
print("="*80)
