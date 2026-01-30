"""
Create agent DNA profiles for the selected thread.
Matches thread user IDs with processed_agents_raw_1_political.csv
and outputs a clean CSV with all DNA features for ABM initialization.
"""
import pandas as pd
import json

print("="*80)
print("CREATING AGENT DNA PROFILES FOR SELECTED THREAD")
print("="*80)

# Load thread metadata
print("\nLoading thread metadata...")
with open('output/selected_thread_metadata.json', 'r') as f:
    thread = json.load(f)

root_user_id = thread['root_tweet']['user_id']
reply_user_ids = thread['replies']['user_ids']
all_user_ids = [root_user_id] + reply_user_ids

print(f"✓ Thread has {len(all_user_ids)} unique users")
print(f"  - Root user: {root_user_id}")
print(f"  - Reply users: {len(reply_user_ids)}")

# Load processed agents DNA
print("\nLoading agent DNA profiles...")
agents_df = pd.read_csv('output/processed_agents_raw_1_political.csv')
print(f"✓ Loaded {len(agents_df):,} agent profiles")

# Filter to only thread participants
thread_agents = agents_df[agents_df['user_id'].isin(all_user_ids)].copy()

# Add role column (root vs active replier)
thread_agents['role'] = thread_agents['user_id'].apply(
    lambda x: 'root' if x == root_user_id else 'active'
)

# Sort: root first, then by user_id
thread_agents = thread_agents.sort_values('role', ascending=False).reset_index(drop=True)

print(f"\n✓ Matched {len(thread_agents)} agents")
print(f"  - Root: {len(thread_agents[thread_agents['role']=='root'])}")
print(f"  - Active: {len(thread_agents[thread_agents['role']=='active'])}")

# Check for missing users
missing = set(all_user_ids) - set(thread_agents['user_id'])
if missing:
    print(f"\n⚠ Warning: {len(missing)} users not found in DNA profiles:")
    for uid in list(missing)[:10]:
        print(f"  - {uid}")
else:
    print(f"\n✓ All thread users have DNA profiles (100% coverage)")

# Summary statistics
print("\n" + "="*80)
print("AGENT DNA SUMMARY")
print("="*80)

print("\nPolitical Distribution:")
for label, count in thread_agents['political_label'].value_counts().items():
    pct = count / len(thread_agents) * 100
    avg_score = thread_agents[thread_agents['political_label']==label]['political_score'].mean()
    print(f"  {label:8s}: {count:3d} ({pct:5.1f}%) | Avg confidence: {avg_score:.3f}")

print("\nEmotion Distribution:")
for label, count in thread_agents['emotion_label'].value_counts().head(5).items():
    pct = count / len(thread_agents) * 100
    print(f"  {label:10s}: {count:3d} ({pct:5.1f}%)")

print("\nSentiment Distribution:")
for label, count in thread_agents['sentiment_label'].value_counts().items():
    pct = count / len(thread_agents) * 100
    print(f"  {label:10s}: {count:3d} ({pct:5.1f}%)")

print("\nAggression Metrics:")
print(f"  Mean hate score: {thread_agents['hate_score'].mean():.3f}")
print(f"  Mean offensive score: {thread_agents['offensive_score'].mean():.3f}")
print(f"  Mean aggression (hate+offensive): {(thread_agents['hate_score'] + thread_agents['offensive_score']).mean():.3f}")
print(f"  Max aggression: {(thread_agents['hate_score'] + thread_agents['offensive_score']).max():.3f}")

print("\nEngagement Metrics:")
print(f"  Total tweets: {thread_agents['tweet_count'].sum():,}")
print(f"  Total views: {thread_agents['view_count'].sum():,}")
print(f"  Avg tweets/user: {thread_agents['tweet_count'].mean():.1f}")
print(f"  Avg views/user: {thread_agents['view_count'].mean():.1f}")

# Save to CSV
output_file = 'output/agents_for_tweet.csv'
thread_agents.to_csv(output_file, index=False)

print("\n" + "="*80)
print(f"✓ Agent DNA profiles saved to: {output_file}")
print("="*80)

# Show sample rows
print("\nSample agent profiles:")
print(thread_agents[['user_id', 'role', 'political_label', 'political_score',
                      'emotion_label', 'hate_score', 'offensive_score']].head(10))

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Use agents_for_tweet.csv to initialize Active agents in Mesa ABM")
print("2. Spawn Lurker agents using Pew 2024 distributions")
print("3. Build Historical Replay simulation with bounded confidence + backfire")
print("="*80)
