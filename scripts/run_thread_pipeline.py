"""
Master pipeline script for thread extraction and agent DNA matching.

Reads config/thread_config.yaml and runs the complete pipeline:
1. Find/validate target thread
2. Extract thread tweets and timeline
3. Match users to DNA profiles
4. Generate agents_for_tweet.csv

Usage:
    python scripts/run_thread_pipeline.py

Edit config/thread_config.yaml to change thread selection or parameters.
"""
import pandas as pd
import json
import yaml
import re
from datetime import datetime
from pathlib import Path

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_view_count(viewCount_str):
    """Extract 'count' from viewCount dict string."""
    try:
        match = re.search(r"'count':\s*'(\d+)'", str(viewCount_str))
        return int(match.group(1)) if match else 0
    except:
        return 0

def extract_user_id(user_str):
    """Extract user_id from user dict string."""
    try:
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        return int(match.group(1)) if match else None
    except:
        return None

def log_message(msg, config):
    """Print and log message."""
    print(msg)
    log_path = config['paths']['pipeline_log']
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

# =============================================================================
# STEP 1: LOAD CONFIG
# =============================================================================

print("="*80)
print("THREAD EXTRACTION PIPELINE")
print("="*80)

config_path = Path('config/thread_config.yaml')
if not config_path.exists():
    print(f"ERROR: Config file not found: {config_path}")
    exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n✓ Loaded config from {config_path}")

# Clear log file
log_path = config['paths']['pipeline_log']
Path(log_path).parent.mkdir(exist_ok=True)
with open(log_path, 'w') as f:
    f.write(f"Pipeline started at {datetime.now().isoformat()}\n")
    f.write("="*80 + "\n")

# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================

log_message("\n[STEP 1] Loading raw tweets...", config)
df = pd.read_csv(config['paths']['raw_tweets'])
df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
df['views'] = df['viewCount'].apply(extract_view_count)
log_message(f"✓ Loaded {len(df):,} tweets", config)

log_message("\n[STEP 2] Loading agent DNA profiles...", config)
agents_df = pd.read_csv(config['paths']['agent_dna'])
log_message(f"✓ Loaded {len(agents_df):,} agent profiles", config)

# =============================================================================
# STEP 3: SELECT TARGET THREAD
# =============================================================================

target_tweet_id = config['target_tweet_id']

if target_tweet_id is None:
    log_message("\n[STEP 3] Auto-selecting best thread...", config)

    method = config['auto_select']['method']
    min_replies = config['auto_select']['min_replies']
    require_root = config['auto_select']['require_root']

    # Filter candidates
    if require_root:
        candidates = df[df['in_reply_to_status_id_str'].isna()].copy()
        log_message(f"  Filtering to root tweets only: {len(candidates):,} candidates", config)
    else:
        candidates = df.copy()

    candidates = candidates[candidates['replyCount'] > 0]

    # Count actual replies for each candidate
    results = []
    for idx, root in candidates.iterrows():
        conv_id = root['conversationId']
        conv_tweets = df[df['conversationId'] == conv_id]
        actual_count = len(conv_tweets) - 1  # Exclude root

        if actual_count >= min_replies:
            results.append({
                'tweet_id': root['id'],
                'actual_replies': actual_count,
                'expected_replies': int(root['replyCount']),
                'views': root['views'],
                'coverage': actual_count / root['replyCount'] if root['replyCount'] > 0 else 0
            })

    if not results:
        log_message(f"ERROR: No threads found with >={min_replies} replies", config)
        exit(1)

    results_df = pd.DataFrame(results)

    # Select based on method
    if method == "most_actual_replies":
        target_tweet_id = results_df.sort_values('actual_replies', ascending=False).iloc[0]['tweet_id']
        log_message(f"  Selected tweet with most actual replies: {target_tweet_id}", config)
    elif method == "highest_views":
        target_tweet_id = results_df.sort_values('views', ascending=False).iloc[0]['tweet_id']
        log_message(f"  Selected tweet with highest views: {target_tweet_id}", config)
    elif method == "best_coverage":
        target_tweet_id = results_df.sort_values('coverage', ascending=False).iloc[0]['tweet_id']
        log_message(f"  Selected tweet with best coverage: {target_tweet_id}", config)

    # Update config with selected tweet
    config['target_tweet_id'] = int(target_tweet_id)
else:
    log_message(f"\n[STEP 3] Using configured tweet ID: {target_tweet_id}", config)

# =============================================================================
# STEP 4: EXTRACT THREAD
# =============================================================================

log_message("\n[STEP 4] Extracting thread...", config)

# Get root tweet
root_tweet = df[df['id'] == target_tweet_id]
if len(root_tweet) == 0:
    log_message(f"ERROR: Tweet {target_tweet_id} not found in dataset", config)
    exit(1)

root_tweet = root_tweet.iloc[0]
conversation_id = root_tweet['conversationId']

# Get all tweets in conversation
thread_tweets = df[df['conversationId'] == conversation_id].copy()
thread_tweets = thread_tweets.sort_values('datetime')

# Separate root from replies
actual_replies = thread_tweets[thread_tweets['id'] != root_tweet['id']].copy()

log_message(f"✓ Root tweet: {target_tweet_id}", config)
log_message(f"✓ Conversation ID: {conversation_id}", config)
log_message(f"✓ Total tweets: {len(thread_tweets)}", config)
log_message(f"✓ Actual replies: {len(actual_replies)}", config)

# Extract user IDs
thread_tweets['user_id'] = thread_tweets['user'].apply(extract_user_id)
actual_replies['user_id'] = actual_replies['user'].apply(extract_user_id)
root_user_id = extract_user_id(root_tweet['user'])

thread_tweets = thread_tweets.dropna(subset=['user_id'])
actual_replies = actual_replies.dropna(subset=['user_id'])
thread_tweets['user_id'] = thread_tweets['user_id'].astype(int)
actual_replies['user_id'] = actual_replies['user_id'].astype(int)

log_message(f"✓ Unique users: {thread_tweets['user_id'].nunique()}", config)
log_message(f"✓ Root user ID: {root_user_id}", config)

# Timeline
duration = thread_tweets['datetime'].max() - thread_tweets['datetime'].min()
log_message(f"✓ Timeline: {thread_tweets['datetime'].min()} to {thread_tweets['datetime'].max()}", config)
log_message(f"✓ Duration: {duration}", config)

# =============================================================================
# STEP 5: MATCH TO DNA PROFILES
# =============================================================================

log_message("\n[STEP 5] Matching users to DNA profiles...", config)

reply_user_ids = actual_replies['user_id'].tolist()
all_user_ids = [root_user_id] + reply_user_ids

# Filter agents
thread_agents = agents_df[agents_df['user_id'].isin(all_user_ids)].copy()
thread_agents['role'] = thread_agents['user_id'].apply(
    lambda x: 'root' if x == root_user_id else 'active'
)
thread_agents = thread_agents.sort_values('role', ascending=False).reset_index(drop=True)

dna_coverage = len(thread_agents) / len(all_user_ids) * 100
log_message(f"✓ Matched {len(thread_agents)} agents", config)
log_message(f"✓ DNA coverage: {dna_coverage:.1f}%", config)

# Check for missing
missing = set(all_user_ids) - set(thread_agents['user_id'])
if missing:
    log_message(f"⚠ Warning: {len(missing)} users without DNA profiles", config)

# Political distribution
political_dist = thread_agents['political_label'].value_counts()
log_message("\n  Political distribution:", config)
for label, count in political_dist.items():
    pct = count / len(thread_agents) * 100
    log_message(f"    {label}: {count} ({pct:.1f}%)", config)

# Calculate lurker distribution based on strategy
lurker_strategy = config['abm']['lurker_dist_strategy']
if lurker_strategy == "match_active_agents":
    lurker_dist = {label: count / len(thread_agents) for label, count in political_dist.items()}
    log_message(f"\n  Lurker distribution: Matching active agents", config)
elif lurker_strategy == "pew_2024":
    lurker_dist = config['abm']['pew_2024_dist']
    log_message(f"\n  Lurker distribution: Using Pew 2024 national", config)
else:
    log_message(f"ERROR: Unknown lurker_dist_strategy: {lurker_strategy}", config)
    exit(1)

for label, prob in lurker_dist.items():
    log_message(f"    {label}: {prob:.1%}", config)

# =============================================================================
# STEP 6: SAVE OUTPUTS
# =============================================================================

log_message("\n[STEP 6] Saving outputs...", config)

# Create temporal event sequence for simulation replay
thread_tweets_sorted = thread_tweets.sort_values('epoch')
temporal_events = []

for idx, tweet in thread_tweets_sorted.iterrows():
    event = {
        'tweet_id': int(tweet['id']),
        'user_id': int(tweet['user_id']),
        'timestamp': tweet['datetime'].isoformat(),
        'epoch': int(tweet['epoch']),
        'seconds_since_start': int(tweet['epoch'] - thread_tweets_sorted['epoch'].min()),
        'is_root': bool(tweet['id'] == root_tweet['id']),
        'text': str(tweet['text'])[:200]  # Truncate for metadata
    }
    temporal_events.append(event)

# 1. Thread metadata JSON
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
        'coverage_percent': float((len(actual_replies) / root_tweet['replyCount'] * 100) if root_tweet['replyCount'] > 0 else 100),
        'tweet_ids': [int(x) for x in actual_replies['id'].tolist()],
        'user_ids': [int(x) for x in actual_replies['user_id'].tolist()],
        'timestamps': actual_replies['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'epochs': [int(x) for x in actual_replies['epoch'].tolist()]
    },
    'timeline': {
        'start': thread_tweets['datetime'].min().isoformat(),
        'end': thread_tweets['datetime'].max().isoformat(),
        'start_epoch': int(thread_tweets['epoch'].min()),
        'end_epoch': int(thread_tweets['epoch'].max()),
        'duration_seconds': int((thread_tweets['datetime'].max() - thread_tweets['datetime'].min()).total_seconds()),
        'total_events': len(temporal_events)
    },
    'temporal_events': temporal_events,
    'abm_config': {
        'active_agents': int(len(actual_replies)),
        'lurker_agents': int(len(actual_replies) * config['abm']['lurker_ratio']),
        'total_agents': int(len(actual_replies) * (1 + config['abm']['lurker_ratio']) + 1),
        'lurker_ratio': config['abm']['lurker_ratio'],
        'lurker_dist_strategy': lurker_strategy,
        'lurker_political_dist': {k: float(v) for k, v in lurker_dist.items()}
    }
}

output_json = config['paths']['thread_metadata']
with open(output_json, 'w') as f:
    json.dump(thread_data, f, indent=2)
log_message(f"✓ Saved: {output_json}", config)

# 2. Thread tweets CSV
output_csv = config['paths']['thread_tweets']
thread_tweets.to_csv(output_csv, index=False)
log_message(f"✓ Saved: {output_csv}", config)

# 3. User IDs text file
output_users = config['paths']['thread_user_ids']
with open(output_users, 'w') as f:
    f.write(f"# Thread {target_tweet_id} - User IDs\n")
    f.write(f"# Root user: {root_user_id}\n")
    f.write(f"# Reply users ({len(actual_replies)} total):\n\n")
    for user_id in sorted(actual_replies['user_id'].unique()):
        f.write(f"{user_id}\n")
log_message(f"✓ Saved: {output_users}", config)

# 4. Agents for thread CSV
output_agents = config['paths']['agents_for_thread']
thread_agents.to_csv(output_agents, index=False)
log_message(f"✓ Saved: {output_agents}", config)

# =============================================================================
# STEP 7: UPDATE CONFIG WITH METADATA
# =============================================================================

log_message("\n[STEP 7] Updating config with metadata...", config)

config['thread_info'] = {
    'root_tweet_id': int(root_tweet['id']),
    'root_user_id': int(root_user_id) if root_user_id else None,
    'conversation_id': int(conversation_id),
    'timestamp': root_tweet['datetime'].isoformat(),
    'actual_replies': int(len(actual_replies)),
    'unique_users': int(thread_tweets['user_id'].nunique()),
    'duration_seconds': int((thread_tweets['datetime'].max() - thread_tweets['datetime'].min()).total_seconds()),
    'dna_coverage_percent': float(dna_coverage)
}

config['active_agents_dist'] = {
    label: int(count) for label, count in political_dist.items()
}

config['lurker_agents_dist'] = {
    label: float(prob) for label, prob in lurker_dist.items()
}

config['last_run'] = datetime.now().isoformat()

# Save updated config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
log_message(f"✓ Updated: {config_path}", config)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nSelected Thread:")
print(f"  Tweet ID: {root_tweet['id']}")
print(f"  Posted: {root_tweet['datetime']}")
print(f"  Text: \"{root_tweet['text'][:100]}...\"")
print(f"\nThread Statistics:")
print(f"  Actual replies: {len(actual_replies)}")
print(f"  Unique users: {thread_tweets['user_id'].nunique()}")
print(f"  Duration: {duration} ({int(duration.total_seconds())} seconds)")
print(f"  DNA coverage: {dna_coverage:.1f}%")
print(f"\nTemporal Timeline:")
print(f"  Start: {thread_tweets['datetime'].min()}")
print(f"  End: {thread_tweets['datetime'].max()}")
print(f"  Total events: {len(temporal_events)}")
print(f"  Events per minute: {len(temporal_events) / (duration.total_seconds() / 60):.2f}")
print(f"\nPolitical Distribution:")
for label, count in political_dist.items():
    pct = count / len(thread_agents) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")
print(f"\nABM Configuration:")
print(f"  Active agents: {len(actual_replies)}")
print(f"  Lurker agents (ratio {config['abm']['lurker_ratio']}:1): {len(actual_replies) * config['abm']['lurker_ratio']:,}")
print(f"  Total agents: {len(actual_replies) * (1 + config['abm']['lurker_ratio']) + 1:,}")
print(f"\nOutput Files:")
print(f"  ✓ {config['paths']['thread_metadata']}")
print(f"  ✓ {config['paths']['thread_tweets']}")
print(f"  ✓ {config['paths']['thread_user_ids']}")
print(f"  ✓ {config['paths']['agents_for_thread']}")
print(f"  ✓ {config['paths']['pipeline_log']}")
print("="*80)
