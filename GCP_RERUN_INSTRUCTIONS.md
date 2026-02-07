# GCP Pipeline Rerun Instructions

## What Was Fixed

1. **Step 3 Skip Logic** - Now validates that the agents file has at least 100 real agents before skipping extraction
2. **Error Handling** - Pipeline now fails fast with clear error messages instead of silently continuing
3. **Data Validation** - Each step validates its outputs before proceeding
4. **Better Logging** - Shows agent counts, political distributions, and thread stats

## Steps to Rerun on GCP

### 1. SSH into your GCP instance

```bash
gcloud compute ssh instance-template-20260203-20260203-155804 --zone=us-west4-b
```

### 2. Navigate to project directory

```bash
cd ~/Dissertation_Final
```

### 3. Pull the latest fixes

```bash
git pull origin main
```

### 4. Clean up bad files

```bash
# Option A: Delete only Step 3-6 outputs (recommended - keeps downloaded data)
rm -rf batch_output/* batch_simulations/*
rm -f processed_agents/processed_agents_august.csv

# Option B: Use the cleanup script
bash scripts/cleanup_pipeline.sh

# Verify cleanup
ls -lh processed_agents/
ls -lh batch_output/
```

### 5. Verify Step 1 & 2 data exists

```bash
# Check data was downloaded (should show ~100-117 files)
ls data/aug_chunk_*.csv | wc -l

# Check threads were found (should show 100 rows + 1 header)
wc -l output/august_thread_depths.csv

# If either is missing, you need to run from Step 1:
python scripts/gcp_full_pipeline.py --step 1 --threads 100
python scripts/gcp_full_pipeline.py --step 2 --threads 100
```

### 6. Run Step 3 (Extract and Classify Agents)

This is the step that was skipped before. It will take **1-2 hours** on GPU.

```bash
# Run in background with logging
nohup python scripts/gcp_full_pipeline.py --step 3 --threads 100 > pipeline_step3.log 2>&1 &

# Get the process ID
echo $! > pipeline.pid

# Monitor progress
tail -f pipeline_step3.log
```

**What to expect:**
- "Found X unique users in target threads" (should be 1000+)
- "Total tweets to process: X" (should be 10,000+)
- Classification progress bar showing tweets/second
- "Saved X agent profiles" (should be 100+)

**If it fails:**
- Check the log: `grep -i error pipeline_step3.log`
- Verify data exists: `ls data/aug_chunk_*.csv | wc -l`

### 7. Run Steps 4-6 (Prepare, Simulate, Aggregate)

Once Step 3 completes successfully:

```bash
# Run Steps 4-6 together (faster)
python scripts/gcp_full_pipeline.py --step 4 --threads 100
python scripts/gcp_full_pipeline.py --step 5 --workers 4
python scripts/gcp_full_pipeline.py --step 6
```

**Or run as one command:**

```bash
# Run Steps 4-6 in background
nohup bash -c "
  python scripts/gcp_full_pipeline.py --step 4 --threads 100 && \
  python scripts/gcp_full_pipeline.py --step 5 --workers 4 && \
  python scripts/gcp_full_pipeline.py --step 6
" > pipeline_step456.log 2>&1 &

# Monitor
tail -f pipeline_step456.log
```

**Step 5 will take 3-5 hours** (100 simulations with 10 rounds each using Dolphin LLM).

### 8. Verify Outputs

```bash
# Check final outputs exist and have content
ls -lh batch_output/aggregated_results.csv      # Should be >100KB
ls -lh batch_output/aggregated_statistics.json  # Should be >1KB
ls -lh processed_agents/processed_agents_august.csv  # Should be >1MB

# Check row counts
wc -l batch_output/aggregated_results.csv  # Should be 101 (100 threads + header)
wc -l processed_agents/processed_agents_august.csv  # Should be >1000

# Check agent data is valid
head -5 processed_agents/processed_agents_august.csv
# Should show real user_ids (numbers), NOT 'nan'
```

### 9. Download Results to Local

From your **local machine**:

```bash
cd ~/Desktop/Dissertation_Final

# Download all results
gcloud compute scp --recurse instance-template-20260203-20260203-155804:~/Dissertation_Final/batch_output/ ./batch_output_gcp/ --zone=us-west4-b

gcloud compute scp instance-template-20260203-20260203-155804:~/Dissertation_Final/processed_agents/processed_agents_august.csv ./processed_agents/ --zone=us-west4-b
```

## Troubleshooting

### "No unique users found"
- Data wasn't downloaded correctly
- Check: `ls data/ | head -20`
- Fix: Run Step 1 again

### "Only extracted X agents (need 100+)"
- Threads don't have enough participants
- Check: `head output/august_thread_depths.csv`
- Fix: Rerun Step 2 with more threads: `--threads 200`

### "CUBLAS_STATUS_NOT_INITIALIZED"
- GPU warnings, but should still work
- Models will fallback to CPU path
- Can ignore unless classification fails completely

### Simulations taking too long
- 100 threads × 10 rounds × ~30s/round = ~8 hours total
- With 4 workers: ~2-3 hours
- Normal for LLM-based simulation

### Out of memory
- Reduce workers: `--workers 2`
- Or run sequential: Use `gcp_batch_runner.py --mode sequential`

## Expected Timeline

| Step | Duration | What it does |
|------|----------|--------------|
| Step 1 | 30-60 min | Download august data (if not cached) |
| Step 2 | 10-20 min | Find top 100 threads |
| **Step 3** | **1-2 hours** | **Classify agents (GPU)** |
| Step 4 | 2-5 min | Prepare simulation configs |
| **Step 5** | **2-4 hours** | **Run 100 simulations (4 workers)** |
| Step 6 | 1-2 min | Aggregate results |
| **Total** | **4-7 hours** | **End-to-end (Steps 3-6)** |

## How to Check If It's Working

### Step 3 (Classification)
```bash
# Should see increasing progress
grep "Classifying:" pipeline_step3.log | tail -5

# Should eventually show completion
grep "Saved.*agent profiles" pipeline_step3.log
```

### Step 5 (Simulations)
```bash
# Check how many simulations completed
ls batch_output/thread_*/summary.json | wc -l

# Watch parallel progress
watch -n 5 'ls batch_output/thread_*/summary.json | wc -l'
```

### Overall Health Check
```bash
# Check for errors
grep -i "error\|failed\|exception" pipeline_*.log | grep -v "CUBLAS_STATUS" | tail -20

# Check success messages
grep "✓" pipeline_*.log | tail -20
```

## Quick Start (TL;DR)

```bash
# On GCP instance
cd ~/Dissertation_Final
git pull
rm -f processed_agents/processed_agents_august.csv
rm -rf batch_output/* batch_simulations/*

# Run Steps 3-6
nohup python scripts/gcp_full_pipeline.py --step 3 --threads 100 > step3.log 2>&1 && \
nohup bash -c "
  python scripts/gcp_full_pipeline.py --step 4 --threads 100 && \
  python scripts/gcp_full_pipeline.py --step 5 --workers 4 && \
  python scripts/gcp_full_pipeline.py --step 6
" > step456.log 2>&1 &

# Monitor
tail -f step3.log
# (wait for Step 3 to finish, then)
tail -f step456.log
```
