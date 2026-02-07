#!/bin/bash
# =============================================================================
# GCP Batch Simulation - Master Runner Script
# =============================================================================
# This script runs the complete batch simulation pipeline on GCP:
# 1. Prepares simulation directories from august_thread_depths
# 2. Runs all simulations (parallel or sequential)
# 3. Aggregates results for analysis
#
# Usage:
#   ./scripts/run_batch_pipeline.sh [OPTIONS]
#
# Options:
#   --prepare-only      Only prepare simulations, don't run
#   --run-only          Only run simulations (assumes already prepared)
#   --aggregate-only    Only aggregate results
#   --limit N           Limit to N threads (default: all)
#   --workers N         Parallel workers (default: 4)
#   --sequential        Run sequentially instead of parallel
#   --rounds N          Rounds per simulation (default: 10)
#
# =============================================================================

set -e  # Exit on error

# Default values
LIMIT=""
WORKERS=4
MODE="parallel"
ROUNDS=10
PREPARE=true
RUN=true
AGGREGATE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prepare-only)
            RUN=false
            AGGREGATE=false
            shift
            ;;
        --run-only)
            PREPARE=false
            AGGREGATE=false
            shift
            ;;
        --aggregate-only)
            PREPARE=false
            RUN=false
            shift
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --workers)
            WORKERS=$2
            shift 2
            ;;
        --sequential)
            MODE="sequential"
            shift
            ;;
        --rounds)
            ROUNDS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================================="
echo "GCP BATCH SIMULATION PIPELINE"
echo "============================================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo "Workers: $WORKERS"
echo "Rounds: $ROUNDS"
echo ""

# Step 1: Prepare simulations
if [ "$PREPARE" = true ]; then
    echo "============================================================================="
    echo "STEP 1: PREPARING BATCH SIMULATIONS"
    echo "============================================================================="
    python scripts/prepare_batch_simulations.py $LIMIT
    echo ""
fi

# Step 2: Run simulations
if [ "$RUN" = true ]; then
    echo "============================================================================="
    echo "STEP 2: RUNNING SIMULATIONS"
    echo "============================================================================="
    python scripts/gcp_batch_runner.py \
        --mode $MODE \
        --workers $WORKERS \
        --rounds $ROUNDS
    echo ""
fi

# Step 3: Aggregate results
if [ "$AGGREGATE" = true ]; then
    echo "============================================================================="
    echo "STEP 3: AGGREGATING RESULTS"
    echo "============================================================================="
    python scripts/aggregate_results.py
    echo ""
fi

echo "============================================================================="
echo "PIPELINE COMPLETE"
echo "============================================================================="
echo ""
echo "Results are available in:"
echo "  - batch_output/aggregated_results.csv"
echo "  - batch_output/aggregated_statistics.json"
echo ""
