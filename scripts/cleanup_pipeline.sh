#!/bin/bash
# Cleanup script to reset pipeline state before rerunning
# Usage: bash scripts/cleanup_pipeline.sh [--all | --from-step N]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================="
echo "PIPELINE CLEANUP SCRIPT"
echo "=================================="

# Parse arguments
CLEANUP_MODE="partial"
if [ "$1" == "--all" ]; then
    CLEANUP_MODE="all"
elif [ "$1" == "--from-step" ] && [ -n "$2" ]; then
    STEP=$2
else
    echo ""
    echo "Usage:"
    echo "  bash scripts/cleanup_pipeline.sh --all              # Delete everything (fresh start)"
    echo "  bash scripts/cleanup_pipeline.sh --from-step 3      # Delete from Step 3 onwards"
    echo ""
    echo "By default, deletes Step 3-6 outputs (keeps downloaded data and threads)"
    echo ""
fi

echo ""
echo "Cleanup mode: $CLEANUP_MODE"
echo ""

# Always safe to delete these (intermediate outputs)
echo "Deleting batch outputs..."
rm -rf batch_output/*
rm -rf batch_simulations/*
echo "✓ Deleted batch_output/ and batch_simulations/"

# Delete agent files (Step 3 output)
echo ""
echo "Deleting agent files..."
rm -f processed_agents/processed_agents_august.csv
rm -f processed_agents/processed_agents_raw_august.csv
echo "✓ Deleted processed_agents/processed_agents_august.csv"

if [ "$CLEANUP_MODE" == "all" ]; then
    echo ""
    echo "FULL CLEANUP - Deleting threads and data..."

    # Delete Step 2 outputs
    rm -f output/august_thread_depths.csv
    rm -f output/top_threads_metadata.json
    echo "✓ Deleted output/august_thread_depths.csv"

    # Delete Step 1 outputs (WARNING: Will need to re-download)
    echo ""
    echo "⚠️  WARNING: This will delete all downloaded data (~5GB)!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        rm -f data/aug_chunk_*.csv
        echo "✓ Deleted data/aug_chunk_*.csv"
    else
        echo "Skipped data deletion"
    fi
fi

echo ""
echo "=================================="
echo "CLEANUP COMPLETE"
echo "=================================="
echo ""
echo "You can now run the pipeline from the desired step:"
echo "  python scripts/gcp_full_pipeline.py --step 3 --threads 100  # Start from Step 3"
echo "  python scripts/gcp_full_pipeline.py --step all              # Full pipeline"
echo ""
