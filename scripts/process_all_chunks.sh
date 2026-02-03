#!/bin/bash
# Batch process all 20 chunks of USC X-24 election dataset
# Optimized for GCP GPU instance

echo "=========================================="
echo "USC X-24 Dataset Batch Processor"
echo "=========================================="
echo "This will process chunks 2-20 (chunk 1 already complete)"
echo "Estimated time: 6-10 hours on GPU, 20-30 hours on CPU"
echo ""

# Check if we're resuming or starting fresh
START_CHUNK=2
if [ "$1" != "" ]; then
    START_CHUNK=$1
    echo "Starting from chunk $START_CHUNK"
fi

# Process chunks
for i in $(seq $START_CHUNK 20); do
    echo ""
    echo "=========================================="
    echo "Processing chunk $i/20"
    echo "=========================================="
    
    # Check if output already exists
    if [ -f "processed_agents/processed_agents_chunk_$i.csv" ]; then
        echo "⚠ Output file already exists: processed_agents/processed_agents_chunk_$i.csv"
        read -p "Skip this chunk? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping chunk $i"
            continue
        fi
    fi
    
    # Run processing
    python scripts/step1_classify_chunked.py $i
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ ERROR: Processing failed for chunk $i"
        echo "You can resume from this chunk by running:"
        echo "  bash scripts/process_all_chunks.sh $i"
        exit 1
    fi
    
    echo ""
    echo "✅ Chunk $i complete"
    
    # Show progress
    COMPLETED=$((i - 1))
    REMAINING=$((20 - i))
    echo "Progress: $COMPLETED/20 chunks complete, $REMAINING remaining"
done

echo ""
echo "=========================================="
echo "✅ ALL CHUNKS PROCESSED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Output location: processed_agents/"
echo ""
ls -lh processed_agents/processed_agents_chunk_*.csv | wc -l | xargs echo "Total files:"
du -sh processed_agents/ | awk '{print "Total size:", $1}'
echo ""
echo "Next steps:"
echo "1. Download processed_agents/ folder from GCP to local machine"
echo "2. Run thread pipeline: python scripts/run_thread_pipeline.py"
echo "3. Run simulation: python sim/thread_simulation.py"
