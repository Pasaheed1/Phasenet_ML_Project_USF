#!/bin/bash
# evaluate_all_experiments.sh - Memory Efficient Evaluation Only
# Evaluates already-trained models' predictions

# Define base paths for EVALUATION only
PRED_DATA_DIR="/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp"
MANUAL_PICKS="../test_data/converted_back_to_npz_3comp/npz_data_list_detailed.csv"
PYTHON_PATH="/opt/anaconda3/envs/phasenet_py311/bin/python"
EVAL_SCRIPT="/Users/saheeadewaledalli/PhaseNet/phasenet/phasenet_comp_results.py"

echo "============================================================"
echo "MEMORY-EFFICIENT EVALUATION OF TRAINED MODELS"
echo "============================================================"

# Configuration - ONLY evaluation parameters
CONFIGS=("baseline" "augmentation" "temporal_loss" "deeper" "wider" "full")
SAMPLING_RATE=100
TOLERANCE_MS=500
CHUNK_SIZE=5000  # Process in smaller chunks for M4 memory

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ ERROR: Evaluation script not found: $EVAL_SCRIPT"
    echo "   Please ensure phasenet_comprehensive_results.py is in the current directory"
    exit 1
fi

# Check if manual picks exist
if [ ! -f "$MANUAL_PICKS" ]; then
    echo "❌ ERROR: Manual picks file not found: $MANUAL_PICKS"
    exit 1
fi

echo ""
echo "📊 Evaluation Configuration:"
echo "   Manual picks: $MANUAL_PICKS"
echo "   Sampling rate: $SAMPLING_RATE Hz"
echo "   Tolerance: $TOLERANCE_MS ms"
echo "   Chunk size: $CHUNK_SIZE"
echo "   Models to evaluate: ${CONFIGS[@]}"
echo ""

# Check if results directories exist
echo "🔍 Checking for prediction results..."
for CONFIG in "${CONFIGS[@]}"; do
    PICKS_FILE="results/$CONFIG/picks.csv"
    if [ -f "$PICKS_FILE" ]; then
        PICK_COUNT=$(wc -l < "$PICKS_FILE")
        echo "   ✅ $CONFIG: $PICK_COUNT picks found"
    else
        echo "   ❌ $CONFIG: No picks.csv found"
    fi
done

echo ""
echo "🚀 Starting evaluation process..."
echo ""

# Evaluate each configuration
SUCCESSFUL_EVALS=0
TOTAL_EVALS=0

for CONFIG in "${CONFIGS[@]}"; do
    TOTAL_EVALS=$((TOTAL_EVALS + 1))
    
    PICKS_FILE="results/$CONFIG/picks.csv"
    OUTPUT_FILE="results/$CONFIG/phasenet_comprehensive_results.txt"
    
    echo ""
    echo "🔍 Evaluating: $CONFIG"
    echo "------------------------------------------------------------"
    echo "   Input:  $PICKS_FILE"
    echo "   Output: $OUTPUT_FILE"
    
    # Check if picks file exists
    if [ ! -f "$PICKS_FILE" ]; then
        echo "   ❌ SKIPPING: Picks file not found"
        echo "   💡 Run prediction first: ./run_predictions.sh"
        continue
    fi
    
    # Count lines in picks file
    PICK_COUNT=$(wc -l < "$PICKS_FILE")
    echo "   Picks count: $PICK_COUNT"
    
    # Create output directory
    mkdir -p "results/$CONFIG"
    
    # Run memory-efficient evaluation
    echo "   🚀 Running evaluation (chunk_size: $CHUNK_SIZE)..."
    
    $PYTHON_PATH "$EVAL_SCRIPT" \
        --predicted_picks "$PICKS_FILE" \
        --manual_picks "$MANUAL_PICKS" \
        --output_dir "$OUTPUT_FILE" \
        --sampling_rate "$SAMPLING_RATE" \
        --tolerance_ms "$TOLERANCE_MS" \
    
    # Check if evaluation was successful
    if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
        echo "   ✅ SUCCESS: Evaluation completed"
        SUCCESSFUL_EVALS=$((SUCCESSFUL_EVALS + 1))
        
        # Show summary of results
        echo ""
        echo "   📈 Results summary:"
        if grep -q "P-PHASE RESULTS:" "$OUTPUT_FILE"; then
            echo "   P-phase:"
            grep -E "Matched picks:|MAE:|RMSE:" "$OUTPUT_FILE" | grep -A2 "P-PHASE" | head -3 | sed 's/^/     /'
        fi
        if grep -q "S-PHASE RESULTS:" "$OUTPUT_FILE"; then
            echo "   S-phase:"
            grep -E "Matched picks:|MAE:|RMSE:" "$OUTPUT_FILE" | grep -A2 "S-PHASE" | head -3 | sed 's/^/     /'
        fi
    else
        echo "   ❌ FAILED: Evaluation failed"
        echo "   💡 Tip: Try reducing chunk_size further if memory issues persist"
    fi
    
    echo "------------------------------------------------------------"
done

# Final summary
echo ""
echo "============================================================"
echo "EVALUATION COMPLETE - SUMMARY"
echo "============================================================"
echo ""
echo "📊 Overall Results:"
echo "   Successful evaluations: $SUCCESSFUL_EVALS/$TOTAL_EVALS"
echo ""
echo "📁 Evaluation results saved in:"
for CONFIG in "${CONFIGS[@]}"; do
    RESULT_FILE="results/$CONFIG/phasenet_comprehensive_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        echo "   ✅ $RESULT_FILE"
    else
        echo "   ❌ $RESULT_FILE (missing)"
    fi
done

echo ""
echo "🔧 Evaluation parameters:"
echo "   Chunk size: $CHUNK_SIZE"
echo "   Tolerance: $TOLERANCE_MS ms"
echo "   Sampling rate: $SAMPLING_RATE Hz"

# Performance comparison
if [ $SUCCESSFUL_EVALS -gt 1 ]; then
    echo ""
    echo "🏆 Performance Comparison (MAE - Mean Absolute Error):"
    for CONFIG in "${CONFIGS[@]}"; do
        RESULT_FILE="results/$CONFIG/phasenet_comprehensive_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            P_MAE=$(grep "MAE:" "$RESULT_FILE" | head -1 | awk '{print $3}')
            S_MAE=$(grep "MAE:" "$RESULT_FILE" | tail -1 | awk '{print $3}')
            P_MATCHES=$(grep "Matched picks:" "$RESULT_FILE" | head -1 | awk '{print $4}')
            S_MATCHES=$(grep "Matched picks:" "$RESULT_FILE" | tail -1 | awk '{print $4}')
            echo "   $CONFIG: P-MAE=${P_MAE}ms (n=$P_MATCHES), S-MAE=${S_MAE}ms (n=$S_MATCHES)"
        fi
    done
fi

echo ""
echo "📌 Next steps:"
echo "   1. Review detailed results in results/*/phasenet_comprehensive_results.txt"
echo "   2. Run comparison script: python compare_ablations.py"
echo "   3. Generate visualizations from the results"

echo ""
echo "============================================================"