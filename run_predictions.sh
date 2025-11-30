#!/bin/bash
# run_predictions.sh - Just run predictions with existing models

# Define base paths
PRED_DATA_DIR="/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp"
PRED_DATA_LIST="/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp/npz_data_list.csv"
PYTHON_PATH="/opt/anaconda3/envs/phasenet_py311/bin/python"
PREDICT_SCRIPT="/Users/saheeadewaledalli/PhaseNet/phasenet/predict.py"

echo "Running predictions with existing trained models..."

# Evaluate all models using the ACTUAL model locations in log directories
for config in baseline augmentation temporal_loss deeper wider full; do
    echo "Evaluating $config model..."
    mkdir -p results/$config
    
    # Find the most recent model directory in the logs
    LATEST_LOG=$(ls -td logs/$config/*/ | head -1)
    ACTUAL_MODEL_DIR="$LATEST_LOG/models"
    
    echo "Using model from: $ACTUAL_MODEL_DIR"
    
    $PYTHON_PATH $PREDICT_SCRIPT \
        --model_dir "$ACTUAL_MODEL_DIR" \
        --data_dir "$PRED_DATA_DIR" \
        --data_list "$PRED_DATA_LIST" \
        --result_dir "results/$config" \
        --result_fname picks \
        --batch_size 1
        
    echo "✓ $config evaluation completed"
    echo ""
done

echo "All predictions complete!"