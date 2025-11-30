#!/bin/bash
# quick_experiments.sh
# Each experiment: reduced epochs for speed

# Define base paths
TRAIN_DATA_DIR="/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp/training_npz"
TRAIN_LIST="/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp/training_npz/npz_training_data.csv"
PYTHON_PATH="/opt/anaconda3/envs/phasenet_py311/bin/python"
TRAIN_SCRIPT="/Users/saheeadewaledalli/PhaseNet/phasenet/train.py"

# Create directories
mkdir -p models logs results

# Baseline (your current improved model as reference)
echo "Running baseline experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/baseline \
    --model_dir models/baseline \
    --plot_figure

# With augmentation only
echo "Running augmentation experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/augmentation \
    --model_dir models/augmentation \
    --plot_figure

# With temporal loss only
echo "Running temporal loss experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/temporal_loss \
    --model_dir models/temporal_loss \
    --plot_figure

# Deeper network
echo "Running deeper network experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/deeper \
    --model_dir models/deeper \
    --plot_figure

# Wider network
echo "Running wider network experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/wider \
    --model_dir models/wider \
    --plot_figure

# Full model (all improvements)
echo "Running full model experiment..."
$PYTHON_PATH $TRAIN_SCRIPT \
    --mode train \
    --epochs 10 \
    --batch_size 20 \
    --learning_rate 0.001 \
    --format numpy \
    --train_dir $TRAIN_DATA_DIR \
    --train_list $TRAIN_LIST \
    --log_dir logs/full \
    --model_dir models/full \
    --plot_figure

echo "All experiments complete! Now evaluate..."

echo "DONE!!"