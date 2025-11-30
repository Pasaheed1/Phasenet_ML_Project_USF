# Phasenet-SAA
The purpose of this project is to be able to upgrade the original PhaseNet code.


# PhaseNet-SAA Improvements: Implementation & Results
## 1. Novel Contributions Implemented

### 1.1 Data Augmentation
**Location**: `data_reader.py`, lines XX-YY

**Implementation**:
- Gaussian noise injection (σ = 0.01-0.1 × signal std)
- Random amplitude scaling (0.7-1.3×)
- Random time shifts (±50 samples)
- Random polarity flips (20% probability)

**Motivation**: Seismic data varies in SNR, amplitude, and timing. Augmentation 
improves model robustness to these variations without collecting more data.

**Code snippet**:
```python
def augment_waveform(data, config):
    # Add noise
    noise = np.random.normal(0, 0.05*np.std(data), data.shape)
    data = data + noise
    # Scale amplitude
    data = data * random.uniform(0.7, 1.3)
    # ... (see full implementation)
```

### 1.2 Temporal Consistency Loss
**Location**: `model.py`, lines XX-YY

**Implementation**:
- Penalizes S-wave picks before P-wave picks
- Enforces minimum P-S separation (>50 samples = 0.5s @ 100Hz)
- Uses soft constraints via ReLU penalty

**Motivation**: Physical constraint - S-waves always arrive after P-waves. 
Enforcing this improves pick reliability.

**Mathematical formulation**:
```
L_temporal = α × ReLU(t_P - t_S + δ_min) + β × ReLU(δ_min - (t_S - t_P))
where t_P, t_S are expected pick times, δ_min = 50 samples minimum separation
```

### 1.3 Network Architecture Scaling
**Location**: `model.py`, `ModelConfig` class

**Implementation**:
- Deeper: 6 encoder/decoder levels (was 5)
- Wider: 16 root filters (was 8)
- Combined: 6 levels × 16 filters

**Motivation**: Increased capacity to learn complex seismic patterns without 
overfitting (prevented by augmentation and regularization).

## 2. Experimental Design

### 2.1 Ablation Study Design

| Experiment | Augmentation | Temporal Loss | Depth | Width | Purpose |
|------------|--------------|---------------|-------|-------|---------|
| Baseline   | ❌ | ❌ | 5 | 8 | Reference (your improved model) |
| +Augmentation | ✅ | ❌ | 5 | 8 | Isolate augmentation effect |
| +Temporal Loss | ❌ | ✅ | 5 | 8 | Isolate temporal loss effect |
| +Deeper | ❌ | ❌ | 6 | 8 | Isolate depth effect |
| +Wider | ❌ | ❌ | 5 | 16 | Isolate width effect |
| Full Model | ✅ | ✅ | 6 | 16 | Combined effect |

### 2.2 Training Details
- Dataset: [YOUR DATASET SIZE] seismic waveforms
- Epochs: 10 (quick experiments, normally 100+)
- Batch size: 20
- Optimizer: Adam, lr=0.01
- Metrics: MAE (ms), RMSE (ms), Precision, Recall, F1

### 2.3 Evaluation Protocol
- Test set: [N] unseen seismic events
- Manual picks as ground truth
- Evaluation tolerance: ±50ms (5 samples @ 100Hz)

## 3. Results

### 3.1 Ablation Study Results
```
=== ABLATION RESULTS (10 epochs, preliminary) ===
| Configuration    | P-MAE (ms) | S-MAE (ms) | P-F1  | S-F1  | Notes |
|------------------|------------|------------|-------|-------|-------|
| Baseline         | XX.X       | YY.Y       | 0.XXX | 0.YYY | Your improved model |
| +Augmentation    | XX.X       | YY.Y       | 0.XXX | 0.YYY | Change: ±Z.Z ms |
| +Temporal Loss   | XX.X       | YY.Y       | 0.XXX | 0.YYY | Change: ±Z.Z ms |
| +Deeper (d=6)    | XX.X       | YY.Y       | 0.XXX | 0.YYY | Change: ±Z.Z ms |
| +Wider (f=16)    | XX.X       | YY.Y       | 0.XXX | 0.YYY | Change: ±Z.Z ms |
| Full Model       | XX.X       | YY.Y       | 0.XXX | 0.YYY | Total change: ±Z.Z ms |

[AFTER EXPERIMENTS COMPLETE, FILL IN NUMBERS FROM results/*/picks.csv 
 USING YOUR EXISTING evaluation scripts]
```

### 3.2 Key Findings

**Data Augmentation**:
- Impact: [±X%] improvement in MAE
- Observation: [Improved/Degraded] robustness to noise
- Analysis: [Your interpretation]

**Temporal Consistency Loss**:
- Impact: [±X%] improvement in F1 score
- Observation: [Reduced/Increased] unrealistic picks
- Analysis: [Your interpretation]

**Network Scaling**:
- Deeper (6 levels): [±X%] change
- Wider (16 filters): [±X%] change
- Analysis: [Diminishing returns? Overfitting? Improvement?]

### 3.3 Comparison to Literature

| Metric | Baseline | Your Full Model | Paper (Zhu 2019) | Notes |
|--------|----------|-----------------|------------------|-------|
| P-phase MAE | XX.X ms | XX.X ms | ~2.1 ms* | *Travel-time residual |
| S-phase MAE | YY.Y ms | YY.Y ms | ~3.3 ms* | *Different metric |
| P-phase F1 | 0.XXX | 0.XXX | 0.939 | Detection performance |
| S-phase F1 | 0.YYY | 0.YYY | 0.853 | Detection performance |

**Note**: Direct comparison to paper is difficult because:
1. Different datasets (regional vs. their dataset)
2. Different metrics (picking error vs. travel-time residual)
3. Different evaluation protocols

## 4. Discussion

### 4.1 What Worked
[Based on your results, identify the most effective improvements]

### 4.2 What Didn't Work
[Be honest about what didn't help or hurt performance]

### 4.3 Why Results Differ from Paper
1. **Dataset differences**: Your regional data vs. their broader dataset
2. **Metric differences**: Direct picking accuracy vs. earthquake location residuals
3. **Training time**: 10 epochs (quick) vs. their likely 100+ epochs

### 4.4 Future Work
- Train full models (100+ epochs) for fair comparison
- Test on multiple regions for generalization
- Implement ensemble methods
- Add more sophisticated augmentation (phase-aware)

## 5. Conclusions

This work implemented and evaluated three novel improvements to PhaseNet:
1. **Data augmentation** - [Summary of impact]
2. **Temporal consistency loss** - [Summary of impact]
3. **Network scaling** - [Summary of impact]

Combined, these modifications achieved [X%] improvement over baseline 
on [metric], demonstrating that [your conclusion].

**Code availability**: All modifications are clearly documented in the codebase 
with line-by-line comments explaining implementation details.

## 6. Code References

### Key Files Modified:
1. `data_reader.py`: Lines XX-YY (augmentation)
2. `model.py`: Lines XX-YY (temporal loss), Lines XX-YY (config)
3. `experiment_configs.py`: NEW FILE (experiment configurations)
4. `quick_experiments.sh`: NEW FILE (reproducible experiment script)

### How to Reproduce:
```bash
# Run all ablation experiments
./quick_experiments.sh

# Evaluate specific configuration
python predict.py --model_dir models/full --result_dir results/full
python evaluate.py --results results/full

# Generate comparison plots
python plot_ablation_results.py
```
