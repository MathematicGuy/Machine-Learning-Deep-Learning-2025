# Exercise: Card Fraud Detection with SMOTE

## Overview
This notebook demonstrates proper implementation of a training loop for binary classification (fraud detection) using logistic regression with SMOTE for handling imbalanced data.

## File
- `Exercise_Card_Fraud_Detection_with_SMOTE.ipynb` - Main notebook with corrected training loop

## Key Learning Points

### 1. The Problem: Inefficient Training Loop
The original code had a critical performance issue:
```python
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        # ... batch training ...
        
        # ❌ PROBLEM: Computing accuracy on ENTIRE dataset for EVERY batch
        train_acc = compute_accuracy(X_train, y_train, theta)
        val_acc = compute_accuracy(X_val, y_val, theta)
```

**Why this is bad:**
- With 100 batches/epoch and 50 epochs = 5,000 accuracy computations on full dataset
- Extremely wasteful and slow
- No benefit since accuracy changes minimally per batch

### 2. The Solution: Epoch-Level Evaluation
```python
for epoch in range(epochs):
    train_batch_losses = []
    
    for i in range(0, X_train.shape[0], batch_size):
        # ... batch training ...
        # ✓ Only track batch loss
        train_batch_losses.append(train_loss)
    
    # ✓ Compute accuracy ONCE per epoch
    train_acc = compute_accuracy(X_train, y_train, theta)
    val_acc = compute_accuracy(X_val, y_val, theta)
```

**Benefits:**
- 100x fewer accuracy computations
- Much faster training
- Still provides epoch-level metrics for monitoring
- Industry standard practice

### 3. Performance Impact
- **Original**: 5,000 full-dataset evaluations (50 epochs × 100 batches)
- **Corrected**: 50 full-dataset evaluations (50 epochs × 1)
- **Speedup**: ~100x for accuracy computation

## Requirements
```
numpy
matplotlib
scikit-learn
imbalanced-learn
```

Install with:
```bash
pip install numpy matplotlib scikit-learn imbalanced-learn
```

## Running the Notebook
1. Install dependencies
2. Open `Exercise_Card_Fraud_Detection_with_SMOTE.ipynb` in Jupyter
3. Run all cells sequentially
4. Observe the efficient training process and visualizations

## What You'll Learn
- ✅ Proper training loop implementation
- ✅ Batch vs epoch-level metrics
- ✅ Using SMOTE for imbalanced datasets
- ✅ Logistic regression from scratch
- ✅ Performance optimization in ML training

## Notes
- The notebook uses synthetic data for demonstration
- For real fraud detection, use actual transaction data
- Training time should be significantly reduced compared to the buggy version
- Metrics are tracked and visualized for analysis
