# ICD-10 Classification Pipeline - Technical Documentation

**Task**: Single-label ICD-10 diagnosis code classification from clinical notes  

## Executive Summary

This project implements a complete machine learning pipeline for automated ICD-10 diagnosis code classification from clinical notes using transformer-based models. The solution leverages the SynthEHR ICD-10-CM Clinical Notes dataset and employs Bio_ClinicalBERT, a domain-specific BERT model fine-tuned for clinical text understanding. Through careful data analysis, we discovered that despite initial appearances, this dataset presents a single-label classification problem where each clinical note maps to exactly one ICD-10 code.

## Critical Discovery: Single-Label Dataset

### Initial Assumption vs Reality
- **Initial Assumption**: Multi-label classification (multiple ICD codes per clinical note)
- **Actual Dataset Structure**: Single-label classification (exactly one ICD code per note)
- **Discovery Method**: Comprehensive dataset analysis revealed all samples contain single-element lists
- **Impact**: Complete refactoring of training and evaluation pipelines for optimal performance

### Verification Code
```python
# Analysis revealing single-label nature
for i in range(10000):
    codes = dataset['train'][i]['codes']
    if isinstance(codes, list) and len(codes) > 1:
        print(f"Multi-label found at {i}")  # Never executes
```

## Architecture Overview

### Core Components

1. **Data Pipeline** (`train.py:25-87`)
   - Automated dataset loading from HuggingFace Hub
   - Label space reduction to top 100 most frequent ICD codes
   - Single-label encoding with integer class indices
   - Filtering ensures both train and test sets contain only relevant codes

2. **Model Architecture**
   - **Base Model**: Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
   - **Task Adaptation**: Sequence classification head for single-label prediction
   - **Output Layer**: Linear layer with CrossEntropyLoss
   - **No Problem Type Specification**: Allows model to infer single-label from data

3. **Training Pipeline** (`train.py:156-235`)
   - Optimized for Apple Silicon (MPS) and CUDA GPUs
   - Single-label metrics (accuracy, weighted F1, macro F1)
   - Top-k accuracy for clinical relevance
   - Automatic device detection and optimization

4. **Evaluation Framework** (`evaluate.py:58-212`)
   - Single-label classification metrics
   - Detailed confusion matrix analysis
   - Per-class performance for most common ICD codes
   - Misclassification pattern analysis

## Technical Implementation

### Data Preprocessing

```python
def preprocess_function(examples, tokenizer, label_to_id, max_length=256):
    """Tokenize texts and encode labels for single-label classification"""
    # Clinical note tokenization
    tokenized = tokenizer(examples["user"], truncation=True, 
                         padding="max_length", max_length=max_length)
    
    # Single-label encoding as class indices
    labels = []
    for codes in examples["codes"]:
        if isinstance(codes, str):
            codes = [codes]
        code = codes[0]  # Take first (and only) code
        labels.append(label_to_id.get(code, 0))
    
    tokenized["labels"] = labels
    return tokenized
```

### Label Space Optimization

```python
# Reduce to top 100 most frequent codes for manageable classification
code_counts = Counter()
for sample in dataset["train"]:
    codes = sample["codes"]
    code_counts.update(codes if isinstance(codes, list) else [codes])

top_codes = {code for code, _ in code_counts.most_common(100)}
```

### Model Configuration

- **Tokenizer**: Bio_ClinicalBERT tokenizer with medical vocabulary
- **Sequence Length**: 256 tokens (optimized for faster training)
- **Classification Head**: Linear layer mapping to 100 classes
- **Loss Function**: CrossEntropyLoss for single-label classification

### Training Specifications

- **Optimizer**: AdamW with default weight decay
- **Learning Rate**: 5e-5 (configurable via CLI)
- **Batch Size**: 32 (MPS/CPU) / 64 (CUDA) 
- **Warmup**: 5% of training steps
- **MPS Optimizations**: Disabled pin_memory and multiprocess dataloading

## Performance Evaluation

### Metrics Implementation

1. **Accuracy**
   - Primary metric for single-label classification
   - Direct comparison of predicted vs true class

2. **F1 Scores**
   - Weighted F1: Weighted by class support
   - Macro F1: Unweighted average across classes
   - Handles class imbalance in reduced label space

3. **Top-k Accuracy**
   - Clinical decision support metric
   - Measures if correct diagnosis appears in top k predictions
   - Evaluated at k=1, 3, 5 for clinical relevance

4. **Confusion Analysis**
   - Most common misclassification pairs
   - Per-class error rates
   - Detailed classification report for common codes

### Updated Compute Metrics
```python
def compute_metrics(eval_pred):
    """Compute single-label classification metrics"""
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=1)
    
    accuracy = (predicted_classes == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_classes, average='weighted', zero_division=0
    )
    
    # Top-k accuracy for clinical relevance
    top_k_accs = []
    for k in [1, 3, 5]:
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
        top_k_correct = [labels[i] in top_k_preds[i] for i in range(len(labels))]
        top_k_accs.append((k, np.mean(top_k_correct)))
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "macro_f1": macro_f1,
        **{f"top_{k}_accuracy": acc for k, acc in top_k_accs}
    }
```

### Baseline Results

**Demonstration Run (5000 samples, 2 epochs, 100 classes):**
- Accuracy: 4.4%
- Weighted F1: 0.0160
- Macro F1: 0.0175
- Top-1 Accuracy: 4.4%
- Top-3 Accuracy: 4.4%
- Top-5 Accuracy: 6.7%

*Note: Low performance due to extreme data scarcity (179 train / 45 test samples after filtering)*

## Production Considerations

### Data Challenges

1. **Extreme Class Imbalance**
   - 100 classes with only 224 total samples after filtering
   - Average of ~2 samples per class
   - Requires significant data augmentation or larger sample sizes

2. **Label Space Reduction Trade-offs**
   - Top 100 codes capture most frequent diagnoses
   - Filtering reduces dataset size dramatically
   - Balance between coverage and trainable data volume

### Scalability Improvements

1. **Device-Specific Optimizations**
   ```python
   dataloader_num_workers=0 if device.type == "mps" else 8,
   dataloader_pin_memory=False if device.type == "mps" else True,
   ```

2. **Memory Efficiency**
   - Reduced sequence length (256 vs 512)
   - Removed multi-label binary matrices
   - Integer labels instead of float arrays

3. **Training Stability**
   - Removed incompatible label smoothing
   - Fixed data type mismatches
   - Proper single-label loss function

### Code Quality Enhancements

- **Type Safety**: Consistent integer labels throughout pipeline
- **Error Handling**: Graceful handling of unknown codes
- **Backward Compatibility**: Evaluation script handles both old and new label formats
- **Clear Warnings**: Informative messages about data limitations

## Lessons Learned

### 1. **Always Verify Data Assumptions**
- Initial multi-label assumption led to training errors
- Comprehensive data exploration revealed true structure
- Single-label approach dramatically simplified implementation

### 2. **Platform-Specific Considerations**
- Apple Silicon (MPS) requires special handling
- PyTorch MPS backend has limitations (no pin_memory, multiprocessing)
- Device-specific optimizations improve stability

### 3. **Metric Selection Matters**
- Multi-label metrics obscured poor performance
- Single-label metrics provide clearer insights
- Top-k accuracy remains relevant for clinical use

## Recommendations for Production

### Immediate Improvements
1. **Increase Training Data**: Use 50,000+ samples for better class coverage
2. **Reduce Label Space**: Consider top 20-50 codes for better sample-to-class ratio
3. **Data Augmentation**: Implement paraphrasing or synthetic data generation

### Architecture Enhancements
1. **Class Weights**: Handle extreme imbalance with weighted loss
2. **Few-Shot Learning**: Consider prototypical networks for rare codes
3. **Hierarchical Classification**: Leverage ICD code structure

### Deployment Considerations
1. **Confidence Thresholds**: Return top-k predictions with probabilities
2. **Unknown Code Handling**: Graceful degradation for out-of-vocabulary codes
3. **Model Versioning**: Track label space changes between deployments

## Conclusion

This pipeline demonstrates the importance of thorough data analysis before implementation. The discovery that the SynthEHR dataset is single-label rather than multi-label led to significant architectural simplifications and more appropriate evaluation metrics. While current performance is limited by data scarcity, the implementation provides a solid foundation for ICD-10 classification with clear paths for improvement through increased data volume and advanced techniques for handling class imbalance.

---

**Repository Structure:**
```
icd10-bert/
├── train.py                    # Single-label training script
├── evaluate.py                 # Single-label evaluation  
├── icd10_classification_colab.ipynb  # GPU-optimized notebook
├── requirements.txt            # Dependency specifications
├── TECHNICAL_DOCUMENTATION.md  # This document
└── model_output/              # Trained model artifacts
```

**Key Dependencies:**
- transformers >= 4.35.0 (HuggingFace model integration)
- datasets >= 2.14.0 (SynthEHR dataset loading)
- scikit-learn >= 1.3.0 (Single-label evaluation metrics)
- torch >= 2.0.0 (Neural network framework)
