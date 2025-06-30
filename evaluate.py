#!/usr/bin/env python3
"""
ICD-10 Classification Evaluation Script
Provides detailed evaluation metrics and analysis with progress bars
Optimized for M4 MacBook Pro Max with MPS support
Updated for single-label classification (each sample has exactly one ICD-10 code)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm  # Add progress bar support


def load_model_and_tokenizer(model_dir):
    """Load trained model, tokenizer, and label encoder"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    with open(f"{model_dir}/label_encoder.json", "r") as f:
        label_data = json.load(f)
        # Handle both old format (list) and new format (dict)
        if isinstance(label_data, dict):
            label_to_id = label_data["label_to_id"]
            id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
            label_classes = label_data["labels"]
        else:
            # Old format compatibility
            label_classes = label_data
            label_to_id = {label: idx for idx, label in enumerate(label_classes)}
            id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    return model, tokenizer, label_classes, label_to_id, id_to_label


def predict_batch(texts, model, tokenizer, device, max_length=256):  # Reduced from 512 to match train.py
    """Make predictions on a batch of texts"""
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        # For single-label classification, return raw logits
        logits = outputs.logits.cpu().numpy()
    
    return logits


def evaluate_model(model_dir, dataset_name="FiscaAI/synth-ehr-icd10cm-prompt", 
                   sample_size=None):
    """Comprehensive evaluation of the trained model"""
    
    # Check for GPU/MPS (Metal Performance Shaders for Apple Silicon) - matching train.py
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} - GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Load model
    model, tokenizer, label_classes, label_to_id, id_to_label = load_model_and_tokenizer(model_dir)
    model.to(device)
    print(f"Model loaded with {len(label_classes)} ICD-10 codes")
    
    # Load test data (create test split since dataset only has train)
    print("\nLoading test dataset...")
    full_dataset = load_dataset(dataset_name)
    
    # Apply same filtering as in train.py if using reduced label space
    if sample_size and sample_size <= 10000:  # Assuming we're using filtered data
        # First, compute top codes across entire sample
        from collections import Counter
        code_counts = Counter()
        sample_data = full_dataset["train"].select(range(min(sample_size, len(full_dataset["train"]))))
        
        for sample in sample_data:
            codes = sample["codes"]
            if isinstance(codes, str):
                codes = [codes]
            code_counts.update(codes)
        
        # Keep only top 100 most frequent codes
        top_codes = {code for code, _ in code_counts.most_common(100)}
        
        # Filter dataset
        def filter_and_transform(example):
            codes = example["codes"]
            if isinstance(codes, str):
                codes = [codes]
            filtered_codes = [c for c in codes if c in top_codes]
            if filtered_codes:
                example["codes"] = filtered_codes
                return True
            return False
        
        filtered_dataset = sample_data.filter(filter_and_transform)
        
        # Create test split from filtered data
        train_size = int(0.8 * len(filtered_dataset))
        dataset_splits = filtered_dataset.train_test_split(train_size=train_size, seed=42)
        dataset = dataset_splits["test"]
    else:
        # Use full dataset
        train_size = int(0.8 * len(full_dataset["train"]))
        dataset_splits = full_dataset["train"].train_test_split(train_size=train_size, seed=42)
        dataset = dataset_splits["test"]
    
    print(f"Test dataset size: {len(dataset)} samples")
    
    # Prepare true labels for single-label classification
    print("\nPreparing true labels...")
    true_labels = []
    skipped_samples = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing labels"):
        codes = dataset[i]["codes"]
        if isinstance(codes, str):
            codes = [codes]
        
        # Single-label: take first code
        code = codes[0] if codes else None
        if code and code in label_to_id:
            true_labels.append(label_to_id[code])
        else:
            skipped_samples += 1
            true_labels.append(0)  # Default to first class
    
    true_labels = np.array(true_labels)
    if skipped_samples > 0:
        print(f"Warning: {skipped_samples} samples had codes not in the training vocabulary")
    
    # Make predictions with progress bar
    print("\nMaking predictions...")
    all_logits = []
    batch_size = 32  # Same as train.py
    
    # Calculate total number of batches for progress bar
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with tqdm(total=num_batches, desc="Predicting batches") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i:i+batch_size]["user"]
            logits = predict_batch(batch_texts, model, tokenizer, device, max_length=256)  # Match train.py
            all_logits.append(logits)
            pbar.update(1)
    
    logits = np.vstack(all_logits)
    # For single-label, get predicted class from argmax
    predicted_classes = np.argmax(logits, axis=1)
    
    # Calculate metrics
    print("\n=== Overall Metrics ===")
    accuracy = (predicted_classes == true_labels).mean()
    
    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_classes, average='weighted', zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_classes, average='macro', zero_division=0
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    
    # Top-k accuracy for single-label classification
    print("\n=== Top-K Accuracy ===")
    for k in [1, 3, 5]:
        if k <= logits.shape[1]:  # Ensure k doesn't exceed number of classes
            # Get indices of top-k predictions
            top_k_preds = np.argsort(logits, axis=1)[:, -k:]
            # Check if true label is in top-k
            top_k_correct = np.array([true_labels[i] in top_k_preds[i] for i in range(len(true_labels))])
            top_k_acc = top_k_correct.mean()
            print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
    
    # Per-class performance for most common codes
    print("\n=== Performance on Most Common ICD Codes ===")
    code_frequencies = Counter()
    for codes in tqdm(dataset["codes"], desc="Counting code frequencies"):
        if isinstance(codes, str):
            codes = [codes]
        code_frequencies.update(codes)
    
    common_codes = [code for code, _ in code_frequencies.most_common(10)]
    
    # Generate classification report for common codes
    from sklearn.metrics import classification_report
    # Get indices of common codes
    common_code_indices = [label_to_id[code] for code in common_codes if code in label_to_id]
    
    # Filter predictions and true labels to only include samples with common codes
    mask = np.isin(true_labels, common_code_indices)
    if mask.any():
        filtered_true = true_labels[mask]
        filtered_pred = predicted_classes[mask]
        
        # Map back to code names for readability
        target_names = [code for code in common_codes if code in label_to_id]
        target_indices = [label_to_id[code] for code in target_names]
        
        print("\nClassification Report for Top 10 Most Common Codes:")
        print(classification_report(
            filtered_true, filtered_pred,
            labels=target_indices,
            target_names=target_names,
            zero_division=0
        ))
    
    # Error analysis
    print("\n=== Error Analysis ===")
    
    # Confusion matrix analysis
    from sklearn.metrics import confusion_matrix
    incorrect_mask = predicted_classes != true_labels
    num_errors = incorrect_mask.sum()
    print(f"Total misclassifications: {num_errors} ({num_errors/len(true_labels)*100:.1f}%)")
    
    if num_errors > 0:
        # Analyze misclassifications
        misclassified_true = true_labels[incorrect_mask]
        misclassified_pred = predicted_classes[incorrect_mask]
        
        # Count confusion pairs
        confusion_pairs = Counter()
        for true_idx, pred_idx in zip(misclassified_true, misclassified_pred):
            true_code = id_to_label.get(true_idx, f"Unknown_{true_idx}")
            pred_code = id_to_label.get(pred_idx, f"Unknown_{pred_idx}")
            confusion_pairs[(true_code, pred_code)] += 1
        
        print("\nMost Common Misclassifications (True -> Predicted):")
        for (true_code, pred_code), count in confusion_pairs.most_common(10):
            print(f"  {true_code} -> {pred_code}: {count} times")
        
        # Codes most often misclassified
        misclassified_counts = Counter(misclassified_true)
        print("\nCodes Most Often Misclassified:")
        for idx, count in misclassified_counts.most_common(5):
            code = id_to_label.get(idx, f"Unknown_{idx}")
            total_count = (true_labels == idx).sum()
            print(f"  {code}: {count}/{total_count} times ({count/total_count*100:.1f}%)")
    
    # Save detailed results
    results = {
        "accuracy": float(accuracy),
        "weighted_f1": float(f1),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "macro_f1": float(macro_f1),
        "top_k_accuracy": {
            k: float(np.mean([true_labels[i] in np.argsort(logits, axis=1)[i, -k:] 
                            for i in range(len(true_labels))]))
            for k in [1, 3, 5] if k <= logits.shape[1]
        },
        "num_samples": len(dataset),
        "num_labels": len(label_classes),
        "num_errors": int(num_errors),
        "error_rate": float(num_errors/len(true_labels)),
        "device": str(device),
        "max_length": 256,  # Document the optimization
        "classification_type": "single_label"
    }
    
    with open(f"{model_dir}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Detailed results saved to {model_dir}/evaluation_results.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICD-10 classification model")
    parser.add_argument("--model_dir", default="./model_output", 
                        help="Directory containing trained model")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="Evaluate on subset of test data (use same as training for consistency)")
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_dir=args.model_dir,
            sample_size=args.sample_size
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()