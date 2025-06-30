#!/usr/bin/env python3
"""
ICD-10 Classification Training Script
Can be run locally or in Colab for GPU acceleration
See notebook
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_and_prepare_data(dataset_name="FiscaAI/synth-ehr-icd10cm-prompt", sample_size=50000):
    """Load SynthEHR dataset and prepare for multi-label classification"""
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # Split the train set since there's no test split
    # Default to 5000 samples for fast-ish training on MacBook
    if sample_size:
        dataset["train"] = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
    
    # First, compute top codes across ENTIRE dataset before splitting
    # This ensures test set will have samples with these codes
    code_counts = Counter()
    for sample in dataset["train"]:
        codes = sample["codes"]
        if isinstance(codes, str):
            codes = [codes]
        code_counts.update(codes)
    
    # Keep only top 100 most frequent codes
    top_codes = {code for code, count in code_counts.most_common(100)}
    print(f"Reducing to top {len(top_codes)} most frequent codes")
    
    # Filter dataset to only include samples with at least one top code
    # Also transform codes to only keep the top codes
    def filter_and_transform(example):
        codes = example["codes"]
        if isinstance(codes, str):
            codes = [codes]
        # Keep only codes that are in top_codes
        filtered_codes = [c for c in codes if c in top_codes]
        if filtered_codes:  # Only keep samples with at least one valid code
            example["codes"] = filtered_codes
            return True
        return False
    
    filtered_dataset = dataset["train"].filter(filter_and_transform)
    print(f"Filtered dataset size: {len(filtered_dataset)} (from {len(dataset['train'])})")
    
    # NOW create train/test split (80/20) on filtered data
    train_size = int(0.8 * len(filtered_dataset))
    split_dataset = filtered_dataset.train_test_split(train_size=train_size, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    print(f"Final dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Safety check
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty after filtering. Try increasing sample_size or reducing the number of top codes.")
    
    # Extract unique ICD codes from both splits (now will be max 100)
    all_codes = []
    for codes in train_dataset["codes"]:
        all_codes.extend(codes if isinstance(codes, list) else [codes])
    for codes in test_dataset["codes"]:
        all_codes.extend(codes if isinstance(codes, list) else [codes])
    
    unique_codes = sorted(list(set(all_codes)))
    print(f"Found {len(unique_codes)} unique ICD-10 codes")
    
    # Create label-to-id mapping for single-label classification
    label_to_id = {code: idx for idx, code in enumerate(unique_codes)}
    id_to_label = {idx: code for code, idx in label_to_id.items()}
    
    # Reconstruct dataset dict
    dataset_dict = {"train": train_dataset, "test": test_dataset}
    return dataset_dict, label_to_id, id_to_label

def preprocess_function(examples, tokenizer, label_to_id, max_length=256):
    """Tokenize texts and encode labels for single-label classification"""
    # Tokenize the clinical notes
    tokenized = tokenizer(
        examples["user"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    
    # Convert ICD codes to single-label format (class indices)
    labels = []
    for codes in examples["codes"]:
        if isinstance(codes, str):
            codes = [codes]
        # Since each sample has exactly one code, take the first one
        code = codes[0] if codes else None
        if code and code in label_to_id:
            labels.append(label_to_id[code])
        else:
            # This shouldn't happen after filtering, but just in case
            labels.append(0)  # Default to first class
    
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(eval_pred):
    """Compute single-label classification metrics"""
    predictions, labels = eval_pred
    
    # For single-label, predictions are logits, labels are class indices
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (predicted_classes == labels).mean()
    
    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_classes, average='weighted', zero_division=0
    )
    
    # Calculate macro F1 (unweighted average across classes)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predicted_classes, average='macro', zero_division=0
    )
    
    # Top-k accuracy for single-label
    top_k_accs = []
    for k in [1, 3, 5]:
        if k <= predictions.shape[1]:  # Ensure k doesn't exceed number of classes
            # Get indices of top-k predictions
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
            # Check if true label is in top-k
            top_k_correct = [labels[i] in top_k_preds[i] for i in range(len(labels))]
            top_k_acc = np.mean(top_k_correct)
            top_k_accs.append((k, top_k_acc))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        **{f"top_{k}_accuracy": acc for k, acc in top_k_accs}
    }


def main():
    parser = argparse.ArgumentParser(description="Train ICD-10 classification model")
    parser.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT", 
                        help="Pre-trained model to use (bio_clinicalbert)")
    parser.add_argument("--output_dir", default="./model_output", 
                        help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training (optimized for macbook)")
    parser.add_argument("--epochs", type=int, default=2, 
                        help="Number of training epochs (reduced for speed)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate (higher for faster convergence)")
    parser.add_argument("--max_length", type=int, default=256, 
                        help="Maximum sequence length (reduced for speed)")
    parser.add_argument("--sample_size", type=int, default=5000, 
                        help="Sample size for quick testing (default 5000)")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Check for GPU/MPS (Metal Performance Shaders for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} - GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Load data
    dataset, label_to_id, id_to_label = load_and_prepare_data(sample_size=args.sample_size)
    num_labels = len(label_to_id)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        # This is actually a single-label classification problem!
        # Each sample has exactly one ICD-10 code
    )
    
    # Preprocess datasets
    from datasets import DatasetDict
    dataset_dict = DatasetDict(dataset)
    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer, label_to_id, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Training arguments optimized for macbook
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,  # Use the arg value
        warmup_ratio=0.05,  # Reduced warmup for faster training
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=25,  # Less frequent logging
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=args.fp16 and device.type == "cuda",
        report_to="none",  # Disable wandb/tensorboard for simplicity
        dataloader_num_workers=0 if device.type == "mps" else 8,  # MPS doesn't support multiprocess
        dataloader_pin_memory=False if device.type == "mps" else True,  # Disable pin_memory for MPS
        gradient_accumulation_steps=1,
        remove_unused_columns=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,  # No longer needs mlb parameter
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save label mappings
    with open(f"{args.output_dir}/label_encoder.json", "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "labels": sorted(label_to_id.keys())  # For compatibility
        }, f)
    
    # Final evaluation
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for key, value in results.items():
        if key.startswith("eval_"):
            print(f"{key[5:]}: {value:.4f}")
    
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()