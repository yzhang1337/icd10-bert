#!/usr/bin/env python3
"""
Data Preparation Script for Promptfoo ICD-10 Evaluation
Converts SynthEHR dataset to promptfoo-compatible test format
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
import pandas as pd


def load_synthehr_data(sample_size: int = None, test_split_only: bool = True) -> List[Dict[str, Any]]:
    """Load SynthEHR dataset and extract test cases
    
    Args:
        sample_size: Maximum number of samples to load
        test_split_only: Whether to use only test split (matching model training)
        
    Returns:
        List of test cases with clinical notes and ICD codes
    """
    print("Loading SynthEHR dataset...")
    
    try:
        dataset = load_dataset("FiscaAI/synth-ehr-icd10cm-prompt")
        print(f"Dataset loaded: {len(dataset['train'])} samples")
        
        # Create the same train/test split used in model training
        train_size = int(0.8 * len(dataset["train"]))
        dataset_splits = dataset["train"].train_test_split(train_size=train_size, seed=42)
        
        if test_split_only:
            # Use test split to match evaluation in evaluate.py
            test_data = dataset_splits["test"]
            print(f"Using test split: {len(test_data)} samples")
        else:
            # Use full dataset
            test_data = dataset["train"]
            print(f"Using full dataset: {len(test_data)} samples")
        
        # Limit sample size if specified
        if sample_size and sample_size < len(test_data):
            test_data = test_data.select(range(sample_size))
            print(f"Limited to {sample_size} samples")
        
        # Convert to list of dictionaries
        test_cases = []
        for i, example in enumerate(test_data):
            clinical_note = example["user"]
            icd_codes = example["codes"]
            
            # Ensure codes is a list
            if isinstance(icd_codes, str):
                icd_codes = [icd_codes]
            
            test_case = {
                "id": f"synthehr_{i}",
                "clinical_note": clinical_note,
                "expected_codes": icd_codes,
                "metadata": {
                    "source": "SynthEHR",
                    "note_length": len(clinical_note),
                    "num_codes": len(icd_codes)
                }
            }
            test_cases.append(test_case)
        
        print(f"Prepared {len(test_cases)} test cases")
        return test_cases
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def create_promptfoo_test_cases(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert test cases to promptfoo format
    
    Args:
        test_cases: List of test cases from SynthEHR
        
    Returns:
        List of test cases in promptfoo format
    """
    promptfoo_cases = []
    
    for case in test_cases:
        promptfoo_case = {
            # Variables that will be substituted into prompts
            "vars": {
                "clinical_note": case["clinical_note"]
            },
            # Expected output for evaluation
            "assert": [
                {
                    "type": "python",
                    "value": case["expected_codes"]
                }
            ],
            # Metadata for analysis
            "metadata": case["metadata"]
        }
        promptfoo_cases.append(promptfoo_case)
    
    return promptfoo_cases


def save_test_cases(test_cases: List[Dict[str, Any]], output_path: str):
    """Save test cases in YAML format for promptfoo
    
    Args:
        test_cases: List of promptfoo test cases
        output_path: Path to save the test cases
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the full test configuration
    test_config = {
        "description": "ICD-10 diagnosis code extraction test cases from SynthEHR dataset",
        "tests": test_cases
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved {len(test_cases)} test cases to {output_file}")


def create_sample_test_cases() -> List[Dict[str, Any]]:
    """Create a small set of sample test cases for quick testing"""
    sample_cases = [
        {
            "clinical_note": "Patient presents with acute chest pain and shortness of breath. ECG shows ST elevation in leads II, III, aVF consistent with inferior myocardial infarction.",
            "expected_codes": ["I21.19", "R06.02", "R50.9"]
        },
        {
            "clinical_note": "45-year-old female with type 2 diabetes mellitus, poorly controlled. HbA1c is 9.2%. Patient also has diabetic nephropathy.",
            "expected_codes": ["E11.65", "E11.22"]
        },
        {
            "clinical_note": "Routine prenatal visit at 32 weeks gestation. No complications. Fetal growth appropriate for gestational age.",
            "expected_codes": ["Z34.83"]
        },
        {
            "clinical_note": "Patient with chronic obstructive pulmonary disease presents with acute exacerbation. Current smoker, 1 pack per day for 20 years.",
            "expected_codes": ["J44.1", "Z87.891"]
        },
        {
            "clinical_note": "Motor vehicle accident resulting in closed fracture of left femur shaft. Patient conscious and alert.",
            "expected_codes": ["S72.302A", "V89.2XXA"]
        }
    ]
    
    # Convert to full test case format
    test_cases = []
    for i, case in enumerate(sample_cases):
        test_case = {
            "id": f"sample_{i}",
            "clinical_note": case["clinical_note"],
            "expected_codes": case["expected_codes"],
            "metadata": {
                "source": "Manual",
                "note_length": len(case["clinical_note"]),
                "num_codes": len(case["expected_codes"])
            }
        }
        test_cases.append(test_case)
    
    return test_cases


def analyze_dataset(test_cases: List[Dict[str, Any]]):
    """Analyze the test dataset and print statistics"""
    print("\n=== Dataset Analysis ===")
    
    # Basic statistics
    print(f"Total test cases: {len(test_cases)}")
    
    # Note length statistics
    note_lengths = [case["metadata"]["note_length"] for case in test_cases]
    print(f"Note length - Mean: {sum(note_lengths) / len(note_lengths):.1f}, "
          f"Min: {min(note_lengths)}, Max: {max(note_lengths)}")
    
    # Code count statistics
    code_counts = [case["metadata"]["num_codes"] for case in test_cases]
    print(f"Codes per note - Mean: {sum(code_counts) / len(code_counts):.1f}, "
          f"Min: {min(code_counts)}, Max: {max(code_counts)}")
    
    # Most common codes
    all_codes = []
    for case in test_cases:
        all_codes.extend(case["expected_codes"])
    
    from collections import Counter
    code_counter = Counter(all_codes)
    print(f"\nUnique ICD codes: {len(code_counter)}")
    print("Most common codes:")
    for code, count in code_counter.most_common(10):
        print(f"  {code}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for promptfoo ICD-10 evaluation")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of test cases to prepare (default: 100)")
    parser.add_argument("--output_dir", default="../datasets",
                        help="Output directory for test cases")
    parser.add_argument("--use_sample", action="store_true",
                        help="Use manual sample cases instead of SynthEHR")
    parser.add_argument("--full_dataset", action="store_true",
                        help="Use full dataset instead of test split only")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze the dataset, don't save test cases")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_sample:
        # Use manual sample cases
        print("Creating sample test cases...")
        test_cases = create_sample_test_cases()
    else:
        # Load from SynthEHR dataset
        test_cases = load_synthehr_data(
            sample_size=args.sample_size,
            test_split_only=not args.full_dataset
        )
    
    # Analyze dataset
    analyze_dataset(test_cases)
    
    if not args.analyze_only:
        # Convert to promptfoo format
        promptfoo_cases = create_promptfoo_test_cases(test_cases)
        
        # Save test cases
        output_file = output_dir / "test_cases.yaml"
        save_test_cases(promptfoo_cases, output_file)
        
        # Also save a JSON version for easier programmatic access
        json_file = output_dir / "test_cases.json"
        with open(json_file, 'w') as f:
            json.dump({
                "description": "ICD-10 test cases from SynthEHR dataset",
                "test_cases": test_cases,
                "promptfoo_format": promptfoo_cases
            }, f, indent=2)
        
        print(f"Also saved JSON format to {json_file}")
        
        print("\n=== Ready for Promptfoo Evaluation ===")
        print(f"Test cases: {len(promptfoo_cases)}")
        print(f"Next steps:")
        print(f"1. cd {Path(__file__).parent.parent}")
        print(f"2. promptfoo eval")
        print(f"3. promptfoo view")


if __name__ == "__main__":
    main()