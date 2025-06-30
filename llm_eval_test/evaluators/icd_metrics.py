#!/usr/bin/env python3
"""
Custom Promptfoo Evaluator for Medical ICD-10 Metrics
Implements specialized evaluation metrics for multi-label ICD-10 classification
"""

import json
import re
from typing import List, Dict, Any, Set, Union
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


class ICDMetricsEvaluator:
    """Custom evaluator for ICD-10 classification metrics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize evaluator with configuration
        
        Args:
            config: Configuration dict from promptfoo
        """
        self.config = config or {}
        self.metric = self.config.get("metric", "micro_f1")
        self.k_values = self.config.get("k_values", [1, 5, 10])
        
    def parse_icd_codes(self, text: Union[str, List[str]]) -> Set[str]:
        """Parse ICD-10 codes from various text formats
        
        Args:
            text: Raw text output from model/LLM or list of codes
            
        Returns:
            Set of normalized ICD-10 codes
        """
        if isinstance(text, list):
            # Already a list of codes
            codes = text
        elif isinstance(text, str):
            if text.strip().startswith('{') or text.strip().startswith('['):
                # Try to parse as JSON
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        # Structured output format
                        codes = data.get('all_codes', [])
                        if not codes:
                            # Fallback to combining all code fields
                            codes = []
                            for section in ['primary_diagnoses', 'secondary_diagnoses', 'symptoms_and_findings']:
                                section_data = data.get(section, [])
                                if isinstance(section_data, list):
                                    codes.extend([item.get('code', '') for item in section_data if isinstance(item, dict)])
                    elif isinstance(data, list):
                        codes = data
                    else:
                        codes = []
                except json.JSONDecodeError:
                    # Fallback to regex extraction
                    codes = self._extract_codes_regex(text)
            else:
                # Plain text format
                codes = self._extract_codes_regex(text)
        else:
            codes = []
        
        # Normalize and validate codes
        normalized_codes = set()
        for code in codes:
            if isinstance(code, str):
                normalized_code = self._normalize_icd_code(code.strip())
                if normalized_code and self._is_valid_icd_code(normalized_code):
                    normalized_codes.add(normalized_code)
        
        return normalized_codes
    
    def _extract_codes_regex(self, text: str) -> List[str]:
        """Extract ICD codes using regex patterns"""
        # ICD-10 pattern: Letter followed by 2+ digits, optional periods and more digits
        icd_pattern = r'\b[A-Z]\d{2}(?:\.\d+)*\b'
        codes = re.findall(icd_pattern, text.upper())
        return codes
    
    def _normalize_icd_code(self, code: str) -> str:
        """Normalize ICD-10 code format"""
        # Remove extra whitespace and convert to uppercase
        code = code.strip().upper()
        
        # Remove common prefixes/suffixes
        code = re.sub(r'^(ICD-?10:?\s*)', '', code)
        code = re.sub(r'\s*\(.*\)$', '', code)  # Remove descriptions in parentheses
        
        # Ensure proper format (letter + digits with optional periods)
        match = re.match(r'^([A-Z])(\d{2})(.*)$', code)
        if match:
            letter, first_digits, rest = match.groups()
            # Clean up the rest part
            rest = re.sub(r'[^\d\.]', '', rest)
            return f"{letter}{first_digits}{rest}"
        
        return code
    
    def _is_valid_icd_code(self, code: str) -> bool:
        """Check if code follows ICD-10 format"""
        # Basic validation: starts with letter, followed by at least 2 digits
        return bool(re.match(r'^[A-Z]\d{2}', code)) and len(code) >= 3
    
    def calculate_f1_scores(self, predicted_codes: Set[str], true_codes: Set[str]) -> Dict[str, float]:
        """Calculate micro and macro F1 scores for multi-label classification
        
        Args:
            predicted_codes: Set of predicted ICD codes
            true_codes: Set of true ICD codes
            
        Returns:
            Dictionary with micro and macro F1 scores
        """
        if not true_codes and not predicted_codes:
            return {"micro_f1": 1.0, "macro_f1": 1.0, "precision": 1.0, "recall": 1.0}
        
        if not true_codes:
            return {"micro_f1": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Calculate precision, recall, F1
        intersection = predicted_codes.intersection(true_codes)
        
        if not predicted_codes:
            precision = 0.0
        else:
            precision = len(intersection) / len(predicted_codes)
        
        recall = len(intersection) / len(true_codes)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # For single-instance evaluation, micro and macro F1 are the same
        return {
            "micro_f1": f1,
            "macro_f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def calculate_top_k_accuracy(self, predicted_codes: List[str], true_codes: Set[str], k_values: List[int] = None) -> Dict[str, float]:
        """Calculate top-k accuracy for ranked predictions
        
        Args:
            predicted_codes: List of predicted codes (assumed to be ranked by confidence)
            true_codes: Set of true ICD codes
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with top-k accuracy for each k
        """
        if k_values is None:
            k_values = self.k_values
        
        results = {}
        
        for k in k_values:
            if not true_codes:
                results[f"top_{k}_accuracy"] = 1.0 if not predicted_codes else 0.0
                continue
            
            # Take top k predictions
            top_k_predictions = set(predicted_codes[:k])
            
            # Check if any prediction is correct
            has_correct = len(top_k_predictions.intersection(true_codes)) > 0
            results[f"top_{k}_accuracy"] = 1.0 if has_correct else 0.0
        
        return results
    
    def evaluate_single(self, predicted_output: Any, expected_output: Any) -> Dict[str, Any]:
        """Evaluate a single prediction vs expected output
        
        Args:
            predicted_output: Model/LLM output (text, list, or dict)
            expected_output: Ground truth ICD codes (text, list, or dict)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Parse codes from both outputs
        predicted_codes = self.parse_icd_codes(predicted_output)
        expected_codes = self.parse_icd_codes(expected_output)
        
        results = {}
        
        # Calculate F1 scores
        if self.metric in ["micro_f1", "macro_f1", "f1", "all"]:
            f1_results = self.calculate_f1_scores(predicted_codes, expected_codes)
            results.update(f1_results)
        
        # Calculate top-k accuracy (treat predicted as ranked list)
        if self.metric in ["top_k_accuracy", "top_k", "all"]:
            predicted_list = list(predicted_codes)  # Convert to list for ranking
            topk_results = self.calculate_top_k_accuracy(predicted_list, expected_codes)
            results.update(topk_results)
        
        # Add code-level details
        results.update({
            "predicted_codes": list(predicted_codes),
            "expected_codes": list(expected_codes),
            "correct_codes": list(predicted_codes.intersection(expected_codes)),
            "missed_codes": list(expected_codes - predicted_codes),
            "extra_codes": list(predicted_codes - expected_codes),
            "num_predicted": len(predicted_codes),
            "num_expected": len(expected_codes),
            "num_correct": len(predicted_codes.intersection(expected_codes))
        })
        
        return results


# Promptfoo Evaluator Interface Functions
def get_evaluator(config: Dict[str, Any] = None):
    """Get evaluator instance for promptfoo"""
    return ICDMetricsEvaluator(config)


def evaluate(output: Any, expected: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main promptfoo evaluator interface function
    
    Args:
        output: Model/LLM output to evaluate
        expected: Expected/ground truth output
        context: Additional context from promptfoo
        
    Returns:
        Evaluation results in promptfoo format
    """
    try:
        # Get configuration from context
        config = context.get("config", {}) if context else {}
        
        # Create evaluator
        evaluator = ICDMetricsEvaluator(config)
        
        # Evaluate
        results = evaluator.evaluate_single(output, expected)
        
        # Format for promptfoo
        score = results.get("micro_f1", 0.0)  # Default metric for overall score
        
        response = {
            "pass": score > 0.5,  # Pass threshold
            "score": score,
            "reason": f"Micro F1: {score:.3f}, Predicted: {results['num_predicted']}, Expected: {results['num_expected']}, Correct: {results['num_correct']}",
            "details": results
        }
        
        return response
        
    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Evaluation error: {str(e)}",
            "details": {"error": str(e)}
        }


if __name__ == "__main__":
    # Test the evaluator
    evaluator = ICDMetricsEvaluator()
    
    # Test case 1: Perfect match
    predicted = ["I21.09", "E11.9"]
    expected = ["I21.09", "E11.9"]
    result1 = evaluator.evaluate_single(predicted, expected)
    print("Test 1 (Perfect match):", result1)
    
    # Test case 2: Partial match
    predicted = ["I21.09", "E11.9", "R06.00"]
    expected = ["I21.09", "E10.9"]  # Different diabetes code
    result2 = evaluator.evaluate_single(predicted, expected)
    print("Test 2 (Partial match):", result2)
    
    # Test case 3: Text parsing
    predicted_text = "Based on the clinical note, the relevant ICD-10 codes are:\nI21.09\nE11.9\nR06.00"
    expected_text = '["I21.09", "E10.9"]'
    result3 = evaluator.evaluate_single(predicted_text, expected_text)
    print("Test 3 (Text parsing):", result3)
    
    # Test case 4: JSON parsing
    predicted_json = '{"all_codes": ["I21.09", "E11.9"], "primary_diagnoses": [{"code": "I21.09", "description": "MI"}]}'
    result4 = evaluator.evaluate_single(predicted_json, expected)
    print("Test 4 (JSON parsing):", result4)