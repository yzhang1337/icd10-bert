#!/usr/bin/env python3
"""
Custom Promptfoo Provider for Fine-tuned Bio_ClinicalBERT ICD-10 Model
Wraps the trained model to work with promptfoo's evaluation framework
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directories to path to import from main project
sys.path.append(str(Path(__file__).parent.parent.parent))


class FineTunedICDProvider:
    """Promptfoo-compatible provider for fine-tuned ICD-10 classification model"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with model configuration
        
        Args:
            config: Provider configuration from promptfooconfig.yaml
        """
        self.model_dir = config.get("model_dir", "../model_output_demo")
        self.max_length = config.get("max_length", 512)
        self.threshold = config.get("threshold", 0.5)
        self.batch_size = config.get("batch_size", 32)
        
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_classes = None
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model, tokenizer, and label encoder"""
        try:
            model_path = Path(self.model_dir)
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
            print(f"Loading fine-tuned model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label encoder
            label_path = model_path / "label_encoder.json"
            if not label_path.exists():
                raise FileNotFoundError(f"Label encoder not found: {label_path}")
                
            with open(label_path, "r") as f:
                self.label_classes = json.load(f)
            
            print(f"✅ Model loaded successfully:")
            print(f"   Device: {self.device}")
            print(f"   Labels: {len(self.label_classes)} ICD codes")
            print(f"   Model: {self.model.config.name_or_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, clinical_note: str) -> List[str]:
        """Make prediction on a single clinical note
        
        Args:
            clinical_note: Input clinical text
            
        Returns:
            List of predicted ICD-10 codes
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            clinical_note,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
        
        # Convert probabilities to predicted codes
        predicted_indices = np.where(predictions[0] > self.threshold)[0]
        predicted_codes = [self.label_classes[idx] for idx in predicted_indices]
        
        return predicted_codes
    
    def predict_batch(self, clinical_notes: List[str]) -> List[List[str]]:
        """Make predictions on a batch of clinical notes
        
        Args:
            clinical_notes: List of clinical text inputs
            
        Returns:
            List of lists of predicted ICD-10 codes
        """
        all_predictions = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(clinical_notes), self.batch_size):
            batch = clinical_notes[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Convert to ICD codes
            for pred in predictions:
                predicted_indices = np.where(pred > self.threshold)[0]
                predicted_codes = [self.label_classes[idx] for idx in predicted_indices]
                all_predictions.append(predicted_codes)
        
        return all_predictions


# Promptfoo Provider Interface Functions
def call_api(prompt: str, options: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main promptfoo provider interface function
    
    Args:
        prompt: The clinical note text
        options: Provider options from config
        context: Additional context from promptfoo
        
    Returns:
        Response in promptfoo format
    """
    try:
        # Initialize provider if not already done
        if not hasattr(call_api, "provider"):
            config = options or {}
            call_api.provider = FineTunedICDProvider(config)
        
        # Make prediction
        predicted_codes = call_api.provider.predict_single(prompt)
        
        # Format response for promptfoo
        response = {
            "output": predicted_codes,
            "tokenUsage": {
                "total": len(call_api.provider.tokenizer.encode(prompt)),
                "prompt": len(call_api.provider.tokenizer.encode(prompt)),
                "completion": len(predicted_codes)
            },
            "cost": 0.0,  # No cost for local inference
            "cached": False,
            "logProbs": None
        }
        
        return response
        
    except Exception as e:
        return {
            "error": f"Fine-tuned model prediction failed: {str(e)}",
            "output": [],
            "tokenUsage": {"total": 0, "prompt": 0, "completion": 0},
            "cost": 0.0
        }


def get_provider_info() -> Dict[str, Any]:
    """Return provider information for promptfoo"""
    return {
        "id": "finetune-bio-clinical-bert",
        "label": "Fine-tuned Bio_ClinicalBERT",
        "description": "Custom fine-tuned Bio_ClinicalBERT model for ICD-10 classification",
        "supports_completion": True,
        "supports_chat": False,
        "supports_embedding": False,
        "local": True,
        "cost_per_token": 0.0
    }


if __name__ == "__main__":
    # Test the provider locally
    config = {
        "model_dir": "../../model_output_demo",
        "threshold": 0.5,
        "max_length": 512
    }
    
    provider = FineTunedICDProvider(config)
    
    # Test with sample clinical note
    test_note = "Patient presents with acute chest pain and shortness of breath. History of hypertension."
    predictions = provider.predict_single(test_note)
    
    print(f"Test Clinical Note: {test_note}")
    print(f"Predicted ICD Codes: {predictions}")
    
    # Test promptfoo interface
    response = call_api(test_note, config)
    print(f"Promptfoo Response: {response}")