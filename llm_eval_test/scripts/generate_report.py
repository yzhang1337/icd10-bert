#!/usr/bin/env python3
"""
Generate Medical-Specific Analysis Reports from Promptfoo Results
Creates detailed reports with clinical insights and performance analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalReportGenerator:
    """Generate medical-specific analysis reports from promptfoo results"""
    
    def __init__(self, results_dir: Path = None):
        """Initialize report generator
        
        Args:
            results_dir: Directory containing promptfoo results
        """
        self.results_dir = results_dir or Path("../results")
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_promptfoo_results(self, result_file: str) -> Dict[str, Any]:
        """Load and parse promptfoo results
        
        Args:
            result_file: Path to promptfoo results JSON
            
        Returns:
            Parsed results dictionary
        """
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def extract_medical_metrics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Extract medical performance metrics into a DataFrame
        
        Args:
            results: Promptfoo results dictionary
            
        Returns:
            DataFrame with performance metrics by provider/prompt combination
        """
        metrics_data = []
        
        # Iterate through results to extract metrics
        # Note: This structure depends on actual promptfoo output format
        # May need adjustment based on real results
        
        for test_result in results.get("results", []):
            provider = test_result.get("provider", "unknown")
            prompt = test_result.get("prompt", "unknown")
            
            # Extract evaluation metrics
            for evaluation in test_result.get("evaluations", []):
                if "details" in evaluation:
                    details = evaluation["details"]
                    
                    metric_row = {
                        "provider": provider,
                        "prompt": prompt,
                        "test_case": test_result.get("test_case_id", "unknown"),
                        "micro_f1": details.get("micro_f1", 0.0),
                        "macro_f1": details.get("macro_f1", 0.0),
                        "precision": details.get("precision", 0.0),
                        "recall": details.get("recall", 0.0),
                        "num_predicted": details.get("num_predicted", 0),
                        "num_expected": details.get("num_expected", 0),
                        "num_correct": details.get("num_correct", 0),
                        "predicted_codes": details.get("predicted_codes", []),
                        "expected_codes": details.get("expected_codes", []),
                        "correct_codes": details.get("correct_codes", []),
                        "missed_codes": details.get("missed_codes", []),
                        "extra_codes": details.get("extra_codes", [])
                    }
                    
                    # Add top-k accuracy metrics
                    for k in [1, 5, 10]:
                        metric_row[f"top_{k}_accuracy"] = details.get(f"top_{k}_accuracy", 0.0)
                    
                    metrics_data.append(metric_row)
        
        return pd.DataFrame(metrics_data)
    
    def analyze_provider_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by provider
        
        Args:
            df: DataFrame with metrics data
            
        Returns:
            Dictionary with provider performance analysis
        """
        provider_analysis = {}
        
        for provider in df["provider"].unique():
            provider_data = df[df["provider"] == provider]
            
            analysis = {
                "test_cases": len(provider_data),
                "avg_micro_f1": provider_data["micro_f1"].mean(),
                "avg_macro_f1": provider_data["macro_f1"].mean(),
                "avg_precision": provider_data["precision"].mean(),
                "avg_recall": provider_data["recall"].mean(),
                "avg_top_1_accuracy": provider_data["top_1_accuracy"].mean(),
                "avg_top_5_accuracy": provider_data["top_5_accuracy"].mean(),
                "avg_top_10_accuracy": provider_data["top_10_accuracy"].mean(),
                "total_predicted": provider_data["num_predicted"].sum(),
                "total_expected": provider_data["num_expected"].sum(),
                "total_correct": provider_data["num_correct"].sum()
            }
            
            # Calculate overall precision/recall across all test cases
            if analysis["total_predicted"] > 0:
                analysis["overall_precision"] = analysis["total_correct"] / analysis["total_predicted"]
            else:
                analysis["overall_precision"] = 0.0
                
            if analysis["total_expected"] > 0:
                analysis["overall_recall"] = analysis["total_correct"] / analysis["total_expected"]
            else:
                analysis["overall_recall"] = 0.0
            
            provider_analysis[provider] = analysis
        
        return provider_analysis
    
    def analyze_prompt_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by prompt strategy
        
        Args:
            df: DataFrame with metrics data
            
        Returns:
            Dictionary with prompt performance analysis
        """
        prompt_analysis = {}
        
        for prompt in df["prompt"].unique():
            prompt_data = df[df["prompt"] == prompt]
            
            analysis = {
                "test_cases": len(prompt_data),
                "avg_micro_f1": prompt_data["micro_f1"].mean(),
                "avg_macro_f1": prompt_data["macro_f1"].mean(),
                "avg_precision": prompt_data["precision"].mean(),
                "avg_recall": prompt_data["recall"].mean(),
                "providers_tested": prompt_data["provider"].nunique(),
                "best_provider": prompt_data.loc[prompt_data["micro_f1"].idxmax(), "provider"] if len(prompt_data) > 0 else "none"
            }
            
            prompt_analysis[prompt] = analysis
        
        return prompt_analysis
    
    def analyze_icd_code_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by ICD code categories
        
        Args:
            df: DataFrame with metrics data
            
        Returns:
            Dictionary with ICD code analysis
        """
        # Collect all ICD codes and their performance
        code_performance = defaultdict(list)
        
        for _, row in df.iterrows():
            predicted = set(row["predicted_codes"])
            expected = set(row["expected_codes"])
            correct = set(row["correct_codes"])
            
            # Track performance for each expected code
            for code in expected:
                was_predicted = code in predicted
                code_performance[code].append({
                    "predicted": was_predicted,
                    "provider": row["provider"],
                    "prompt": row["prompt"]
                })
        
        # Analyze by ICD category (first letter)
        category_analysis = defaultdict(lambda: {"total": 0, "predicted": 0, "codes": set()})
        
        for code, performances in code_performance.items():
            if code and len(code) > 0:
                category = code[0]  # First letter indicates category
                category_analysis[category]["codes"].add(code)
                
                for perf in performances:
                    category_analysis[category]["total"] += 1
                    if perf["predicted"]:
                        category_analysis[category]["predicted"] += 1
        
        # Calculate recall by category
        for category in category_analysis:
            data = category_analysis[category]
            data["recall"] = data["predicted"] / data["total"] if data["total"] > 0 else 0.0
            data["num_codes"] = len(data["codes"])
            data["codes"] = list(data["codes"])  # Convert set to list for JSON serialization
        
        return dict(category_analysis)
    
    def create_performance_visualizations(self, df: pd.DataFrame):
        """Create performance visualization charts
        
        Args:
            df: DataFrame with metrics data
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ICD-10 Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Provider comparison - F1 scores
        provider_metrics = df.groupby('provider')[['micro_f1', 'macro_f1']].mean()
        provider_metrics.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('F1 Scores by Provider')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].legend(['Micro F1', 'Macro F1'])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top-K accuracy comparison
        topk_cols = ['top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy']
        provider_topk = df.groupby('provider')[topk_cols].mean()
        provider_topk.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Top-K Accuracy by Provider')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(['Top-1', 'Top-5', 'Top-10'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Prompt strategy comparison
        if 'prompt' in df.columns and df['prompt'].nunique() > 1:
            prompt_metrics = df.groupby('prompt')['micro_f1'].mean()
            prompt_metrics.plot(kind='bar', ax=axes[1, 0], width=0.8, color='skyblue')
            axes[1, 0].set_title('Micro F1 by Prompt Strategy')
            axes[1, 0].set_ylabel('Micro F1 Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No prompt variation data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Prompt Strategy Comparison')
        
        # 4. Precision vs Recall scatter
        for provider in df['provider'].unique():
            provider_data = df[df['provider'] == provider]
            axes[1, 1].scatter(provider_data['recall'], provider_data['precision'], 
                             label=provider, alpha=0.7, s=50)
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall by Provider')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add diagonal line for F1 reference
        x = np.linspace(0, 1, 100)
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            y = (f1 * x) / (2 * x - f1)
            y = np.clip(y, 0, 1)
            axes[1, 1].plot(x, y, '--', alpha=0.3, color='gray')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance visualizations saved to: {plot_file}")
    
    def generate_comprehensive_report(self, result_files: List[str]) -> str:
        """Generate a comprehensive medical analysis report
        
        Args:
            result_files: List of promptfoo result file paths
            
        Returns:
            Path to generated report
        """
        print("📋 Generating comprehensive medical analysis report...")
        
        # Combine data from all result files
        all_metrics = []
        
        for result_file in result_files:
            if Path(result_file).exists():
                results = self.load_promptfoo_results(result_file)
                df = self.extract_medical_metrics(results)
                all_metrics.append(df)
        
        if not all_metrics:
            raise ValueError("No valid result files found")
        
        # Combine all dataframes
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Generate analyses
        provider_analysis = self.analyze_provider_performance(combined_df)
        prompt_analysis = self.analyze_prompt_performance(combined_df)
        icd_analysis = self.analyze_icd_code_performance(combined_df)
        
        # Create visualizations
        self.create_performance_visualizations(combined_df)
        
        # Generate comprehensive report
        report = {
            "summary": {
                "total_test_cases": len(combined_df),
                "providers_tested": combined_df["provider"].nunique(),
                "prompts_tested": combined_df["prompt"].nunique(),
                "unique_icd_codes": len(set().union(*combined_df["expected_codes"].apply(set))),
                "overall_micro_f1": combined_df["micro_f1"].mean(),
                "overall_macro_f1": combined_df["macro_f1"].mean()
            },
            "provider_performance": provider_analysis,
            "prompt_performance": prompt_analysis,
            "icd_code_analysis": icd_analysis,
            "recommendations": self._generate_recommendations(provider_analysis, prompt_analysis)
        }
        
        # Save detailed JSON report
        report_file = self.output_dir / "medical_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable markdown report
        markdown_file = self._create_markdown_report(report)
        
        print(f"📊 Comprehensive report saved to: {report_file}")
        print(f"📖 Human-readable report: {markdown_file}")
        
        return str(report_file)
    
    def _generate_recommendations(self, provider_analysis: Dict, prompt_analysis: Dict) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        recommendations = []
        
        # Find best performing provider
        best_provider = max(provider_analysis.keys(), 
                          key=lambda p: provider_analysis[p]["avg_micro_f1"])
        recommendations.append(f"Best overall performer: {best_provider}")
        
        # Analyze prompt strategies
        if len(prompt_analysis) > 1:
            best_prompt = max(prompt_analysis.keys(),
                            key=lambda p: prompt_analysis[p]["avg_micro_f1"])
            recommendations.append(f"Most effective prompt strategy: {best_prompt}")
        
        # Clinical insights
        recommendations.extend([
            "Consider ensemble approaches combining top performers",
            "Focus on improving recall for rare ICD codes",
            "Validate results with clinical experts before deployment"
        ])
        
        return recommendations
    
    def _create_markdown_report(self, report: Dict) -> str:
        """Create human-readable markdown report"""
        markdown_file = self.output_dir / "medical_analysis_report.md"
        
        with open(markdown_file, 'w') as f:
            f.write("# ICD-10 Medical Analysis Report\n\n")
            
            # Summary section
            f.write("## Executive Summary\n\n")
            summary = report["summary"]
            f.write(f"- **Total Test Cases**: {summary['total_test_cases']}\n")
            f.write(f"- **Providers Tested**: {summary['providers_tested']}\n")
            f.write(f"- **Prompt Strategies**: {summary['prompts_tested']}\n")
            f.write(f"- **Unique ICD Codes**: {summary['unique_icd_codes']}\n")
            f.write(f"- **Overall Micro F1**: {summary['overall_micro_f1']:.3f}\n")
            f.write(f"- **Overall Macro F1**: {summary['overall_macro_f1']:.3f}\n\n")
            
            # Provider performance
            f.write("## Provider Performance\n\n")
            for provider, data in report["provider_performance"].items():
                f.write(f"### {provider}\n")
                f.write(f"- Micro F1: {data['avg_micro_f1']:.3f}\n")
                f.write(f"- Macro F1: {data['avg_macro_f1']:.3f}\n")
                f.write(f"- Top-5 Accuracy: {data['avg_top_5_accuracy']:.3f}\n")
                f.write(f"- Overall Precision: {data['overall_precision']:.3f}\n")
                f.write(f"- Overall Recall: {data['overall_recall']:.3f}\n\n")
            
            # Recommendations
            f.write("## Clinical Recommendations\n\n")
            for rec in report["recommendations"]:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            f.write("## Methodology\n\n")
            f.write("This analysis uses multi-label classification metrics appropriate for medical diagnosis coding:\n\n")
            f.write("- **Micro F1**: Averaged across all label instances\n")
            f.write("- **Macro F1**: Averaged across all label classes\n")
            f.write("- **Top-K Accuracy**: Clinical decision support metric\n")
            f.write("- **Precision**: Accuracy of positive predictions\n")
            f.write("- **Recall**: Coverage of true diagnoses\n")
        
        return str(markdown_file)


def main():
    parser = argparse.ArgumentParser(description="Generate medical analysis report from promptfoo results")
    parser.add_argument("result_files", nargs="+", help="Promptfoo result JSON files")
    parser.add_argument("--output_dir", type=Path, help="Output directory for reports")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize report generator
        generator = MedicalReportGenerator(results_dir=args.output_dir)
        
        # Generate comprehensive report
        report_file = generator.generate_comprehensive_report(args.result_files)
        
        print("✅ Medical analysis report generated successfully!")
        print(f"📊 Detailed analysis: {generator.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())