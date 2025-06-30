#!/usr/bin/env python3
"""
Orchestrate Promptfoo Evaluation Runs for ICD-10 Model Comparison
Runs multiple evaluation configurations and generates comparison reports
"""

import json
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import yaml
import time


class PromptfooRunner:
    """Orchestrates promptfoo evaluation runs"""
    
    def __init__(self, config_dir: Path = None):
        """Initialize runner
        
        Args:
            config_dir: Directory containing promptfoo configuration
        """
        self.config_dir = config_dir or Path(__file__).parent.parent
        self.results_dir = self.config_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
    def check_promptfoo_installation(self) -> bool:
        """Check if promptfoo is installed and accessible"""
        try:
            result = subprocess.run(["promptfoo", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Promptfoo version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Promptfoo not found. Install with: npm install -g promptfoo")
            return False
    
    def prepare_test_data(self, sample_size: int = 100, use_sample: bool = False):
        """Prepare test data using the data preparation script
        
        Args:
            sample_size: Number of test cases to prepare
            use_sample: Whether to use manual sample cases
        """
        print(f"📊 Preparing test data (sample_size={sample_size})...")
        
        prep_script = self.config_dir / "scripts" / "prepare_data.py"
        cmd = [
            sys.executable, str(prep_script),
            "--sample_size", str(sample_size),
            "--output_dir", str(self.config_dir / "datasets")
        ]
        
        if use_sample:
            cmd.append("--use_sample")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Test data prepared successfully")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to prepare test data: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def run_evaluation(self, 
                      providers: List[str] = None,
                      prompts: List[str] = None,
                      config_file: str = "promptfooconfig.yaml",
                      output_name: str = None) -> str:
        """Run promptfoo evaluation
        
        Args:
            providers: List of provider IDs to test
            prompts: List of prompt IDs to test  
            config_file: Configuration file name
            output_name: Custom output name for results
            
        Returns:
            Path to results file
        """
        print(f"🚀 Running promptfoo evaluation...")
        
        # Change to config directory
        original_cwd = Path.cwd()
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            # Build command
            cmd = ["promptfoo", "eval", "-c", str(config_path)]
            
            # Add provider filters
            if providers:
                for provider in providers:
                    cmd.extend(["--filter-providers", provider])
            
            # Add prompt filters  
            if prompts:
                for prompt in prompts:
                    cmd.extend(["--filter-prompts", prompt])
            
            # Set output file
            if output_name:
                output_file = self.results_dir / f"{output_name}.json"
                cmd.extend(["--output", str(output_file)])
            else:
                # Use timestamp-based name
                timestamp = int(time.time())
                output_file = self.results_dir / f"evaluation_{timestamp}.json"
                cmd.extend(["--output", str(output_file)])
            
            print(f"Command: {' '.join(cmd)}")
            
            # Run evaluation
            os.chdir(self.config_dir)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("✅ Evaluation completed successfully")
            print(f"Results saved to: {output_file}")
            
            return str(output_file)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
        finally:
            os.chdir(original_cwd)
    
    def run_comparison_suite(self, sample_size: int = 100) -> Dict[str, str]:
        """Run a comprehensive comparison suite
        
        Args:
            sample_size: Number of test cases to use
            
        Returns:
            Dictionary mapping comparison names to result file paths
        """
        print("🏆 Running comprehensive comparison suite...")
        
        # Prepare data first
        self.prepare_test_data(sample_size=sample_size)
        
        results = {}
        
        # Configuration for different comparison runs
        comparisons = [
            {
                "name": "fine_tuned_vs_gpt4",
                "providers": ["finetune-bio-clinical-bert", "gpt-4-turbo"],
                "prompts": ["basic_extraction"],
                "description": "Fine-tuned model vs GPT-4 with basic prompt"
            },
            {
                "name": "prompt_strategies_gpt4",
                "providers": ["gpt-4-turbo"],
                "prompts": ["basic_extraction", "few_shot", "chain_of_thought", "structured_output"],
                "description": "Different prompting strategies with GPT-4"
            },
            {
                "name": "llm_comparison_basic",
                "providers": ["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"],
                "prompts": ["basic_extraction"],
                "description": "LLM comparison with basic prompt"
            },
            {
                "name": "full_comparison",
                "providers": None,  # All providers
                "prompts": None,    # All prompts
                "description": "Complete comparison matrix"
            }
        ]
        
        for comparison in comparisons:
            print(f"\n📋 Running: {comparison['description']}")
            
            try:
                result_file = self.run_evaluation(
                    providers=comparison["providers"],
                    prompts=comparison["prompts"],
                    output_name=comparison["name"]
                )
                results[comparison["name"]] = result_file
                print(f"✅ {comparison['name']} completed")
                
            except Exception as e:
                print(f"❌ {comparison['name']} failed: {e}")
                results[comparison["name"]] = None
        
        return results
    
    def generate_summary_report(self, result_files: Dict[str, str]):
        """Generate a summary report from multiple evaluation results
        
        Args:
            result_files: Dictionary mapping comparison names to result file paths
        """
        print("📊 Generating summary report...")
        
        summary = {
            "timestamp": time.time(),
            "comparisons": {},
            "overall_insights": []
        }
        
        for name, result_file in result_files.items():
            if not result_file or not Path(result_file).exists():
                continue
            
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                # Extract key metrics
                comparison_summary = self._extract_comparison_metrics(results)
                summary["comparisons"][name] = comparison_summary
                
            except Exception as e:
                print(f"⚠️ Failed to process {name}: {e}")
        
        # Generate insights
        summary["overall_insights"] = self._generate_insights(summary["comparisons"])
        
        # Save summary report
        summary_file = self.results_dir / "comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Summary report saved to: {summary_file}")
        
        # Also create a human-readable version
        self._create_readable_report(summary)
    
    def _extract_comparison_metrics(self, results: Dict) -> Dict:
        """Extract key metrics from promptfoo results"""
        # This would need to be adapted based on actual promptfoo output format
        # For now, return a placeholder structure
        return {
            "providers_tested": len(results.get("providers", [])),
            "prompts_tested": len(results.get("prompts", [])),
            "test_cases": len(results.get("tests", [])),
            "avg_score": 0.0,  # Would calculate from actual results
            "top_performer": "unknown"
        }
    
    def _generate_insights(self, comparisons: Dict) -> List[str]:
        """Generate insights from comparison results"""
        insights = [
            "Comparison analysis completed",
            f"Total comparisons run: {len(comparisons)}",
            "See individual result files for detailed metrics"
        ]
        return insights
    
    def _create_readable_report(self, summary: Dict):
        """Create a human-readable report"""
        report_file = self.results_dir / "comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# ICD-10 Model Comparison Report\n\n")
            f.write(f"Generated: {time.ctime(summary['timestamp'])}\n\n")
            
            f.write("## Comparisons Run\n\n")
            for name, data in summary["comparisons"].items():
                f.write(f"### {name.replace('_', ' ').title()}\n")
                f.write(f"- Providers tested: {data['providers_tested']}\n")
                f.write(f"- Prompts tested: {data['prompts_tested']}\n") 
                f.write(f"- Test cases: {data['test_cases']}\n\n")
            
            f.write("## Key Insights\n\n")
            for insight in summary["overall_insights"]:
                f.write(f"- {insight}\n")
        
        print(f"📖 Readable report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Run promptfoo ICD-10 model comparison")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of test cases to use")
    parser.add_argument("--providers", nargs="+",
                        help="Specific providers to test")
    parser.add_argument("--prompts", nargs="+", 
                        help="Specific prompts to test")
    parser.add_argument("--config_dir", type=Path,
                        help="Directory containing promptfoo config")
    parser.add_argument("--single_run", action="store_true",
                        help="Run single evaluation instead of full suite")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare data, don't run evaluation")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PromptfooRunner(config_dir=args.config_dir)
    
    # Check promptfoo installation
    if not runner.check_promptfoo_installation():
        return 1
    
    try:
        if args.prepare_only:
            # Just prepare data
            runner.prepare_test_data(sample_size=args.sample_size)
            
        elif args.single_run:
            # Run single evaluation
            runner.prepare_test_data(sample_size=args.sample_size)
            result_file = runner.run_evaluation(
                providers=args.providers,
                prompts=args.prompts
            )
            print(f"✅ Single evaluation completed: {result_file}")
            
        else:
            # Run full comparison suite
            result_files = runner.run_comparison_suite(sample_size=args.sample_size)
            runner.generate_summary_report(result_files)
            
            print("\n🎉 Comparison suite completed!")
            print("View results with: promptfoo view")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())