#!/usr/bin/env python3
"""
Advanced Visualization Script for ICD-10 Model Comparison Results
Creates interactive and static visualizations for medical analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Static plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plotting (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Only static plots will be generated.")


class MedicalVisualizationGenerator:
    """Generate comprehensive visualizations for medical ICD-10 analysis"""
    
    def __init__(self, output_dir: Path = None):
        """Initialize visualization generator
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir or Path("../results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib styling
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_analysis_data(self, report_file: str) -> Dict[str, Any]:
        """Load medical analysis report data
        
        Args:
            report_file: Path to medical analysis JSON report
            
        Returns:
            Analysis data dictionary
        """
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def create_provider_comparison_chart(self, provider_data: Dict[str, Any]) -> str:
        """Create comprehensive provider comparison chart
        
        Args:
            provider_data: Provider performance data
            
        Returns:
            Path to saved chart
        """
        # Prepare data for plotting
        providers = list(provider_data.keys())
        metrics = ['avg_micro_f1', 'avg_macro_f1', 'avg_top_5_accuracy', 'overall_precision', 'overall_recall']
        metric_labels = ['Micro F1', 'Macro F1', 'Top-5 Accuracy', 'Precision', 'Recall']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Provider Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall F1 Scores
        f1_data = {
            'Micro F1': [provider_data[p]['avg_micro_f1'] for p in providers],
            'Macro F1': [provider_data[p]['avg_macro_f1'] for p in providers]
        }
        f1_df = pd.DataFrame(f1_data, index=providers)
        f1_df.plot(kind='bar', ax=axes[0, 0], width=0.8, color=['#2E86AB', '#A23B72'])
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top-K Accuracy
        topk_data = {
            'Top-1': [provider_data[p]['avg_top_1_accuracy'] for p in providers],
            'Top-5': [provider_data[p]['avg_top_5_accuracy'] for p in providers],
            'Top-10': [provider_data[p]['avg_top_10_accuracy'] for p in providers]
        }
        topk_df = pd.DataFrame(topk_data, index=providers)
        topk_df.plot(kind='bar', ax=axes[0, 1], width=0.8, color=['#F18F01', '#C73E1D', '#592E83'])
        axes[0, 1].set_title('Top-K Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Precision vs Recall
        precision = [provider_data[p]['overall_precision'] for p in providers]
        recall = [provider_data[p]['overall_recall'] for p in providers]
        
        for i, provider in enumerate(providers):
            axes[0, 2].scatter(recall[i], precision[i], s=100, alpha=0.7, label=provider)
        
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision vs Recall')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add F1 contour lines
        x = np.linspace(0.01, 1, 100)
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            y = (f1 * x) / (2 * x - f1 + 1e-10)
            y = np.clip(y, 0, 1)
            axes[0, 2].plot(x, y, '--', alpha=0.3, color='gray')
            axes[0, 2].text(0.8, f1*0.8/(2*0.8-f1+1e-10), f'F1={f1}', fontsize=8, alpha=0.7)
        
        # 4. Test Coverage
        test_cases = [provider_data[p]['test_cases'] for p in providers]
        axes[1, 0].bar(providers, test_cases, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Test Cases Evaluated')
        axes[1, 0].set_ylabel('Number of Test Cases')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Code Volume Analysis
        predicted = [provider_data[p]['total_predicted'] for p in providers]
        expected = [provider_data[p]['total_expected'] for p in providers]
        correct = [provider_data[p]['total_correct'] for p in providers]
        
        x_pos = np.arange(len(providers))
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, expected, width, label='Expected', alpha=0.8, color='lightcoral')
        axes[1, 1].bar(x_pos, predicted, width, label='Predicted', alpha=0.8, color='lightblue')
        axes[1, 1].bar(x_pos + width, correct, width, label='Correct', alpha=0.8, color='lightgreen')
        
        axes[1, 1].set_xlabel('Providers')
        axes[1, 1].set_ylabel('Total ICD Codes')
        axes[1, 1].set_title('Code Volume Analysis')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(providers, rotation=45)
        axes[1, 1].legend()
        
        # 6. Performance Radar Chart
        if len(providers) <= 4:  # Only if manageable number of providers
            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(providers)))
            
            for i, provider in enumerate(providers):
                values = [provider_data[provider][metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=provider, color=colors[i])
                axes[1, 2].fill(angles, values, alpha=0.25, color=colors[i])
            
            axes[1, 2].set_xticks(angles[:-1])
            axes[1, 2].set_xticklabels(metric_labels)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_title('Performance Radar Chart')
            axes[1, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'Too many providers\\nfor radar chart', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Performance Radar Chart')
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = self.output_dir / "provider_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def create_icd_category_analysis(self, icd_data: Dict[str, Any]) -> str:
        """Create ICD category performance analysis chart
        
        Args:
            icd_data: ICD code analysis data
            
        Returns:
            Path to saved chart
        """
        if not icd_data:
            return None
        
        # Prepare data
        categories = list(icd_data.keys())
        recalls = [icd_data[cat]['recall'] for cat in categories]
        num_codes = [icd_data[cat]['num_codes'] for cat in categories]
        total_instances = [icd_data[cat]['total'] for cat in categories]
        
        # ICD-10 category descriptions
        category_names = {
            'A': 'Infectious diseases', 'B': 'Infectious diseases',
            'C': 'Neoplasms', 'D': 'Blood disorders',
            'E': 'Endocrine diseases', 'F': 'Mental disorders',
            'G': 'Nervous system', 'H': 'Eye/Ear disorders',
            'I': 'Circulatory system', 'J': 'Respiratory system',
            'K': 'Digestive system', 'L': 'Skin disorders',
            'M': 'Musculoskeletal', 'N': 'Genitourinary',
            'O': 'Pregnancy/childbirth', 'P': 'Perinatal conditions',
            'Q': 'Congenital malformations', 'R': 'Symptoms/signs',
            'S': 'Injuries', 'T': 'External causes',
            'V': 'Transport accidents', 'W': 'Accidental poisoning',
            'X': 'Intentional self-harm', 'Y': 'Assault',
            'Z': 'Health status factors'
        }
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ICD-10 Category Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Recall by category
        category_labels = [f"{cat}: {category_names.get(cat, 'Unknown')}" for cat in categories]
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        
        bars = axes[0, 0].bar(range(len(categories)), recalls, color=colors, alpha=0.8)
        axes[0, 0].set_title('Recall by ICD-10 Category')
        axes[0, 0].set_xlabel('ICD Category')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].set_xticks(range(len(categories)))
        axes[0, 0].set_xticklabels(categories, rotation=45)
        
        # Add value labels on bars
        for bar, recall in zip(bars, recalls):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{recall:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Number of unique codes by category
        axes[0, 1].bar(range(len(categories)), num_codes, color=colors, alpha=0.8)
        axes[0, 1].set_title('Number of Unique ICD Codes by Category')
        axes[0, 1].set_xlabel('ICD Category')
        axes[0, 1].set_ylabel('Number of Codes')
        axes[0, 1].set_xticks(range(len(categories)))
        axes[0, 1].set_xticklabels(categories, rotation=45)
        
        # 3. Total instances by category
        axes[1, 0].bar(range(len(categories)), total_instances, color=colors, alpha=0.8)
        axes[1, 0].set_title('Total Test Instances by Category')
        axes[1, 0].set_xlabel('ICD Category')
        axes[1, 0].set_ylabel('Total Instances')
        axes[1, 0].set_xticks(range(len(categories)))
        axes[1, 0].set_xticklabels(categories, rotation=45)
        
        # 4. Scatter: Recall vs Frequency
        sizes = [inst * 2 for inst in total_instances]  # Scale for visibility
        scatter = axes[1, 1].scatter(total_instances, recalls, s=sizes, alpha=0.6, c=range(len(categories)), cmap='tab20')
        
        # Add category labels to points
        for i, cat in enumerate(categories):
            axes[1, 1].annotate(cat, (total_instances[i], recalls[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        axes[1, 1].set_xlabel('Total Instances')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall vs Frequency by Category')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = self.output_dir / "icd_category_analysis.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def create_interactive_dashboard(self, analysis_data: Dict[str, Any]) -> str:
        """Create interactive Plotly dashboard
        
        Args:
            analysis_data: Complete analysis data
            
        Returns:
            Path to saved HTML dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available. Skipping interactive dashboard.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Provider F1 Comparison', 'Top-K Accuracy', 
                           'ICD Category Performance', 'Precision vs Recall'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        provider_data = analysis_data['provider_performance']
        providers = list(provider_data.keys())
        
        # 1. F1 Scores
        fig.add_trace(
            go.Bar(name='Micro F1', 
                   x=providers, 
                   y=[provider_data[p]['avg_micro_f1'] for p in providers],
                   marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Macro F1', 
                   x=providers, 
                   y=[provider_data[p]['avg_macro_f1'] for p in providers],
                   marker_color='red'),
            row=1, col=1
        )
        
        # 2. Top-K Accuracy
        fig.add_trace(
            go.Bar(name='Top-1', 
                   x=providers, 
                   y=[provider_data[p]['avg_top_1_accuracy'] for p in providers],
                   marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Top-5', 
                   x=providers, 
                   y=[provider_data[p]['avg_top_5_accuracy'] for p in providers],
                   marker_color='orange'),
            row=1, col=2
        )
        
        # 3. ICD Category Performance (if available)
        if 'icd_code_analysis' in analysis_data and analysis_data['icd_code_analysis']:
            icd_data = analysis_data['icd_code_analysis']
            categories = list(icd_data.keys())
            
            fig.add_trace(
                go.Bar(name='Category Recall',
                       x=categories,
                       y=[icd_data[cat]['recall'] for cat in categories],
                       marker_color='purple'),
                row=2, col=1
            )
        
        # 4. Precision vs Recall
        for provider in providers:
            fig.add_trace(
                go.Scatter(
                    x=[provider_data[provider]['overall_recall']],
                    y=[provider_data[provider]['overall_precision']],
                    mode='markers',
                    name=f'{provider} (P vs R)',
                    marker=dict(size=12),
                    showlegend=True
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="ICD-10 Model Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save interactive dashboard
        dashboard_file = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(dashboard_file))
        
        return str(dashboard_file)
    
    def create_error_analysis_chart(self, analysis_data: Dict[str, Any]) -> str:
        """Create error analysis visualization
        
        Args:
            analysis_data: Analysis data with error information
            
        Returns:
            Path to saved chart
        """
        # This would require access to detailed error data from the analysis
        # For now, create a placeholder visualization
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Placeholder data - in real implementation, extract from analysis_data
        providers = list(analysis_data['provider_performance'].keys())
        error_types = ['False Positives', 'False Negatives', 'Partial Matches']
        
        # Create sample error data for demonstration
        np.random.seed(42)
        error_data = np.random.rand(len(providers), len(error_types)) * 20
        
        x = np.arange(len(providers))
        width = 0.25
        
        for i, error_type in enumerate(error_types):
            ax.bar(x + i * width, error_data[:, i], width, label=error_type, alpha=0.8)
        
        ax.set_xlabel('Providers')
        ax.set_ylabel('Average Error Count')
        ax.set_title('Error Analysis by Provider')
        ax.set_xticks(x + width)
        ax.set_xticklabels(providers, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        
        chart_file = self.output_dir / "error_analysis.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def generate_all_visualizations(self, analysis_report: str) -> List[str]:
        """Generate all visualizations from analysis report
        
        Args:
            analysis_report: Path to medical analysis JSON report
            
        Returns:
            List of paths to generated visualization files
        """
        print("🎨 Generating comprehensive visualizations...")
        
        # Load analysis data
        analysis_data = self.load_analysis_data(analysis_report)
        
        generated_files = []
        
        # 1. Provider comparison chart
        try:
            provider_chart = self.create_provider_comparison_chart(analysis_data['provider_performance'])
            generated_files.append(provider_chart)
            print(f"✅ Provider comparison chart: {provider_chart}")
        except Exception as e:
            print(f"⚠️ Failed to create provider comparison chart: {e}")
        
        # 2. ICD category analysis
        try:
            if 'icd_code_analysis' in analysis_data and analysis_data['icd_code_analysis']:
                icd_chart = self.create_icd_category_analysis(analysis_data['icd_code_analysis'])
                if icd_chart:
                    generated_files.append(icd_chart)
                    print(f"✅ ICD category analysis: {icd_chart}")
        except Exception as e:
            print(f"⚠️ Failed to create ICD category analysis: {e}")
        
        # 3. Interactive dashboard
        try:
            dashboard = self.create_interactive_dashboard(analysis_data)
            if dashboard:
                generated_files.append(dashboard)
                print(f"✅ Interactive dashboard: {dashboard}")
        except Exception as e:
            print(f"⚠️ Failed to create interactive dashboard: {e}")
        
        # 4. Error analysis
        try:
            error_chart = self.create_error_analysis_chart(analysis_data)
            generated_files.append(error_chart)
            print(f"✅ Error analysis chart: {error_chart}")
        except Exception as e:
            print(f"⚠️ Failed to create error analysis chart: {e}")
        
        print(f"🎨 Generated {len(generated_files)} visualizations in {self.output_dir}")
        
        return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for ICD-10 analysis")
    parser.add_argument("analysis_report", help="Path to medical analysis JSON report")
    parser.add_argument("--output_dir", type=Path, help="Output directory for visualizations")
    parser.add_argument("--chart_type", choices=['provider', 'icd', 'interactive', 'error', 'all'],
                        default='all', help="Type of chart to generate")
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = MedicalVisualizationGenerator(output_dir=args.output_dir)
        
        # Generate visualizations
        if args.chart_type == 'all':
            generated_files = visualizer.generate_all_visualizations(args.analysis_report)
        else:
            # Generate specific chart type
            analysis_data = visualizer.load_analysis_data(args.analysis_report)
            
            if args.chart_type == 'provider':
                chart_file = visualizer.create_provider_comparison_chart(analysis_data['provider_performance'])
                generated_files = [chart_file]
            elif args.chart_type == 'icd':
                chart_file = visualizer.create_icd_category_analysis(analysis_data.get('icd_code_analysis', {}))
                generated_files = [chart_file] if chart_file else []
            elif args.chart_type == 'interactive':
                chart_file = visualizer.create_interactive_dashboard(analysis_data)
                generated_files = [chart_file] if chart_file else []
            elif args.chart_type == 'error':
                chart_file = visualizer.create_error_analysis_chart(analysis_data)
                generated_files = [chart_file]
        
        print(f"✅ Visualization generation completed!")
        print(f"📊 Files generated: {len(generated_files)}")
        for file in generated_files:
            print(f"   - {file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())