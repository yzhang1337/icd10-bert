# ICD-10 Model Comparison with Promptfoo

This directory contains a comprehensive evaluation framework for comparing fine-tuned ICD-10 classification models against prompt-based LLM approaches using the [Promptfoo](https://promptfoo.dev/) evaluation framework.

## 🎯 Overview

Compare your fine-tuned Bio_ClinicalBERT model against state-of-the-art LLMs (GPT-4, Claude, etc.) for ICD-10 diagnosis code extraction from clinical notes.

## 🏗️ Architecture

```
llm_eval_test/
├── promptfooconfig.yaml          # Main promptfoo configuration
├── providers/
│   └── finetune_provider.py      # Custom provider for fine-tuned model
├── prompts/
│   ├── basic_extraction.txt      # Simple ICD extraction prompt
│   ├── few_shot.txt             # Few-shot examples
│   ├── chain_of_thought.txt     # CoT reasoning prompt
│   └── structured_output.txt    # JSON format prompt
├── evaluators/
│   └── icd_metrics.py           # Custom medical evaluation metrics
├── datasets/
│   └── test_cases.yaml          # Test cases from SynthEHR data
├── scripts/
│   ├── prepare_data.py          # Data preparation
│   ├── run_comparison.py        # Orchestrate evaluations
│   ├── generate_report.py       # Medical analysis reports
│   └── visualize_results.py     # Advanced visualizations
├── config/
│   └── model_config.json        # Configuration settings
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install promptfoo globally
npm install -g promptfoo

# Install Python dependencies
pip install -r requirements.txt

# Verify promptfoo installation
promptfoo --version
```

### 2. Set Up API Keys

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Prepare Test Data

```bash
# Create test cases from SynthEHR dataset
python scripts/prepare_data.py --sample_size 100

# Or use sample test cases for quick testing
python scripts/prepare_data.py --use_sample
```

### 4. Run Evaluations

```bash
# Quick single evaluation
promptfoo eval

# Comprehensive comparison suite
python scripts/run_comparison.py --sample_size 100

# View results in web UI
promptfoo view
```

## 📊 Evaluation Approaches

### Providers Tested

1. **Fine-tuned Bio_ClinicalBERT** - Your trained model
2. **GPT-4 Turbo** - OpenAI's latest model
3. **GPT-3.5 Turbo** - Cost-effective OpenAI option
4. **Claude 3 Sonnet** - Anthropic's balanced model
5. **Claude 3 Haiku** - Anthropic's fast model

### Prompt Strategies

1. **Basic Extraction** - Direct ICD code extraction
2. **Few-shot Examples** - Learning from medical examples
3. **Chain of Thought** - Step-by-step clinical reasoning
4. **Structured Output** - JSON-formatted responses

### Medical Metrics

- **Micro/Macro F1** - Multi-label classification accuracy
- **Top-K Accuracy** - Clinical decision support relevance
- **Precision/Recall** - Diagnostic accuracy measures
- **ICD Category Analysis** - Performance by medical specialty
- **Error Pattern Analysis** - Clinical insight into failures

## 🔬 Advanced Usage

### Custom Test Data

```bash
# Use your own clinical notes
python scripts/prepare_data.py \
    --sample_size 500 \
    --full_dataset \
    --output_dir ./datasets
```

### Specific Provider Testing

```bash
# Test only fine-tuned model vs GPT-4
python scripts/run_comparison.py \
    --providers finetune-bio-clinical-bert gpt-4-turbo \
    --single_run
```

### Generate Comprehensive Reports

```bash
# Medical analysis with visualizations
python scripts/generate_report.py results/*.json

# Create advanced visualizations
python scripts/visualize_results.py results/medical_analysis_report.json
```

### Configuration Customization

Edit `promptfooconfig.yaml` to:
- Add new LLM providers
- Modify prompt templates
- Adjust evaluation metrics
- Configure output formats

## 📈 Understanding Results

### Key Performance Indicators

1. **Micro F1 Score** - Overall diagnostic accuracy
2. **Top-5 Accuracy** - Clinical usability (finding correct diagnosis in top 5)
3. **Precision** - Avoiding false diagnoses
4. **Recall** - Catching all relevant diagnoses

### Clinical Interpretation

- **High Precision, Low Recall** - Conservative, misses diagnoses
- **High Recall, Low Precision** - Liberal, over-diagnoses
- **Balanced F1** - Good overall performance
- **High Top-K** - Clinically useful as decision support

### Cost Analysis

Results include API costs for LLM providers vs. free local inference for fine-tuned models.

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```bash
   # Check model path in promptfooconfig.yaml
   # Ensure ../model_output_demo exists
   ```

2. **API Rate Limits**
   ```bash
   # Reduce sample size or add delays
   python scripts/run_comparison.py --sample_size 50
   ```

3. **Memory Issues**
   ```bash
   # Use smaller batch sizes
   # Edit providers/finetune_provider.py
   ```

### Dataset Issues

```bash
# Analyze dataset before evaluation
python scripts/prepare_data.py --analyze_only

# Use manual samples if SynthEHR unavailable
python scripts/prepare_data.py --use_sample
```

## 📚 Output Files

### Results Structure

```
results/
├── evaluation_*.json           # Raw promptfoo results
├── medical_analysis_report.json # Medical-specific analysis
├── medical_analysis_report.md   # Human-readable report
├── comparison_summary.json      # High-level comparison
└── visualizations/
    ├── provider_comparison.png  # Performance charts
    ├── icd_category_analysis.png # Medical specialty analysis
    ├── interactive_dashboard.html # Interactive plots
    └── error_analysis.png       # Error pattern analysis
```

### Key Metrics Files

- **Promptfoo Results** - Raw evaluation data
- **Medical Analysis** - Clinical performance insights
- **Comparison Summary** - Executive summary
- **Visualizations** - Charts and interactive dashboards

## 🎯 Best Practices

### For Reliable Results

1. **Use adequate sample size** (≥100 test cases)
2. **Test multiple prompt strategies** for LLMs
3. **Include clinical experts** in result interpretation
4. **Consider cost-performance tradeoffs**
5. **Validate on real clinical data** before deployment

### For Development

1. **Start with sample data** for quick iteration
2. **Use single provider tests** during development
3. **Monitor API costs** during large evaluations
4. **Cache results** to avoid re-evaluation

## 🤝 Integration

### With Existing Workflows

- **CI/CD Integration** - Automated model comparison
- **Clinical Validation** - Expert review workflows
- **Performance Monitoring** - Track model degradation
- **A/B Testing** - Production deployment strategies

### API Integration

The fine-tuned model provider can be extended for:
- REST API deployment
- Batch processing pipelines
- Real-time inference systems
- Clinical decision support integration

## 📞 Support

### Getting Help

1. **Check promptfoo docs** - https://promptfoo.dev/docs
2. **Review error logs** in results directory
3. **Verify API keys** and model paths
4. **Test with sample data** first

### Contributing

- Add new prompt strategies in `prompts/`
- Extend evaluation metrics in `evaluators/`
- Add new LLM providers in `promptfooconfig.yaml`
- Improve visualizations in `scripts/visualize_results.py`

## 🎉 Next Steps

1. **Run your first evaluation** with sample data
2. **Compare fine-tuned vs LLM performance**
3. **Analyze results** with medical experts
4. **Optimize prompt strategies** for better LLM performance
5. **Deploy best performing approach** to production

---

**Happy Evaluating!** 🚀

For questions or issues, please check the troubleshooting section or review the promptfoo documentation.