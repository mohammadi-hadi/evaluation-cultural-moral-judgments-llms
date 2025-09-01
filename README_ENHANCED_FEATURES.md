# Enhanced Features for Moral Alignment Pipeline

## 📚 Overview

This document describes the enhanced features added to the Moral Alignment Pipeline for evaluating Large Language Models (LLMs) on cross-cultural moral judgments. These enhancements enable comprehensive LLM peer evaluation, human arbitration, and advanced data management.

## 🆕 New Components

### 1. **Prompts Manager** (`prompts_manager.py`)
Advanced prompt management system with versioning and template storage.

**Features:**
- Versioned prompt templates for all evaluation types
- Separate storage of templates and filled prompts
- Support for multiple template versions per prompt type
- Comprehensive prompt history tracking
- Export functionality for analysis

**Usage:**
```python
from prompts_manager import PromptsManager

pm = PromptsManager()

# Create log-probability prompts
lp_prompts = pm.create_logprob_prompts(
    country="United States",
    topic="abortion",
    model="gpt-4o"
)

# Create chain-of-thought prompt
cot_prompt = pm.create_cot_prompt(
    country="Germany",
    topic="euthanasia",
    model="claude-3.5-sonnet",
    template_version='detailed_v1'
)

# Save session prompts
pm.save_session_prompts("experiment_001")
```

### 2. **Cross-Evaluation System** (`cross_evaluation.py`)
Enables LLMs to evaluate each other's moral judgments.

**Features:**
- Peer review between all model pairs
- Agreement level detection (agree/partial/disagree)
- Inter-rater reliability metrics
- Consensus score calculation
- Outlier model identification
- Comprehensive disagreement tracking

**Usage:**
```python
from cross_evaluation import CrossEvaluator

evaluator = CrossEvaluator(
    models_config=models_config,
    disagreement_threshold=0.5
)

# Run cross-evaluation
await evaluator.run_cross_evaluation(
    models=['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro'],
    results_df=results,
    sample_size=100
)

# Get consensus scores
consensus = evaluator.get_consensus_scores()

# Identify outlier models
outliers = evaluator.identify_outlier_models(threshold=2.0)
```

### 3. **Human Evaluation Dashboard** (`human_dashboard.py`)
Interactive Streamlit interface for human arbitration.

**Features:**
- Review high-priority disagreement cases
- Provide human scores and preferences
- Track evaluator confidence
- Export evaluation data
- Real-time statistics and analytics
- SQLite database for persistent storage

**Usage:**
```bash
# Launch the dashboard
streamlit run human_dashboard.py
```

**Dashboard Sections:**
- **Evaluate Cases**: Navigate through disagreement cases and provide evaluations
- **Statistics**: View real-time analytics on evaluations
- **Export Data**: Download evaluations in CSV/JSON format
- **About**: Documentation and scoring guidelines

### 4. **Data Storage Manager** (`data_storage.py`)
Comprehensive data persistence and organization system.

**Features:**
- Structured directory organization
- SQLite database for metadata
- Experiment run tracking
- Automatic compression for large files
- Caching system with TTL
- Export/import functionality

**Usage:**
```python
from data_storage import DataStorageManager, ModelResult

storage = DataStorageManager()

# Start experiment run
run_id = storage.start_experiment_run(
    models=['gpt-4o', 'claude-3.5-sonnet'],
    config={'sample_size': 100}
)

# Save model result
result = ModelResult(
    model_name='gpt-4o',
    country='China',
    topic='divorce',
    method='cot',
    score=0.5,
    confidence=0.85
)
storage.save_model_result(result)

# Complete and generate summary
storage.complete_experiment_run()
```

### 5. **Conflict Resolver** (`conflict_resolver.py`)
Advanced disagreement detection and resolution system.

**Features:**
- Multiple conflict types: binary, gradient, multimodal
- Severity assessment (low/medium/high/critical)
- 7 resolution strategies:
  - Consensus (median)
  - Weighted average
  - Confidence-weighted
  - Expertise-weighted
  - Cultural context
  - Outlier removal
  - Clustering-based

**Usage:**
```python
from conflict_resolver import ConflictResolver

resolver = ConflictResolver(
    threshold_binary=1.0,
    threshold_gradient=0.5
)

# Detect conflicts
conflicts = resolver.detect_conflicts(results_df, models)

# Resolve conflicts
resolved = resolver.resolve_conflicts(
    conflicts,
    strategy='confidence_weighted',
    confidence_scores=model_confidences
)

# Analyze patterns
analysis = resolver.analyze_conflict_patterns(conflicts)
```

## 📊 Updated Model Configuration

### Latest 2024 Models Added:

**Meta Llama 3.x Series:**
- Llama-3.3-70B-Instruct
- Llama-3.2-90B-Instruct
- Llama-3.2-11B-Instruct
- Llama-3.2-3B-Instruct
- Llama-3.2-1B-Instruct

**Google Gemini:**
- Gemini-1.5-Pro-002
- Gemini-1.5-Flash-002
- Gemini-1.5-Flash-8B
- Gemini-2.0-Flash-Exp
- Gemma-2-27B-IT
- Gemma-2-9B-IT
- Gemma-2-2B-IT

**OpenAI:**
- GPT-4o (November 2024)
- GPT-4o-mini
- GPT-4-Turbo
- o1-preview (Reasoning model)
- o1-mini

**Anthropic:**
- Claude-3.5-Sonnet
- Claude-3.5-Haiku
- Claude-3-Opus

**Mistral:**
- Mistral-Large-2 (123B)
- Mixtral-8x22B-Instruct (141B)
- Mistral-Small-Latest (22B)

**Cohere:**
- Command-R-Plus (104B)
- Command-R (35B)

### New Deployment Profiles:

```yaml
deployment_profiles:
  cutting_edge_2024:  # Latest state-of-the-art models
    models: [llama-3.3-70b, gpt-4o, claude-3.5-sonnet, gemini-1.5-pro]
    
  lightweight:  # Quick testing with small models
    models: [llama-3.2-1b, gemma-2-2b, gpt-4o-mini]
    
  api_only_2024:  # Latest API models only
    models: [gpt-4o, o1-preview, claude-3.5-sonnet, gemini-1.5-pro]
    
  hybrid:  # Mix of local and API models
    models: [llama-3.2-11b, gemma-2-9b, gpt-4o-mini, claude-3.5-haiku]
```

## 🚀 Quick Start with Enhanced Features

### 1. Install Enhanced Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline with Enhancements
```python
# In Jupyter notebook or Python script
from moral_alignment_complete import run_enhanced_pipeline
from prompts_manager import PromptsManager
from cross_evaluation import CrossEvaluator
from data_storage import DataStorageManager
from conflict_resolver import ConflictResolver

# Initialize components
pm = PromptsManager()
storage = DataStorageManager()
evaluator = CrossEvaluator(models_config)
resolver = ConflictResolver()

# Run enhanced pipeline
results = run_enhanced_pipeline(
    models=['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro'],
    enable_cross_evaluation=True,
    enable_conflict_resolution=True,
    save_all_prompts=True
)
```

### 3. Launch Human Dashboard
```bash
streamlit run human_dashboard.py
```

### 4. Access Results
```python
# Load experiment data
experiment_data = storage.load_experiment_run(run_id)

# Get consensus scores
consensus = evaluator.get_consensus_scores()

# View conflicts
conflicts_df = pd.DataFrame(resolver.conflicts)
```

## 📁 Enhanced Directory Structure

```
outputs/
├── scores/
│   ├── logprob/         # Log-probability scores
│   ├── direct/          # Direct elicitation scores
│   └── cot/             # Chain-of-thought scores
├── traces/
│   ├── raw/             # Raw reasoning traces
│   └── processed/       # Processed and structured traces
├── prompts/
│   ├── templates/       # Prompt templates
│   └── filled/          # Filled prompts
├── evaluations/
│   ├── peer/            # LLM cross-evaluations
│   └── human/           # Human evaluations
├── visualizations/
│   ├── figures/         # Generated plots
│   └── reports/         # Analysis reports
├── experiments/
│   ├── runs/            # Individual experiment runs
│   └── logs/            # Experiment logs
├── cross_evaluation/
│   ├── evaluation_pairs.jsonl
│   ├── disagreement_cases.jsonl
│   ├── high_priority_disagreements.json
│   └── consensus_scores.csv
└── cache/               # Cached intermediate results
```

## 🔄 Workflow Integration

### Complete Enhanced Workflow:

1. **Initialize Components**
   - Set up PromptsManager, DataStorageManager, CrossEvaluator, ConflictResolver

2. **Run Model Evaluations**
   - Execute evaluations with latest 2024 models
   - Store all prompts and responses
   - Track experiment metadata

3. **Cross-Evaluation Phase**
   - Models evaluate each other's outputs
   - Calculate inter-rater reliability
   - Identify disagreements

4. **Conflict Resolution**
   - Detect different types of conflicts
   - Apply resolution strategies
   - Flag cases for human review

5. **Human Arbitration**
   - Review high-priority disagreements via dashboard
   - Provide expert evaluations
   - Export human judgments

6. **Analysis & Reporting**
   - Generate consensus scores
   - Analyze conflict patterns
   - Create comprehensive reports

## 📈 Metrics and Analytics

### New Metrics Available:
- **Inter-rater Reliability**: Pearson, Spearman, Kendall's tau, Cohen's kappa
- **Agreement Distribution**: Full/partial/disagreement percentages
- **Confidence Scores**: Model and human confidence tracking
- **Conflict Severity**: Low/medium/high/critical categorization
- **Resolution Success Rate**: Percentage of auto-resolved conflicts
- **Outlier Detection**: Models that consistently disagree with consensus

## 🔧 Configuration Options

### Enhanced Pipeline Configuration:
```python
config = {
    'prompts': {
        'save_all': True,
        'template_versions': ['minimal_v1', 'detailed_v1'],
        'export_format': 'jsonl'
    },
    'cross_evaluation': {
        'enabled': True,
        'disagreement_threshold': 0.5,
        'sample_size': None  # Evaluate all
    },
    'conflict_resolution': {
        'strategies': ['consensus', 'confidence_weighted'],
        'severity_thresholds': {
            'binary': 1.0,
            'gradient': 0.5,
            'multimodal': 0.3
        }
    },
    'human_dashboard': {
        'auto_launch': False,
        'port': 8501,
        'priority_filter': 'high'
    },
    'storage': {
        'compress': True,
        'cache_ttl_hours': 24,
        'cleanup_days': 30
    }
}
```

## 🐛 Troubleshooting

### Common Issues:

**Dashboard Not Loading:**
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Run with specific port
streamlit run human_dashboard.py --server.port 8502
```

**Database Errors:**
```python
# Reset database
from pathlib import Path
db_path = Path("human_evaluations.db")
if db_path.exists():
    db_path.unlink()
```

**Memory Issues with Large Models:**
```python
# Enable 8-bit quantization
config['load_in_8bit'] = True

# Reduce batch size
config['batch_size'] = 1
```

## 📝 API Reference

### Key Classes and Methods:

**PromptsManager:**
- `create_logprob_prompts()`: Generate log-probability prompts
- `create_cot_prompt()`: Generate chain-of-thought prompt
- `save_session_prompts()`: Save all prompts from session
- `get_prompt_statistics()`: Get prompt usage statistics

**CrossEvaluator:**
- `run_cross_evaluation()`: Execute peer evaluation
- `get_consensus_scores()`: Calculate consensus
- `identify_outlier_models()`: Find disagreeing models

**ConflictResolver:**
- `detect_conflicts()`: Identify disagreements
- `resolve_conflicts()`: Apply resolution strategies
- `analyze_conflict_patterns()`: Pattern analysis

**DataStorageManager:**
- `start_experiment_run()`: Initialize experiment
- `save_model_result()`: Store evaluation result
- `complete_experiment_run()`: Finalize and summarize

## 🚦 Next Steps

1. **Run Initial Evaluation**: Test with lightweight profile
2. **Review Disagreements**: Use human dashboard for arbitration
3. **Analyze Results**: Generate reports and visualizations
4. **Iterate**: Refine prompts and evaluation strategies
5. **Scale Up**: Deploy with full model set

## 📧 Support

For issues or questions about enhanced features:
- Check logs in `outputs/experiments/logs/`
- Review documentation in this file
- Contact: h.mohammadi@uu.nl

---

**Note**: These enhanced features significantly expand the pipeline's capabilities for comprehensive moral alignment evaluation, enabling sophisticated analysis of model disagreements and human oversight.