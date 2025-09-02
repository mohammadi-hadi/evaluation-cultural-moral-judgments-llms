#!/usr/bin/env python3
"""
Repository Organization Script
Organizes files into proper directory structure
"""

import shutil
from pathlib import Path

def organize_repository():
    """Organize repository files into logical structure"""
    
    # File organization mapping
    organization = {
        'src/core': [
            'moral_alignment_tester.py',
            'wvs_processor.py',
            'model_judge.py',
            'run_full_validation.py',
            'validation_suite.py'
        ],
        'src/evaluation': [
            'human_judge_dashboard.py',
            'human_dashboard.py',
            'cross_evaluation.py',
            'conflict_resolver.py'
        ],
        'src/visualization': [
            'moral_visualization.py',
            'visualization_engine.py',
            'paper_outputs.py',
            'output_generator.py'
        ],
        'src/utils': [
            'data_storage.py',
            'environment_manager.py',
            'prompts_manager.py',
            'env_loader.py'
        ],
        'tests': [
            'test_human_dashboard.py',
            'test_validation_demo.py',
            'test_openai_simple.py',
            'test_openai_models.py',
            'test_quick_demo.py'
        ],
        'demos': [
            'demo_with_conflicts.py',
            'simulate_evaluation.py',
            'preview_dashboard_content.py',
            'generate_paper_demo.py',
            'surf_quick_run.py'
        ],
        'docs': [
            'DASHBOARD_README.md',
            'DASHBOARD_LAUNCH_SUMMARY.md',
            'HUMAN_JUDGE_GUIDE.md',
            'VALIDATION_SYSTEM_COMPLETE.md',
            'VALIDATION_REPORT.md',
            'README_IMPLEMENTATION.md',
            'README_ENHANCED_FEATURES.md',
            'README_COMPLETE.md',
            'PLOTS_FIXED.md'
        ]
    }
    
    # Copy files to new locations (don't move yet, just copy for safety)
    for directory, files in organization.items():
        Path(directory).mkdir(parents=True, exist_ok=True)
        for file in files:
            source = Path(file)
            if source.exists():
                dest = Path(directory) / source.name
                try:
                    shutil.copy2(source, dest)
                    print(f"✅ Copied {file} -> {directory}/")
                except Exception as e:
                    print(f"⚠️  Could not copy {file}: {e}")
    
    # Create __init__.py files for Python packages
    for dir in ['src', 'src/core', 'src/evaluation', 'src/visualization', 'src/utils']:
        init_file = Path(dir) / '__init__.py'
        init_file.touch()
        print(f"✅ Created {init_file}")
    
    print("\n✅ Repository organized successfully!")
    print("\nDirectory structure:")
    print("""
    Project06/
    ├── src/              # Source code
    │   ├── core/         # Core components
    │   ├── evaluation/   # Evaluation systems
    │   ├── visualization/# Visualization tools
    │   └── utils/        # Utilities
    ├── tests/            # Test files
    ├── demos/            # Demo scripts
    ├── docs/             # Documentation
    ├── data/             # Data files
    └── outputs/          # Generated outputs
    """)

if __name__ == "__main__":
    organize_repository()