#!/usr/bin/env python3
"""
Cleanup Script - Remove duplicate files from root
Keeps organized structure in subdirectories
"""

import os
from pathlib import Path

def cleanup_duplicates():
    """Remove duplicate Python files from root that are now organized"""
    
    print("=" * 60)
    print("REPOSITORY CLEANUP")
    print("=" * 60)
    
    # Files that have been organized and can be removed from root
    files_to_remove = [
        # Core components (now in src/core/)
        'moral_alignment_tester.py',
        'wvs_processor.py', 
        'model_judge.py',
        'run_full_validation.py',
        'validation_suite.py',
        
        # Evaluation systems (now in src/evaluation/)
        'human_judge_dashboard.py',
        'human_dashboard.py',
        'cross_evaluation.py',
        'conflict_resolver.py',
        
        # Visualization (now in src/visualization/)
        'moral_visualization.py',
        'visualization_engine.py',
        'paper_outputs.py',
        'output_generator.py',
        
        # Utils (now in src/utils/)
        'data_storage.py',
        'environment_manager.py',
        'prompts_manager.py',
        'env_loader.py',
        
        # Tests (now in tests/)
        'test_human_dashboard.py',
        'test_validation_demo.py',
        'test_openai_simple.py',
        'test_openai_models.py',
        'test_quick_demo.py',
        
        # Demos (now in demos/)
        'demo_with_conflicts.py',
        'simulate_evaluation.py',
        'preview_dashboard_content.py',
        'generate_paper_demo.py',
        'surf_quick_run.py',
        
        # Organization script (no longer needed)
        'organize_repository.py'
    ]
    
    # Files to keep in root
    files_to_keep = [
        'run_complete_pipeline.py',  # Master runner
        'verify_outputs.py',          # Verification script
        'cleanup_duplicates.py'       # This script
    ]
    
    print("\n📋 Files to remove from root (duplicates):")
    removed_count = 0
    
    for file in files_to_remove:
        file_path = Path(file)
        if file_path.exists():
            print(f"   • {file}")
            removed_count += 1
    
    print(f"\nTotal files to remove: {removed_count}")
    
    print("\n✅ Files to keep in root:")
    for file in files_to_keep:
        if Path(file).exists():
            print(f"   • {file}")
    
    if removed_count > 0:
        print("\n" + "=" * 60)
        response = input("Remove duplicate files from root? (y/n): ").strip().lower()
        
        if response == 'y':
            for file in files_to_remove:
                file_path = Path(file)
                if file_path.exists():
                    try:
                        file_path.unlink()
                        print(f"✅ Removed: {file}")
                    except Exception as e:
                        print(f"❌ Could not remove {file}: {e}")
            
            print("\n✅ Cleanup complete!")
            print("   Organized files remain in subdirectories")
            print("   Use files from src/, tests/, demos/ directories")
        else:
            print("\n❌ Cleanup cancelled")
            print("   Files remain in both root and subdirectories")
    else:
        print("\n✅ Root directory already clean!")
    
    # Show final structure
    print("\n" + "=" * 60)
    print("FINAL REPOSITORY STRUCTURE")
    print("=" * 60)
    print("""
    Project06/
    ├── src/                    # All source code
    │   ├── core/              # Core components
    │   ├── evaluation/        # Evaluation systems
    │   ├── visualization/     # Visualization tools
    │   └── utils/             # Utilities
    ├── tests/                 # Test files
    ├── demos/                 # Demo scripts
    ├── docs/                  # Documentation
    ├── data/                  # Data files
    ├── outputs/               # Generated outputs
    │
    ├── README.md              # Main documentation
    ├── QUICK_START.md         # Quick start guide
    ├── REPOSITORY_SUMMARY.md  # This summary
    ├── requirements.txt       # Dependencies
    ├── .env                   # API keys
    │
    └── Root scripts:
        ├── run_complete_pipeline.py  # Master runner
        ├── verify_outputs.py          # Verification
        └── cleanup_duplicates.py      # This cleanup script
    """)

if __name__ == "__main__":
    cleanup_duplicates()