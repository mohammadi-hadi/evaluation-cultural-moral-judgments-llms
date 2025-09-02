#!/usr/bin/env python3
"""
Master Pipeline Runner
Runs the complete moral alignment evaluation pipeline
"""

import subprocess
import sys
import time
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout[:500])  # Show first 500 chars
    else:
        print(f"❌ Failed: {description}")
        if result.stderr:
            print(f"Error: {result.stderr[:500]}")
    
    return result.returncode == 0

def main():
    """Run complete pipeline"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     MORAL ALIGNMENT EVALUATION PIPELINE                   ║
    ║     Complete System Execution                             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    print("\n📋 Checking Prerequisites...")
    
    # Check for .env file
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("   Please create .env with: OPENAI_API_KEY=your-key-here")
        return False
    else:
        print("✅ .env file found")
    
    # Check for required directories
    for dir_name in ["src", "outputs", "data", "tests", "demos"]:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {dir_name}/")
    
    # Menu
    print("\n" + "="*60)
    print("SELECT PIPELINE TO RUN:")
    print("="*60)
    print("1. Quick Demo (3 conflicts, ~1 minute)")
    print("2. Full Validation (10 samples, ~5 minutes)")
    print("3. Complete Pipeline (100 samples, ~20 minutes)")
    print("4. Just Launch Dashboard")
    print("5. Verify Outputs")
    print("6. Generate Paper Figures")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        # Quick Demo
        print("\n🚀 Running Quick Demo...")
        
        # Generate conflicts
        if run_command(
            "python demo_with_conflicts.py",
            "Generating demo conflicts"
        ):
            # Check output
            conflict_file = Path("outputs/conflict_demo/conflicts_for_human_review.json")
            if conflict_file.exists():
                with open(conflict_file, 'r') as f:
                    data = json.load(f)
                print(f"\n✅ Generated {data['metadata']['n_conflicts']} conflicts")
                print("\nConflicts ready for human review:")
                for case in data['cases']:
                    print(f"  • {case['case_id']}: {case['severity']} (diff: {case['score_difference']:.2f})")
            
            # Launch dashboard
            print("\n🖥️  Launching dashboard...")
            print("   Open browser to: http://localhost:8501")
            run_command(
                "streamlit run human_judge_dashboard.py",
                "Human Judge Dashboard"
            )
    
    elif choice == "2":
        # Full Validation
        print("\n🚀 Running Full Validation...")
        
        # Process WVS if available
        wvs_file = Path("data/wvs_moral_values_dataset.csv")
        if wvs_file.exists():
            run_command(
                "python wvs_processor.py",
                "Processing WVS data"
            )
        
        # Run validation
        if run_command(
            "python run_full_validation.py --models gpt-3.5-turbo gpt-4o --samples 10",
            "Full validation pipeline"
        ):
            print("\n✅ Validation complete!")
            print("   Check outputs/ directory for results")
    
    elif choice == "3":
        # Complete Pipeline
        print("\n🚀 Running Complete Pipeline...")
        print("⚠️  This will make many API calls and may take 20+ minutes")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            # Process WVS
            run_command(
                "python wvs_processor.py",
                "Processing WVS data"
            )
            
            # Full validation
            run_command(
                "python run_full_validation.py --models gpt-3.5-turbo gpt-4o gpt-4o-mini --samples 100",
                "Complete validation (100 samples)"
            )
            
            # Generate outputs
            run_command(
                "python paper_outputs.py",
                "Generating paper figures"
            )
            
            print("\n✅ Complete pipeline finished!")
    
    elif choice == "4":
        # Just Dashboard
        print("\n🖥️  Launching dashboard...")
        print("   Open browser to: http://localhost:8501")
        run_command(
            "streamlit run human_judge_dashboard.py",
            "Human Judge Dashboard"
        )
    
    elif choice == "5":
        # Verify Outputs
        print("\n🔍 Verifying outputs...")
        run_command(
            "python verify_outputs.py",
            "Output verification"
        )
    
    elif choice == "6":
        # Generate Figures
        print("\n📊 Generating paper figures...")
        run_command(
            "python paper_outputs.py",
            "Paper figure generation"
        )
        
        figures_dir = Path("outputs/paper_demo/figures")
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png"))
            print(f"\n✅ Generated {len(figures)} figures:")
            for fig in figures:
                print(f"   • {fig.name}")
    
    else:
        print("Invalid choice")
        return False
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print("\n📁 Check these directories for outputs:")
    print("   • outputs/conflict_demo/     - Conflicts for review")
    print("   • outputs/peer_review/       - Peer review results")
    print("   • outputs/paper_demo/        - Model results")
    print("   • outputs/plots/             - Visualizations")
    print("\n📊 Database files:")
    print("   • human_evaluations.db       - Human judgments")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)