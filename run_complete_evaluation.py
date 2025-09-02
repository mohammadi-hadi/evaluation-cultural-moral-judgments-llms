#!/usr/bin/env python3
"""
Complete Parallel Evaluation - Run All Three Approaches
API + Local + Server on the same sample dataset
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from run_parallel_evaluation import ParallelEvaluationOrchestrator

def main():
    """Run complete evaluation across all three approaches"""
    
    print("=" * 80)
    print("COMPLETE MORAL ALIGNMENT EVALUATION")
    print("API + LOCAL + SERVER (Same Sample Dataset)")
    print("=" * 80)
    print()
    
    # Configuration
    dataset_sizes = {
        'quick': {'file': 'sample_data/test_dataset_100.csv', 'samples': 100},
        'medium': {'file': 'sample_data/test_dataset_1000.csv', 'samples': 1000}, 
        'full': {'file': 'sample_data/test_dataset_5000.csv', 'samples': 5000}
    }
    
    # Ask user for evaluation size
    print("Select evaluation size:")
    print("1. Quick test (100 samples, ~30 minutes total)")
    print("2. Medium test (1000 samples, ~3-4 hours total)")
    print("3. Full evaluation (5000 samples, ~10-15 hours total)")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == '1':
            config = dataset_sizes['quick']
            break
        elif choice == '2':
            config = dataset_sizes['medium']
            break
        elif choice == '3':
            config = dataset_sizes['full']
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\nSelected: {config['samples']} samples from {config['file']}")
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    # Check API key
    api_available = bool(os.getenv("OPENAI_API_KEY"))
    print(f"  OpenAI API Key: {'✅ Available' if api_available else '❌ Not set'}")
    
    # Check Ollama
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        local_available = result.returncode == 0
        print(f"  Local Ollama: {'✅ Available' if local_available else '❌ Not running'}")
    except:
        local_available = False
        print("  Local Ollama: ❌ Not installed")
    
    # Server status
    print("  Server Models: ⚠️ Manual deployment required")
    
    if not (api_available or local_available):
        print("\n❌ ERROR: No models available. Please:")
        if not api_available:
            print("  - Set OpenAI API key: export OPENAI_API_KEY='your-key'")
        if not local_available:
            print("  - Start Ollama: ollama serve")
        return
    
    print(f"\n🚀 Starting evaluation with {config['samples']} samples...")
    
    # Initialize orchestrator
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/complete_evaluation_{timestamp}"
    
    orchestrator = ParallelEvaluationOrchestrator(
        dataset_path=config['file'],
        output_dir=output_dir
    )
    
    # Run parallel evaluation
    print("\n" + "="*60)
    print("STARTING PARALLEL EVALUATION")
    print("This will run API, Local, and prepare Server samples simultaneously")
    print("="*60)
    
    # Start monitoring in separate thread
    def monitor_progress():
        """Monitor evaluation progress"""
        time.sleep(10)  # Wait for evaluation to start
        
        while True:
            try:
                # Check if evaluation is still running
                output_path = Path(output_dir)
                if not output_path.exists():
                    time.sleep(5)
                    continue
                
                # Find latest run directory
                runs = sorted(output_path.glob("run_*"))
                if not runs:
                    time.sleep(5)
                    continue
                
                latest_run = runs[-1]
                combined_file = latest_run / "combined_results.json"
                
                if combined_file.exists():
                    try:
                        with open(combined_file) as f:
                            data = json.load(f)
                            summary = data.get('summary', {})
                            
                            print(f"\n📊 Progress Update:")
                            for approach, info in summary.get('by_approach', {}).items():
                                status_icon = "✅" if info['status'] == 'completed' else "🔄" if info['status'] == 'running' else "⏳"
                                print(f"  {status_icon} {approach.upper()}: {info['count']} results ({info['status']})")
                    except:
                        pass
                
                time.sleep(30)  # Update every 30 seconds
                
            except KeyboardInterrupt:
                break
            except:
                time.sleep(5)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    # Run the evaluation
    try:
        orchestrator.run_parallel_evaluation(
            max_samples=config['samples'],
            run_api=api_available,
            run_local=local_available,
            run_server=True  # This creates server samples file
        )
        
        print("\n✅ EVALUATION COMPLETED!")
        
        # Show results summary
        output_path = Path(output_dir)
        runs = sorted(output_path.glob("run_*"))
        if runs:
            latest_run = runs[-1]
            combined_file = latest_run / "combined_results.json"
            
            if combined_file.exists():
                with open(combined_file) as f:
                    data = json.load(f)
                    summary = data.get('summary', {})
                    
                    print(f"\n📊 FINAL RESULTS:")
                    print(f"Total Evaluations: {summary.get('total_evaluations', 0)}")
                    
                    for approach, info in summary.get('by_approach', {}).items():
                        print(f"  {approach.upper()}: {info['count']} results ({info['status']})")
                    
                    # Show server instructions
                    server_samples = latest_run / "server_samples.json"
                    if server_samples.exists():
                        print(f"\n🖥️ SERVER NEXT STEPS:")
                        print(f"1. Copy server package: scp -r server_deployment_package/ your-server:/tmp/")
                        print(f"2. Copy samples: scp {server_samples} your-server:/tmp/server_samples.json")
                        print(f"3. On server: cd /data/storage_4_tb/moral-alignment-pipeline")
                        print(f"4. Run: python run_evaluation.py --dataset /tmp/server_samples.json --models ALL")
        
        print(f"\nResults saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()