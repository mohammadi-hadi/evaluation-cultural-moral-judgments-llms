#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard for Parallel Evaluation
Shows progress across API, Local, and Server evaluations
"""

import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class EvaluationMonitor:
    """Real-time monitoring dashboard for moral alignment evaluation"""
    
    def __init__(self, output_dir: str = "outputs/parallel_evaluation"):
        """Initialize monitor
        
        Args:
            output_dir: Directory containing evaluation outputs
        """
        self.output_dir = Path(output_dir)
        self.last_update = {}
        self.clear_command = 'cls' if os.name == 'nt' else 'clear'
    
    def find_latest_run(self) -> Optional[Path]:
        """Find the latest run directory"""
        if not self.output_dir.exists():
            return None
        
        runs = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        return runs[-1] if runs else None
    
    def get_api_status(self, run_dir: Path) -> Dict:
        """Get API evaluation status"""
        api_dir = run_dir / "api"
        status = {'count': 0, 'models': set(), 'cost': 0.0, 'batch_jobs': []}
        
        if not api_dir.exists():
            return status
        
        # Check result files
        for file in api_dir.glob("api_results_*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        status['count'] += len(data)
                        for item in data:
                            if 'model' in item:
                                status['models'].add(item['model'])
                            if 'cost' in item:
                                status['cost'] += item['cost']
            except:
                pass
        
        # Check batch status
        for file in api_dir.glob("batch_status_*.json"):
            try:
                with open(file) as f:
                    batch_jobs = json.load(f)
                    status['batch_jobs'] = batch_jobs
            except:
                pass
        
        status['models'] = list(status['models'])
        return status
    
    def get_local_status(self, run_dir: Path) -> Dict:
        """Get local evaluation status"""
        local_dir = run_dir / "local"
        status = {'count': 0, 'models': set(), 'cached': 0}
        
        if not local_dir.exists():
            return status
        
        # Check result files
        for file in local_dir.glob("*_results.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        status['count'] += len(data)
                        for item in data:
                            if 'model' in item:
                                status['models'].add(item['model'])
                            if item.get('cached', False):
                                status['cached'] += 1
            except:
                pass
        
        # Check cache directory
        cache_dir = local_dir / "cache"
        if cache_dir.exists():
            status['cache_files'] = len(list(cache_dir.glob("*.json")))
        
        status['models'] = list(status['models'])
        return status
    
    def get_server_status(self, run_dir: Path) -> Dict:
        """Get server evaluation status"""
        status = {'status': 'pending', 'message': ''}
        
        # Check for server samples file
        server_samples = run_dir / "server_samples.json"
        if server_samples.exists():
            status['status'] = 'samples_ready'
            status['message'] = 'Samples prepared for server'
        
        # Check for server results
        server_results = run_dir / "server_results.json"
        if server_results.exists():
            try:
                with open(server_results) as f:
                    data = json.load(f)
                    status['status'] = 'completed'
                    status['count'] = len(data) if isinstance(data, list) else 0
            except:
                pass
        
        return status
    
    def get_overall_progress(self, run_dir: Path) -> Dict:
        """Get overall evaluation progress"""
        combined_file = run_dir / "combined_results.json"
        
        if combined_file.exists():
            try:
                with open(combined_file) as f:
                    data = json.load(f)
                    return data.get('summary', {})
            except:
                pass
        
        return {}
    
    def format_time_elapsed(self, run_dir: Path) -> str:
        """Calculate time elapsed since run started"""
        # Extract timestamp from directory name
        try:
            timestamp_str = run_dir.name.replace('run_', '')
            start_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            elapsed = datetime.now() - start_time
            
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            seconds = int(elapsed.total_seconds() % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except:
            return "Unknown"
    
    def display_dashboard(self):
        """Display the monitoring dashboard"""
        os.system(self.clear_command)
        
        print("=" * 80)
        print("MORAL ALIGNMENT EVALUATION MONITOR".center(80))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("=" * 80)
        
        # Find latest run
        run_dir = self.find_latest_run()
        if not run_dir:
            print("\n⚠️  No evaluation runs found")
            print(f"   Looking in: {self.output_dir}")
            return
        
        print(f"\n📂 Current Run: {run_dir.name}")
        print(f"⏱️  Time Elapsed: {self.format_time_elapsed(run_dir)}")
        
        # API Status
        print("\n" + "─" * 80)
        print("API MODELS (OpenAI)")
        print("─" * 80)
        api_status = self.get_api_status(run_dir)
        
        if api_status['count'] > 0 or api_status['batch_jobs']:
            print(f"✅ Results: {api_status['count']}")
            if api_status['models']:
                print(f"   Models: {', '.join(api_status['models'])}")
            if api_status['cost'] > 0:
                print(f"   Cost: ${api_status['cost']:.4f}")
            
            if api_status['batch_jobs']:
                print("\n   Batch Jobs:")
                for job in api_status['batch_jobs']:
                    print(f"   - {job.get('model', 'Unknown')}: {job.get('status', 'Unknown')} ({job.get('batch_id', 'N/A')[:8]}...)")
        else:
            print("⏳ No API results yet or API not configured")
        
        # Local Status
        print("\n" + "─" * 80)
        print("LOCAL MODELS (Ollama)")
        print("─" * 80)
        local_status = self.get_local_status(run_dir)
        
        if local_status['count'] > 0:
            print(f"✅ Results: {local_status['count']}")
            if local_status['models']:
                print(f"   Models: {', '.join(local_status['models'])}")
            if local_status['cached'] > 0:
                print(f"   Cached: {local_status['cached']} ({local_status['cached']/local_status['count']*100:.1f}%)")
            if 'cache_files' in local_status:
                print(f"   Cache Files: {local_status['cache_files']}")
        else:
            print("⏳ No local results yet")
        
        # Server Status
        print("\n" + "─" * 80)
        print("SERVER MODELS (4xA100)")
        print("─" * 80)
        server_status = self.get_server_status(run_dir)
        
        if server_status['status'] == 'completed':
            print(f"✅ Completed: {server_status.get('count', 0)} results")
        elif server_status['status'] == 'samples_ready':
            print(f"📋 {server_status['message']}")
            print(f"   Run notebook on server: server/run_all_models.ipynb")
        else:
            print("⏳ Server evaluation not started")
            print("   Use --server flag or run manually on GPU server")
        
        # Overall Progress
        print("\n" + "─" * 80)
        print("OVERALL PROGRESS")
        print("─" * 80)
        summary = self.get_overall_progress(run_dir)
        
        if summary:
            print(f"Total Evaluations: {summary.get('total_evaluations', 0)}")
            
            if 'by_approach' in summary:
                for approach, info in summary['by_approach'].items():
                    status_icon = "✅" if info['status'] == 'completed' else "⏳" if info['status'] == 'running' else "❌" if info['status'] == 'failed' else "⚠️"
                    print(f"  {status_icon} {approach.upper()}: {info['count']} results ({info['status']})")
        else:
            print("Evaluation in progress...")
        
        # Check for samples file
        samples_file = run_dir / "evaluation_samples.json"
        if samples_file.exists():
            try:
                with open(samples_file) as f:
                    samples = json.load(f)
                    print(f"\nTotal Samples: {len(samples)}")
            except:
                pass
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def run(self, refresh_interval: int = 5):
        """Run the monitoring dashboard
        
        Args:
            refresh_interval: Seconds between updates
        """
        print("Starting evaluation monitor...")
        print(f"Monitoring: {self.output_dir}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("\nPress Ctrl+C to stop\n")
        
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor moral alignment evaluation progress")
    parser.add_argument("--output-dir", default="outputs/parallel_evaluation",
                       help="Output directory to monitor")
    parser.add_argument("--refresh", type=int, default=5,
                       help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor = EvaluationMonitor(output_dir=args.output_dir)
    monitor.run(refresh_interval=args.refresh)

if __name__ == "__main__":
    main()