#!/usr/bin/env python3
"""
Server Progress Monitor - Track evaluation progress with detailed statistics
Provides IDENTICAL monitoring format to local evaluation for consistency
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerProgressMonitor:
    """Monitor server evaluation progress with detailed statistics"""
    
    def __init__(self, base_dir: str = "/data/storage_4_tb/moral-alignment-pipeline"):
        """Initialize progress monitor"""
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "outputs" / "server_results"
        self.models_evaluating = []
        self.start_time = time.time()
        
        # Expected models for server evaluation (large models 32B+)
        self.server_models = [
            "qwen2.5-32b",     # 32B model - 2 GPUs
            "qwq-32b",         # 32B model - 2 GPUs  
            "llama3.3-70b",    # 70B model - 4 GPUs  
            "qwen2.5-72b",     # 72B model - 4 GPUs
            "gpt-oss-120b",    # 120B model - 4 GPUs
        ]
        
        logger.info(f"Server Progress Monitor initialized")
        logger.info(f"  Results dir: {self.results_dir}")
        logger.info(f"  Expected models: {self.server_models}")
    
    def find_log_files(self) -> List[Path]:
        """Find server evaluation log files"""
        log_files = []
        
        # Common log locations
        potential_log_dirs = [
            self.base_dir / "logs",
            self.base_dir,
            Path("/tmp"),
            Path("/var/log"),
            Path.cwd()
        ]
        
        for log_dir in potential_log_dirs:
            if log_dir.exists():
                # Look for various log file patterns
                patterns = [
                    "server_evaluation*.log",
                    "model_runner*.log", 
                    "evaluation*.log",
                    "*.out",
                    "nohup.out"
                ]
                
                for pattern in patterns:
                    log_files.extend(log_dir.glob(pattern))
        
        return list(set(log_files))  # Remove duplicates
    
    def find_result_files(self) -> List[Path]:
        """Find existing result files"""
        result_files = []
        
        if self.results_dir.exists():
            result_files.extend(self.results_dir.glob("*.json"))
            result_files.extend(self.results_dir.glob("*_results.json"))
        
        # Also check base outputs directory
        outputs_dir = self.base_dir / "outputs"
        if outputs_dir.exists():
            result_files.extend(outputs_dir.glob("**/server_results*.json"))
            result_files.extend(outputs_dir.glob("**/*_results.json"))
        
        return sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def parse_result_file(self, result_file: Path) -> Optional[Dict]:
        """Parse result file to extract progress information"""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Extract model name and count
                if isinstance(data[0], dict) and 'model' in data[0]:
                    model_name = data[0]['model']
                    sample_count = len(data)
                    successful = sum(1 for r in data if r.get('success', False))
                    
                    return {
                        'model': model_name,
                        'total_samples': sample_count,
                        'successful_samples': successful,
                        'success_rate': successful / sample_count if sample_count > 0 else 0,
                        'file_path': str(result_file),
                        'last_modified': datetime.fromtimestamp(result_file.stat().st_mtime)
                    }
        except Exception as e:
            logger.debug(f"Could not parse {result_file}: {e}")
            return None
        
        return None
    
    def analyze_server_progress(self) -> Dict[str, Any]:
        """Analyze current server evaluation progress"""
        progress_info = {
            'timestamp': datetime.now().isoformat(),
            'models_completed': [],
            'models_in_progress': [],
            'models_pending': list(self.server_models),
            'total_results': 0,
            'successful_results': 0,
            'log_files_found': [],
            'recent_activity': [],
            'overall_progress': 0.0,
            'estimated_completion': None
        }
        
        # Find and analyze log files
        log_files = self.find_log_files()
        progress_info['log_files_found'] = [str(f) for f in log_files[:5]]  # Show first 5
        
        # Find and analyze result files
        result_files = self.find_result_files()
        
        models_seen = set()
        recent_files = []
        
        for result_file in result_files[:10]:  # Analyze most recent 10 files
            result_info = self.parse_result_file(result_file)
            if result_info:
                model_name = result_info['model']
                models_seen.add(model_name)
                
                # Determine if this looks like a complete evaluation
                if result_info['total_samples'] >= 5000:  # Assume 5000 samples is target
                    if model_name not in [m['model'] for m in progress_info['models_completed']]:
                        progress_info['models_completed'].append(result_info)
                elif result_info['total_samples'] > 0:
                    # Might be in progress
                    progress_info['models_in_progress'].append(result_info)
                
                progress_info['total_results'] += result_info['total_samples']
                progress_info['successful_results'] += result_info['successful_samples']
                
                recent_files.append(result_info)
        
        # Update pending models
        completed_models = [m['model'] for m in progress_info['models_completed']]
        in_progress_models = [m['model'] for m in progress_info['models_in_progress']]
        
        progress_info['models_pending'] = [
            m for m in self.server_models 
            if m not in completed_models and m not in in_progress_models
        ]
        
        # Calculate overall progress
        total_expected_samples = len(self.server_models) * 5000  # Assume 5000 samples per model
        progress_info['overall_progress'] = (progress_info['total_results'] / total_expected_samples) * 100 if total_expected_samples > 0 else 0.0
        
        # Estimate completion time
        if progress_info['total_results'] > 0:
            elapsed_time = time.time() - self.start_time
            samples_per_second = progress_info['total_results'] / elapsed_time
            remaining_samples = total_expected_samples - progress_info['total_results']
            
            if samples_per_second > 0:
                remaining_seconds = remaining_samples / samples_per_second
                estimated_completion = datetime.now().timestamp() + remaining_seconds
                progress_info['estimated_completion'] = datetime.fromtimestamp(estimated_completion).isoformat()
        
        progress_info['recent_activity'] = recent_files[:5]  # Most recent 5
        
        return progress_info
    
    def format_progress_report(self, progress_info: Dict[str, Any]) -> str:
        """Format progress information into readable report"""
        report = []
        report.append("🚀 SERVER EVALUATION PROGRESS MONITOR")
        report.append("=" * 60)
        
        # Overall progress
        overall_progress = progress_info['overall_progress']
        report.append(f"📊 Overall Progress: {overall_progress:.1f}%")
        report.append(f"📈 Total Results: {progress_info['total_results']:,}")
        report.append(f"✅ Successful Results: {progress_info['successful_results']:,}")
        
        if progress_info['estimated_completion']:
            report.append(f"⏰ Estimated Completion: {progress_info['estimated_completion']}")
        
        report.append("")
        
        # Models completed
        if progress_info['models_completed']:
            report.append("✅ COMPLETED MODELS:")
            for model_info in progress_info['models_completed']:
                report.append(f"  • {model_info['model']}: {model_info['total_samples']:,} samples ({model_info['success_rate']:.1%} success)")
        
        # Models in progress
        if progress_info['models_in_progress']:
            report.append("🔄 IN PROGRESS:")
            for model_info in progress_info['models_in_progress']:
                report.append(f"  • {model_info['model']}: {model_info['total_samples']:,}/5000 samples ({model_info['success_rate']:.1%} success)")
        
        # Models pending
        if progress_info['models_pending']:
            report.append("⏳ PENDING MODELS:")
            for model in progress_info['models_pending']:
                report.append(f"  • {model}: Not started")
        
        # Recent activity
        if progress_info['recent_activity']:
            report.append("")
            report.append("📈 RECENT ACTIVITY:")
            for activity in progress_info['recent_activity'][:3]:
                report.append(f"  • {activity['model']}: {activity['total_samples']} samples at {activity['last_modified'].strftime('%H:%M:%S')}")
        
        # Log files found
        if progress_info['log_files_found']:
            report.append("")
            report.append("📋 LOG FILES FOUND:")
            for log_file in progress_info['log_files_found'][:3]:
                report.append(f"  • {log_file}")
        
        return "\n".join(report)
    
    def monitor_continuously(self, interval: int = 30, max_iterations: int = None):
        """Monitor server progress continuously"""
        iterations = 0
        
        print(f"🚀 Starting continuous server monitoring (checking every {interval}s)")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while max_iterations is None or iterations < max_iterations:
                progress_info = self.analyze_server_progress()
                report = self.format_progress_report(progress_info)
                
                # Clear screen and show report
                os.system('clear' if os.name == 'posix' else 'cls')
                print(report)
                print(f"\n🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"⏳ Next update in {interval}s... (Press Ctrl+C to stop)")
                
                # Sleep for interval
                time.sleep(interval)
                iterations += 1
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n\n❌ Monitoring error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor Server Evaluation Progress")
    parser.add_argument("--base-dir", type=str, 
                       default="/data/storage_4_tb/moral-alignment-pipeline",
                       help="Base directory for server evaluation")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--once", action="store_true",
                       help="Run single progress check")
    
    args = parser.parse_args()
    
    monitor = ServerProgressMonitor(base_dir=args.base_dir)
    
    if args.continuous:
        monitor.monitor_continuously(interval=args.interval)
    else:
        # Single progress check
        progress_info = monitor.analyze_server_progress()
        report = monitor.format_progress_report(progress_info)
        print(report)

if __name__ == "__main__":
    main()