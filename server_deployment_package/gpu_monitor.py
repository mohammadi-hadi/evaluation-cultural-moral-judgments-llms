#!/usr/bin/env python3
"""
GPU Monitoring and Optimization for Server Models
Real-time monitoring and automatic optimization suggestions
"""

import os
import time
import psutil
import subprocess
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
from datetime import datetime

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization: float
    gpu_utilization: float
    temperature_c: float
    power_watts: float
    power_limit_watts: float
    processes: List[Dict]
    timestamp: str

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization: float
    disk_used_gb: float
    disk_total_gb: float
    disk_utilization: float
    timestamp: str

class GPUMonitor:
    """Real-time GPU monitoring and optimization"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize GPU monitor
        
        Args:
            monitoring_interval: Seconds between monitoring updates
        """
        global NVML_AVAILABLE
        
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.monitoring_thread = None
        
        # Check availability
        if not NVML_AVAILABLE:
            logger.warning("⚠️ nvidia-ml-py3 not available - limited GPU monitoring")
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch CUDA not available - limited GPU monitoring")
        
        # Initialize NVIDIA ML
        if NVML_AVAILABLE:
            try:
                self.device_count = nvml.nvmlDeviceGetCount()
                self.gpu_handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                logger.info(f"📊 GPU Monitor initialized for {self.device_count} GPUs")
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                NVML_AVAILABLE = False
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all devices"""
        if not NVML_AVAILABLE:
            return []
        
        metrics = []
        timestamp = datetime.now().isoformat()
        
        try:
            for i, handle in enumerate(self.gpu_handles):
                # Basic info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory.used / (1024**2)
                memory_total_mb = memory.total / (1024**2)
                memory_util = (memory.used / memory.total) * 100
                
                # Utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                
                # Temperature
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # Power
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power = 0
                    power_limit = 0
                
                # Processes
                try:
                    pids = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    processes = []
                    for pid_info in pids:
                        try:
                            pid = pid_info.pid
                            memory_used = pid_info.usedGpuMemory / (1024**2)  # MB
                            
                            # Try to get process name
                            try:
                                proc = psutil.Process(pid)
                                name = proc.name()
                                cmdline = ' '.join(proc.cmdline()[:3])  # First 3 args
                            except:
                                name = "unknown"
                                cmdline = ""
                            
                            processes.append({
                                'pid': pid,
                                'name': name,
                                'cmdline': cmdline,
                                'gpu_memory_mb': memory_used
                            })
                        except:
                            continue
                except:
                    processes = []
                
                metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    memory_utilization=memory_util,
                    gpu_utilization=gpu_util,
                    temperature_c=temp,
                    power_watts=power,
                    power_limit_watts=power_limit,
                    processes=processes,
                    timestamp=timestamp
                ))
                
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide performance metrics"""
        timestamp = datetime.now().isoformat()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_util = memory.percent
        
        # Disk (current directory)
        disk = psutil.disk_usage('.')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_util = (disk.used / disk.total) * 100
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            memory_utilization=memory_util,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_utilization=disk_util,
            timestamp=timestamp
        )
    
    def print_status(self, gpu_metrics: List[GPUMetrics], system_metrics: SystemMetrics):
        """Print formatted status to console"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🚀 GPU MONITORING - Server Model Evaluation")
        print("=" * 80)
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System overview
        print("💻 SYSTEM METRICS")
        print("-" * 40)
        print(f"CPU Usage:      {system_metrics.cpu_percent:6.1f}%")
        print(f"System Memory:  {system_metrics.memory_used_gb:6.1f}GB / {system_metrics.memory_total_gb:.1f}GB ({system_metrics.memory_utilization:5.1f}%)")
        print(f"Disk Usage:     {system_metrics.disk_used_gb:6.1f}GB / {system_metrics.disk_total_gb:.1f}GB ({system_metrics.disk_utilization:5.1f}%)")
        print()
        
        # GPU details
        if gpu_metrics:
            print("🎮 GPU METRICS")
            print("-" * 40)
            
            for gpu in gpu_metrics:
                print(f"GPU {gpu.gpu_id}: {gpu.name}")
                print(f"  Memory:    {gpu.memory_used_mb:8.0f}MB / {gpu.memory_total_mb:.0f}MB ({gpu.memory_utilization:5.1f}%)")
                print(f"  GPU Usage: {gpu.gpu_utilization:8.1f}%")
                print(f"  Temperature: {gpu.temperature_c:6.0f}°C")
                print(f"  Power:     {gpu.power_watts:8.1f}W / {gpu.power_limit_watts:.0f}W")
                
                if gpu.processes:
                    print(f"  Processes: {len(gpu.processes)} running")
                    for proc in gpu.processes[:3]:  # Show top 3
                        print(f"    PID {proc['pid']:5d}: {proc['name'][:20]:20s} ({proc['gpu_memory_mb']:6.0f}MB)")
                else:
                    print("  Processes: None")
                print()
            
            # Performance analysis
            total_gpu_memory = sum(gpu.memory_total_mb for gpu in gpu_metrics)
            used_gpu_memory = sum(gpu.memory_used_mb for gpu in gpu_metrics)
            avg_gpu_util = sum(gpu.gpu_utilization for gpu in gpu_metrics) / len(gpu_metrics)
            
            print("📊 PERFORMANCE ANALYSIS")
            print("-" * 40)
            print(f"Total GPU Memory: {used_gpu_memory/1024:.1f}GB / {total_gpu_memory/1024:.1f}GB ({used_gpu_memory/total_gpu_memory*100:.1f}%)")
            print(f"Average GPU Util: {avg_gpu_util:.1f}%")
            
            # Optimization suggestions
            self.print_optimization_suggestions(gpu_metrics, system_metrics)
        
        else:
            print("⚠️ No GPU metrics available")
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def print_optimization_suggestions(self, gpu_metrics: List[GPUMetrics], system_metrics: SystemMetrics):
        """Print optimization suggestions"""
        suggestions = []
        
        # GPU memory analysis
        high_memory_gpus = [gpu for gpu in gpu_metrics if gpu.memory_utilization > 90]
        low_util_gpus = [gpu for gpu in gpu_metrics if gpu.gpu_utilization < 30 and gpu.memory_utilization > 10]
        
        if high_memory_gpus:
            suggestions.append(f"⚠️  {len(high_memory_gpus)} GPU(s) >90% memory - consider smaller batch sizes")
        
        if low_util_gpus:
            suggestions.append(f"💡 {len(low_util_gpus)} GPU(s) underutilized - increase batch size or use tensor parallelism")
        
        # System memory
        if system_metrics.memory_utilization > 85:
            suggestions.append("⚠️  System RAM >85% - may cause swapping and slower performance")
        
        # Temperature warnings
        hot_gpus = [gpu for gpu in gpu_metrics if gpu.temperature_c > 80]
        if hot_gpus:
            suggestions.append(f"🔥 {len(hot_gpus)} GPU(s) >80°C - check cooling and reduce power if needed")
        
        # Power analysis
        high_power_gpus = [gpu for gpu in gpu_metrics if gpu.power_watts > gpu.power_limit_watts * 0.9]
        if high_power_gpus:
            suggestions.append(f"⚡ {len(high_power_gpus)} GPU(s) near power limit - may throttle performance")
        
        if suggestions:
            print()
            print("🔧 OPTIMIZATION SUGGESTIONS")
            print("-" * 40)
            for suggestion in suggestions:
                print(f"  {suggestion}")
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("📊 GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("📊 GPU monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                gpu_metrics = self.get_gpu_metrics()
                system_metrics = self.get_system_metrics()
                
                # Store metrics
                if not self.metrics_queue.full():
                    self.metrics_queue.put({
                        'gpu_metrics': [asdict(gpu) for gpu in gpu_metrics],
                        'system_metrics': asdict(system_metrics),
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def save_metrics(self, output_file: str):
        """Save collected metrics to file"""
        metrics_data = []
        
        while not self.metrics_queue.empty():
            try:
                metrics_data.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        
        if metrics_data:
            with open(output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"📊 Saved {len(metrics_data)} metric snapshots to {output_file}")
        
        return len(metrics_data)
    
    def run_interactive_monitoring(self):
        """Run interactive console monitoring"""
        try:
            while True:
                gpu_metrics = self.get_gpu_metrics()
                system_metrics = self.get_system_metrics()
                self.print_status(gpu_metrics, system_metrics)
                time.sleep(self.monitoring_interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in interactive monitoring: {e}")

def main():
    """Main function for standalone monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Monitoring for Server Models")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--output", type=str, 
                       help="Save metrics to file instead of interactive display")
    parser.add_argument("--duration", type=int, default=0,
                       help="Monitoring duration in seconds (0 for continuous)")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(monitoring_interval=args.interval)
    
    if args.output:
        # Background monitoring mode
        logger.info(f"📊 Starting background monitoring (interval: {args.interval}s)")
        monitor.start_monitoring()
        
        try:
            if args.duration > 0:
                time.sleep(args.duration)
            else:
                print("Press Ctrl+C to stop monitoring and save results...")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        monitor.stop_monitoring()
        count = monitor.save_metrics(args.output)
        print(f"📊 Saved {count} metric snapshots to {args.output}")
    
    else:
        # Interactive monitoring mode
        logger.info("📊 Starting interactive monitoring")
        monitor.run_interactive_monitoring()

if __name__ == "__main__":
    main()