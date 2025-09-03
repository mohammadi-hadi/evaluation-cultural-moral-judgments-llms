#!/usr/bin/env python3
"""
Server GPU Diagnostics - Check what's using the GPUs
Investigate GPU usage when monitor shows no evaluation processes
"""

import subprocess
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and return output"""
    try:
        logger.info(f"🔍 {description}")
        logger.info(f"   Command: {command}")
        
        result = subprocess.run(
            command.split(), 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            return result.stdout.strip()
        else:
            logger.warning(f"⚠️ {description} - ERROR")
            logger.warning(f"   Error: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ {description} - EXCEPTION: {e}")
        return None

def main():
    """Run comprehensive server diagnostics"""
    logger.info("🚀 SERVER GPU DIAGNOSTICS")
    logger.info("=" * 70)
    logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Check GPU status
    logger.info("🖥️  GPU STATUS CHECK")
    logger.info("-" * 50)
    
    # nvidia-smi basic check
    nvidia_output = run_command("nvidia-smi", "Basic GPU status")
    if nvidia_output:
        print("\n" + nvidia_output)
    
    # Check what processes are using GPUs
    logger.info("\n🔍 GPU PROCESS ANALYSIS")
    logger.info("-" * 50)
    
    gpu_processes = run_command("nvidia-smi pmon -c 1", "GPU process monitoring")
    if gpu_processes:
        print(gpu_processes)
    
    # Check running Python processes
    logger.info("\n🐍 PYTHON PROCESS CHECK")
    logger.info("-" * 50)
    
    python_procs = run_command("pgrep -af python", "All Python processes")
    if python_procs:
        for line in python_procs.split('\n'):
            if line.strip():
                print(f"   {line}")
    
    # Check for evaluation-related processes
    logger.info("\n📊 EVALUATION PROCESS CHECK")
    logger.info("-" * 50)
    
    eval_procs = run_command("pgrep -af 'run_server_evaluation|server_model_runner'", 
                            "Evaluation-specific processes")
    if eval_procs:
        print(eval_procs)
    else:
        logger.warning("⚠️  No evaluation processes found")
    
    # Check for result files
    logger.info("\n📁 RESULT FILE CHECK")
    logger.info("-" * 50)
    
    result_files = run_command("find /data/storage_4_tb/moral-alignment-pipeline/outputs -name '*.json' -mmin -30", 
                              "Recent result files (last 30 min)")
    if result_files:
        print("Recent result files:")
        for file in result_files.split('\n'):
            if file.strip():
                # Get file info
                file_info = run_command(f"ls -lah '{file}'", f"File info for {file}")
                print(f"   {file}")
                if file_info:
                    print(f"     {file_info}")
    else:
        logger.warning("⚠️  No recent result files found")
    
    # Check GPU memory details
    logger.info("\n💾 GPU MEMORY ANALYSIS")
    logger.info("-" * 50)
    
    memory_info = run_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv", 
                             "Detailed GPU memory usage")
    if memory_info:
        print(memory_info)
    
    # Check VLLM or model server processes
    logger.info("\n🚀 MODEL SERVER CHECK")
    logger.info("-" * 50)
    
    vllm_procs = run_command("pgrep -af 'vllm|transformers|torch'", "Model server processes")
    if vllm_procs:
        print("Model-related processes:")
        for line in vllm_procs.split('\n'):
            if line.strip():
                print(f"   {line}")
    else:
        logger.info("ℹ️  No VLLM or model server processes detected")
    
    # Network checks (in case there's a server running)
    logger.info("\n🌐 NETWORK SERVICE CHECK")
    logger.info("-" * 50)
    
    network_check = run_command("netstat -tlnp | grep -E ':(8000|8080|11434|5000)'", 
                                "Check for model servers on common ports")
    if network_check:
        print("Active network services:")
        print(network_check)
    else:
        logger.info("ℹ️  No model servers detected on common ports")
    
    logger.info("\n🎯 DIAGNOSTIC COMPLETE")
    logger.info("=" * 70)
    
    # Summary recommendations
    logger.info("\n📋 RECOMMENDATIONS:")
    if not eval_procs:
        logger.info("   • No evaluation processes detected - evaluation may not be running")
        logger.info("   • Check if evaluation was started: cd /data/storage_4_tb/moral-alignment-pipeline")
        logger.info("   • Try: python run_server_evaluation.py --list")
        logger.info("   • Start evaluation: python run_server_evaluation.py")
    
    if gpu_processes and "No running processes found" not in str(gpu_processes):
        logger.info("   • GPUs are being used by some processes")
        logger.info("   • Check the process list above to identify what's using the GPUs")
    
    logger.info("   • Use server_live_monitor.py --once to check evaluation status")
    logger.info("   • Use nvidia-smi -l 5 to continuously monitor GPU usage")

if __name__ == "__main__":
    main()