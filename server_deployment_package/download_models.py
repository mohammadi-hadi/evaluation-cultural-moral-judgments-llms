#!/usr/bin/env python3
"""
Download HuggingFace Models for Server Deployment
Downloads models to /data/storage_4_tb/moral-alignment-pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import subprocess
import torch
from huggingface_hub import snapshot_download, hf_hub_download, login, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model to download"""
    name: str
    hf_path: str
    size_gb: float
    priority: str
    quantization: Optional[str] = None
    notes: str = ""

class ModelDownloader:
    """Download and prepare models for server deployment"""
    
    # Models to download - organized by size and priority
    MODELS = {
        # Small models (< 10GB) - High throughput on single GPU
        "small": [
            # Baseline models
            ModelConfig(
                name="gpt2",
                hf_path="openai-community/gpt2",
                size_gb=0.5,
                priority="CRITICAL",
                notes="Essential baseline model"
            ),
            
            # Llama family - Small
            ModelConfig(
                name="llama3.2:1b",
                hf_path="meta-llama/Llama-3.2-1B-Instruct",
                size_gb=2,
                priority="HIGH",
                notes="Ultra-fast small Llama"
            ),
            ModelConfig(
                name="llama3.2:3b", 
                hf_path="meta-llama/Llama-3.2-3B-Instruct",
                size_gb=6,
                priority="HIGH",
                notes="Efficient small Llama"
            ),
            ModelConfig(
                name="llama3.1:8b",
                hf_path="meta-llama/Llama-3.1-8B-Instruct",
                size_gb=16,
                priority="CRITICAL",
                notes="Latest Llama 8B with 128K context"
            ),
            ModelConfig(
                name="llama3:8b",
                hf_path="meta-llama/Meta-Llama-3-8B-Instruct",
                size_gb=16,
                priority="HIGH",
                notes="Original Llama 3 8B"
            ),
            
            # Mistral family
            ModelConfig(
                name="mistral:7b",
                hf_path="mistralai/Mistral-7B-Instruct-v0.3",
                size_gb=14,
                priority="CRITICAL",
                notes="Excellent general purpose model"
            ),
            
            # Qwen family - Small
            ModelConfig(
                name="qwen2.5:7b",
                hf_path="Qwen/Qwen2.5-7B-Instruct",
                size_gb=14,
                priority="CRITICAL",
                notes="High-performance Qwen 7B"
            ),
            ModelConfig(
                name="qwen3:8b",
                hf_path="Qwen/Qwen3-8B-Instruct",
                size_gb=16,
                priority="HIGH", 
                notes="Latest Qwen 3 generation 8B"
            ),
            
            # Gemma family - Small
            ModelConfig(
                name="gemma:7b",
                hf_path="google/gemma-7b-it",
                size_gb=14,
                priority="HIGH",
                notes="Google's instruction-tuned 7B"
            ),
            ModelConfig(
                name="gemma2:9b",
                hf_path="google/gemma-2-9b-it", 
                size_gb=18,
                priority="HIGH",
                notes="Improved Gemma 2 9B"
            ),
            ModelConfig(
                name="gemma3:4b",
                hf_path="google/gemma-3-4b-it",
                size_gb=8,
                priority="HIGH",
                notes="Latest Gemma 3 4B"
            ),
            
            # Phi family
            ModelConfig(
                name="phi3:3.8b", 
                hf_path="microsoft/Phi-3-mini-4k-instruct",
                size_gb=8,
                priority="HIGH",
                notes="Microsoft's efficient mini model"
            ),
            ModelConfig(
                name="phi-3.5-mini",
                hf_path="microsoft/Phi-3.5-mini-instruct",
                size_gb=8,
                priority="HIGH",
                notes="Latest Phi 3.5 mini"
            ),
            
            # DeepSeek family
            ModelConfig(
                name="deepseek-r1:8b",
                hf_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-8B",
                size_gb=16,
                priority="HIGH",
                notes="DeepSeek reasoning distilled model"
            ),
            
            # Vision models
            ModelConfig(
                name="llava:7b",
                hf_path="llava-hf/llava-1.5-7b-hf",
                size_gb=14,
                priority="MEDIUM",
                notes="Vision-language model 7B"
            )
        ],
        
        # Medium models (7-14GB)
        "medium": [
            ModelConfig(
                name="mistral-7b",
                hf_path="mistralai/Mistral-7B-Instruct-v0.3",
                size_gb=14.0,
                priority="CRITICAL",
                notes="Excellent general purpose"
            ),
            ModelConfig(
                name="llama3.1-8b",
                hf_path="meta-llama/Llama-3.1-8B-Instruct",
                size_gb=16.0,
                priority="CRITICAL",
                notes="Well-balanced, 128K context"
            ),
            ModelConfig(
                name="qwen2.5-7b",
                hf_path="Qwen/Qwen2.5-7B-Instruct",
                size_gb=14.0,
                priority="HIGH",
                notes="Strong multilingual"
            ),
            ModelConfig(
                name="qwen2.5-14b",
                hf_path="Qwen/Qwen2.5-14B-Instruct",
                size_gb=28.0,
                priority="MEDIUM"
            ),
            ModelConfig(
                name="gemma-9b",
                hf_path="google/gemma-2-9b-it",
                size_gb=18.0,
                priority="HIGH"
            ),
            ModelConfig(
                name="phi-4",
                hf_path="microsoft/Phi-4",
                size_gb=28.0,
                priority="MEDIUM",
                notes="Latest Phi, SOTA for size"
            ),
        ],
        
        # Large models (27-72GB)
        "large": [
            ModelConfig(
                name="qwen2.5-32b",
                hf_path="Qwen/Qwen2.5-32B-Instruct",
                size_gb=64.0,
                priority="HIGH",
                notes="Good balance size/performance"
            ),
            ModelConfig(
                name="qwq-32b",
                hf_path="Qwen/QwQ-32B-Preview",
                size_gb=64.0,
                priority="HIGH",
                notes="Reasoning specialist"
            ),
            ModelConfig(
                name="gemma-27b",
                hf_path="google/gemma-2-27b-it",
                size_gb=54.0,
                priority="MEDIUM"
            ),
            ModelConfig(
                name="llama3.3-70b",
                hf_path="meta-llama/Llama-3.3-70B-Instruct",
                size_gb=140.0,
                priority="CRITICAL",
                notes="Best open 70B",
                quantization="8bit"  # Use quantization for 70B
            ),
            ModelConfig(
                name="qwen2.5-72b",
                hf_path="Qwen/Qwen2.5-72B-Instruct",
                size_gb=144.0,
                priority="CRITICAL",
                notes="Excellent cross-cultural",
                quantization="8bit"
            ),
        ],
        
        # Extra large models (100GB+) - for future with cluster
        "xlarge": [
            ModelConfig(
                name="gpt-oss-120b",
                hf_path="openai/gpt-oss-120b",  # Hypothetical path
                size_gb=240.0,
                priority="HIGH",
                notes="Matches o4-mini",
                quantization="4bit"
            ),
            ModelConfig(
                name="mixtral-8x22b",
                hf_path="mistralai/Mixtral-8x22B-Instruct-v0.1",
                size_gb=280.0,
                priority="LOW",
                notes="Large MoE",
                quantization="4bit"
            ),
            ModelConfig(
                name="qwen3-235b",
                hf_path="Qwen/Qwen3-235B",  # Future model
                size_gb=470.0,
                priority="LOW",
                notes="Massive model",
                quantization="4bit"
            ),
        ]
    }
    
    def __init__(self, 
                 base_dir: str = "/data/storage_4_tb/moral-alignment-pipeline",
                 use_symlinks: bool = True):
        """Initialize model downloader
        
        Args:
            base_dir: Base directory for model storage
            use_symlinks: Whether to use symlinks to save space
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_symlinks = use_symlinks
        
        # Track downloaded models
        self.status_file = self.base_dir / "download_status.json"
        self.status = self._load_status()
        
        # Set up HuggingFace authentication
        self._setup_hf_authentication()
        
        logger.info(f"ModelDownloader initialized")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Using symlinks: {self.use_symlinks}")
    
    def _setup_hf_authentication(self):
        """Set up HuggingFace authentication for gated models"""
        # Try environment variables first
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=True)
                logger.info("✅ Authenticated with HuggingFace using token")
                return
            except Exception as e:
                logger.warning(f"Failed to authenticate with provided token: {e}")
        
        # Try to use cached credentials
        try:
            token = HfFolder.get_token()
            if token:
                logger.info("✅ Using cached HuggingFace credentials")
                return
        except Exception:
            pass
        
        # No authentication found
        logger.warning("⚠️  No HuggingFace authentication found")
        logger.warning("   Some models require authentication. To fix:")
        logger.warning("   1. Get token from https://huggingface.co/settings/tokens")
        logger.warning("   2. Set HF_TOKEN environment variable")
        logger.warning("   3. Or run: huggingface-cli login")
        
        # Add list of open models that don't require authentication
        self._log_open_alternatives()
    
    def _log_open_alternatives(self):
        """Log available open models that don't require authentication"""
        open_models = [
            ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "14GB"),
            ("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", "28GB"), 
            ("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "64GB"),
            ("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct", "140GB"),
            ("qwq-32b", "Qwen/QwQ-32B-Preview", "64GB"),
            ("phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "8GB"),
            ("falcon-7b", "tiiuae/falcon-7b-instruct", "14GB"),
            ("mpt-7b", "mosaicml/mpt-7b-instruct", "14GB"),
            ("stablelm-3b", "stabilityai/stablelm-3b-4e1t", "6GB")
        ]
        
        logger.info("📂 Available open models (no authentication required):")
        for name, path, size in open_models:
            logger.info(f"   ✓ {name} ({size}) - {path}")
    
    def _load_status(self) -> Dict:
        """Load download status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"downloaded": [], "failed": [], "in_progress": []}
    
    def _save_status(self):
        """Save download status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def check_disk_space(self) -> float:
        """Check available disk space in GB"""
        import shutil
        stat = shutil.disk_usage(self.base_dir)
        return stat.free / (1024 ** 3)
    
    def download_model(self, model_config: ModelConfig) -> bool:
        """Download a single model
        
        Args:
            model_config: Configuration for the model to download
            
        Returns:
            True if successful, False otherwise
        """
        model_name = model_config.name
        
        # Check if already downloaded
        if model_name in self.status["downloaded"]:
            logger.info(f"Model {model_name} already downloaded")
            return True
        
        # Check disk space
        free_space = self.check_disk_space()
        if free_space < model_config.size_gb * 1.5:  # Need extra space
            logger.error(f"Insufficient disk space for {model_name}: {free_space:.1f}GB available, need {model_config.size_gb * 1.5:.1f}GB")
            return False
        
        # Mark as in progress
        if model_name not in self.status["in_progress"]:
            self.status["in_progress"].append(model_name)
            self._save_status()
        
        logger.info(f"Downloading {model_name} from {model_config.hf_path}")
        logger.info(f"  Size: {model_config.size_gb}GB")
        logger.info(f"  Priority: {model_config.priority}")
        
        model_dir = self.models_dir / model_name
        
        try:
            # Download model using huggingface-hub
            snapshot_download(
                repo_id=model_config.hf_path,
                local_dir=model_dir,
                local_dir_use_symlinks=self.use_symlinks,
                resume_download=True,
                max_workers=4
            )
            
            # Verify download
            if not self._verify_download(model_dir):
                raise Exception("Download verification failed")
            
            # Mark as complete
            self.status["in_progress"].remove(model_name)
            self.status["downloaded"].append(model_name)
            self._save_status()
            
            logger.info(f"✅ Successfully downloaded {model_name}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's an authentication error
            if "401" in error_msg or "Access to model" in error_msg or "restricted" in error_msg:
                logger.error(f"❌ {model_name}: Authentication required")
                logger.error(f"   This model requires access approval and authentication")
                logger.error(f"   1. Visit: https://huggingface.co/{model_config.hf_path}")
                logger.error(f"   2. Click 'Accept License' if prompted")
                logger.error(f"   3. Set HF_TOKEN environment variable")
                logger.error(f"   4. Or run: huggingface-cli login")
            else:
                logger.error(f"❌ Failed to download {model_name}: {e}")
            
            if model_name in self.status["in_progress"]:
                self.status["in_progress"].remove(model_name)
            if model_name not in self.status["failed"]:
                self.status["failed"].append(model_name)
            self._save_status()
            
            return False
    
    def _verify_download(self, model_dir: Path) -> bool:
        """Verify that model was downloaded correctly"""
        # Check for essential files
        essential_files = ["config.json", "tokenizer_config.json"]
        
        for file in essential_files:
            if not (model_dir / file).exists():
                logger.warning(f"Missing {file} in {model_dir}")
                return False
        
        # Check for model weights
        has_weights = False
        for pattern in ["*.bin", "*.safetensors", "*.pt", "*.pth"]:
            if list(model_dir.glob(pattern)):
                has_weights = True
                break
        
        if not has_weights:
            logger.warning(f"No model weights found in {model_dir}")
            return False
        
        return True
    
    def download_category(self, category: str) -> Dict[str, int]:
        """Download all models in a category
        
        Args:
            category: Category name (small, medium, large, xlarge)
            
        Returns:
            Dictionary with success/failure counts
        """
        if category not in self.MODELS:
            logger.error(f"Unknown category: {category}")
            return {"success": 0, "failed": 0}
        
        models = self.MODELS[category]
        logger.info(f"Downloading {len(models)} models from category: {category}")
        
        results = {"success": 0, "failed": 0}
        
        for model_config in models:
            if self.download_model(model_config):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def download_priority_models(self, min_priority: str = "HIGH") -> Dict[str, int]:
        """Download models based on priority
        
        Args:
            min_priority: Minimum priority level (CRITICAL, HIGH, MEDIUM, LOW)
            
        Returns:
            Dictionary with success/failure counts
        """
        priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        min_idx = priority_order.index(min_priority)
        valid_priorities = priority_order[:min_idx + 1]
        
        results = {"success": 0, "failed": 0}
        
        for category, models in self.MODELS.items():
            for model_config in models:
                if model_config.priority in valid_priorities:
                    if self.download_model(model_config):
                        results["success"] += 1
                    else:
                        results["failed"] += 1
        
        return results
    
    def get_status_report(self) -> str:
        """Generate a status report of downloads"""
        total_models = sum(len(models) for models in self.MODELS.values())
        
        report = [
            "=" * 60,
            "MODEL DOWNLOAD STATUS REPORT",
            "=" * 60,
            f"Total models: {total_models}",
            f"Downloaded: {len(self.status['downloaded'])}",
            f"In progress: {len(self.status['in_progress'])}",
            f"Failed: {len(self.status['failed'])}",
            f"Remaining: {total_models - len(self.status['downloaded']) - len(self.status['failed'])}",
            "",
            f"Available disk space: {self.check_disk_space():.1f}GB",
            "",
        ]
        
        if self.status['downloaded']:
            report.append("DOWNLOADED MODELS:")
            for model in self.status['downloaded']:
                report.append(f"  ✓ {model}")
            report.append("")
        
        if self.status['in_progress']:
            report.append("IN PROGRESS:")
            for model in self.status['in_progress']:
                report.append(f"  ⟳ {model}")
            report.append("")
        
        if self.status['failed']:
            report.append("FAILED:")
            for model in self.status['failed']:
                report.append(f"  ✗ {model}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function for downloading models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for server deployment")
    parser.add_argument("--base-dir", default="/data/storage_4_tb/moral-alignment-pipeline",
                       help="Base directory for model storage")
    parser.add_argument("--category", choices=["small", "medium", "large", "xlarge", "all"],
                       help="Category of models to download")
    parser.add_argument("--priority", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                       default="HIGH", help="Minimum priority level to download")
    parser.add_argument("--status", action="store_true", help="Show status report only")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(base_dir=args.base_dir)
    
    if args.status:
        print(downloader.get_status_report())
        return
    
    if args.category:
        if args.category == "all":
            for cat in ["small", "medium", "large"]:
                results = downloader.download_category(cat)
                print(f"\nCategory {cat}: {results['success']} success, {results['failed']} failed")
        else:
            results = downloader.download_category(args.category)
            print(f"\nCategory {args.category}: {results['success']} success, {results['failed']} failed")
    else:
        results = downloader.download_priority_models(args.priority)
        print(f"\nPriority {args.priority}+: {results['success']} success, {results['failed']} failed")
    
    print("\n" + downloader.get_status_report())


if __name__ == "__main__":
    main()