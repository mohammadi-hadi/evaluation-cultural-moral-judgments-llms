#!/usr/bin/env python3
"""
Comprehensive Data Storage Manager for Moral Alignment Pipeline
Handles all data persistence, retrieval, and organization
"""

import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import hashlib
import gzip
import shutil
import yaml

@dataclass
class ExperimentRun:
    """Represents a single experiment run"""
    run_id: str
    timestamp: str
    models: List[str]
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str  # 'running', 'completed', 'failed'
    output_paths: Dict[str, str] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

@dataclass
class ModelResult:
    """Stores results for a single model evaluation"""
    model_name: str
    country: str
    topic: str
    method: str  # 'logprob', 'direct', 'cot'
    score: float
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataStorageManager:
    """Manages all data storage operations for the pipeline"""
    
    def __init__(self, base_dir: Path = Path("outputs"), 
                 compress: bool = True):
        """
        Initialize data storage manager
        
        Args:
            base_dir: Base directory for all outputs
            compress: Whether to compress large files
        """
        self.base_dir = base_dir
        self.compress = compress
        
        # Create directory structure
        self._init_directory_structure()
        
        # Initialize database
        self.db_path = self.base_dir / "pipeline.db"
        self._init_database()
        
        # Track current experiment
        self.current_run_id = None
        
    def _init_directory_structure(self):
        """Create organized directory structure"""
        directories = [
            "scores/logprob",
            "scores/direct",
            "scores/cot",
            "traces/raw",
            "traces/processed",
            "prompts/templates",
            "prompts/filled",
            "evaluations/peer",
            "evaluations/human",
            "visualizations/figures",
            "visualizations/reports",
            "models/checkpoints",
            "models/configs",
            "experiments/runs",
            "experiments/logs",
            "cache"
        ]
        
        for dir_path in directories:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiment runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                models TEXT,
                config TEXT,
                metrics TEXT,
                status TEXT,
                output_paths TEXT,
                error_log TEXT
            )
        """)
        
        # Model results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                model_name TEXT,
                country TEXT,
                topic TEXT,
                method TEXT,
                score REAL,
                reasoning TEXT,
                confidence REAL,
                tokens_used INTEGER,
                latency_ms REAL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)
        
        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                run_id TEXT,
                prompt_type TEXT,
                template TEXT,
                filled_prompt TEXT,
                model TEXT,
                country TEXT,
                topic TEXT,
                parameters TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)
        
        # Cross-evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                evaluator_model TEXT,
                evaluated_model TEXT,
                country TEXT,
                topic TEXT,
                original_score REAL,
                evaluation_score REAL,
                agreement_level TEXT,
                confidence REAL,
                reasoning TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_results_run ON model_results(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_results_model ON model_results(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompts_run ON prompts(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cross_eval_run ON cross_evaluations(run_id)")
        
        conn.commit()
        conn.close()
    
    def start_experiment_run(self, models: List[str], config: Dict[str, Any]) -> str:
        """Start a new experiment run"""
        run_id = self._generate_run_id()
        self.current_run_id = run_id
        
        experiment = ExperimentRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            models=models,
            config=config,
            metrics={},
            status='running'
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiment_runs 
            (run_id, timestamp, models, config, metrics, status, output_paths, error_log)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            experiment.timestamp,
            json.dumps(models),
            json.dumps(config),
            json.dumps(experiment.metrics),
            experiment.status,
            json.dumps(experiment.output_paths),
            json.dumps(experiment.error_log)
        ))
        
        conn.commit()
        conn.close()
        
        # Create run directory
        run_dir = self.base_dir / "experiments" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config file
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return run_id
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_hash}"
    
    def save_model_result(self, result: ModelResult, run_id: Optional[str] = None):
        """Save a model evaluation result"""
        if run_id is None:
            run_id = self.current_run_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_results
            (run_id, model_name, country, topic, method, score, reasoning, 
             confidence, tokens_used, latency_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            result.model_name,
            result.country,
            result.topic,
            result.method,
            result.score,
            result.reasoning,
            result.confidence,
            result.tokens_used,
            result.latency_ms,
            json.dumps(result.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_scores_dataframe(self, df: pd.DataFrame, model_name: str, 
                            method: str, run_id: Optional[str] = None):
        """Save scores dataframe for a model"""
        if run_id is None:
            run_id = self.current_run_id
        
        # Save to CSV
        file_path = self.base_dir / "scores" / method / f"{run_id}_{model_name}_{method}_scores.csv"
        df.to_csv(file_path, index=False)
        
        # Compress if needed
        if self.compress and len(df) > 1000:
            self._compress_file(file_path)
        
        # Update experiment output paths
        self._update_output_path(run_id, f"{model_name}_{method}_scores", str(file_path))
        
        return file_path
    
    def save_traces(self, traces: List[Dict], model_name: str, 
                   run_id: Optional[str] = None):
        """Save reasoning traces"""
        if run_id is None:
            run_id = self.current_run_id
        
        # Save raw traces
        raw_path = self.base_dir / "traces" / "raw" / f"{run_id}_{model_name}_traces.jsonl"
        
        with open(raw_path, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + '\n')
        
        # Compress if needed
        if self.compress and len(traces) > 100:
            self._compress_file(raw_path)
        
        # Process and save structured traces
        processed_traces = self._process_traces(traces)
        processed_path = self.base_dir / "traces" / "processed" / f"{run_id}_{model_name}_processed.json"
        
        with open(processed_path, 'w') as f:
            json.dump(processed_traces, f, indent=2)
        
        # Update output paths
        self._update_output_path(run_id, f"{model_name}_traces_raw", str(raw_path))
        self._update_output_path(run_id, f"{model_name}_traces_processed", str(processed_path))
        
        return raw_path, processed_path
    
    def _process_traces(self, traces: List[Dict]) -> Dict:
        """Process raw traces into structured format"""
        processed = {
            'total_traces': len(traces),
            'by_country': {},
            'by_topic': {},
            'score_distribution': [],
            'reasoning_patterns': []
        }
        
        for trace in traces:
            country = trace.get('country', 'unknown')
            topic = trace.get('topic', 'unknown')
            score = trace.get('score', 0)
            
            # Group by country
            if country not in processed['by_country']:
                processed['by_country'][country] = []
            processed['by_country'][country].append(score)
            
            # Group by topic
            if topic not in processed['by_topic']:
                processed['by_topic'][topic] = []
            processed['by_topic'][topic].append(score)
            
            # Score distribution
            processed['score_distribution'].append(score)
        
        # Calculate statistics
        for country in processed['by_country']:
            scores = processed['by_country'][country]
            processed['by_country'][country] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        for topic in processed['by_topic']:
            scores = processed['by_topic'][topic]
            processed['by_topic'][topic] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return processed
    
    def save_prompt(self, prompt_type: str, template: str, filled: str,
                   model: str, country: str, topic: str, 
                   parameters: Dict, run_id: Optional[str] = None):
        """Save a prompt to database"""
        if run_id is None:
            run_id = self.current_run_id
        
        prompt_id = hashlib.md5(f"{model}_{country}_{topic}_{prompt_type}".encode()).hexdigest()[:12]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO prompts
            (prompt_id, run_id, prompt_type, template, filled_prompt, 
             model, country, topic, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_id,
            run_id,
            prompt_type,
            template,
            filled,
            model,
            country,
            topic,
            json.dumps(parameters)
        ))
        
        conn.commit()
        conn.close()
    
    def save_cross_evaluation(self, evaluation: Dict, run_id: Optional[str] = None):
        """Save cross-evaluation result"""
        if run_id is None:
            run_id = self.current_run_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cross_evaluations
            (run_id, evaluator_model, evaluated_model, country, topic,
             original_score, evaluation_score, agreement_level, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            evaluation['evaluator_model'],
            evaluation['evaluated_model'],
            evaluation['country'],
            evaluation['topic'],
            evaluation['original_score'],
            evaluation['evaluation_score'],
            evaluation['agreement_level'],
            evaluation['confidence'],
            evaluation['reasoning']
        ))
        
        conn.commit()
        conn.close()
    
    def save_visualization(self, figure, name: str, run_id: Optional[str] = None):
        """Save a visualization figure"""
        if run_id is None:
            run_id = self.current_run_id
        
        fig_path = self.base_dir / "visualizations" / "figures" / f"{run_id}_{name}.png"
        
        # Save figure based on type
        if hasattr(figure, 'savefig'):  # Matplotlib
            figure.savefig(fig_path, dpi=300, bbox_inches='tight')
        elif hasattr(figure, 'write_image'):  # Plotly
            figure.write_image(fig_path)
        else:
            # Assume it's already an image
            with open(fig_path, 'wb') as f:
                f.write(figure)
        
        # Update output paths
        self._update_output_path(run_id, f"viz_{name}", str(fig_path))
        
        return fig_path
    
    def save_metrics(self, metrics: Dict, run_id: Optional[str] = None):
        """Save experiment metrics"""
        if run_id is None:
            run_id = self.current_run_id
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE experiment_runs
            SET metrics = ?
            WHERE run_id = ?
        """, (json.dumps(metrics), run_id))
        
        conn.commit()
        conn.close()
        
        # Save to file
        metrics_path = self.base_dir / "experiments" / "runs" / run_id / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics_path
    
    def complete_experiment_run(self, run_id: Optional[str] = None, 
                               status: str = 'completed'):
        """Mark experiment run as complete"""
        if run_id is None:
            run_id = self.current_run_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE experiment_runs
            SET status = ?
            WHERE run_id = ?
        """, (status, run_id))
        
        conn.commit()
        conn.close()
        
        # Generate summary report
        self._generate_run_summary(run_id)
    
    def _generate_run_summary(self, run_id: str):
        """Generate summary report for an experiment run"""
        conn = sqlite3.connect(self.db_path)
        
        # Get experiment info
        exp_df = pd.read_sql_query(
            "SELECT * FROM experiment_runs WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        # Get results summary
        results_df = pd.read_sql_query(
            "SELECT * FROM model_results WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        conn.close()
        
        if exp_df.empty:
            return
        
        exp = exp_df.iloc[0]
        
        summary = {
            'run_id': run_id,
            'timestamp': exp['timestamp'],
            'status': exp['status'],
            'models': json.loads(exp['models']),
            'total_evaluations': len(results_df),
            'unique_countries': results_df['country'].nunique() if not results_df.empty else 0,
            'unique_topics': results_df['topic'].nunique() if not results_df.empty else 0,
            'methods_used': results_df['method'].unique().tolist() if not results_df.empty else [],
            'metrics': json.loads(exp['metrics']) if exp['metrics'] else {},
            'output_paths': json.loads(exp['output_paths']) if exp['output_paths'] else {}
        }
        
        # Calculate model-wise statistics
        if not results_df.empty:
            model_stats = results_df.groupby('model_name').agg({
                'score': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'tokens_used': 'sum',
                'latency_ms': 'mean'
            }).round(3)
            
            summary['model_statistics'] = model_stats.to_dict()
        
        # Save summary
        summary_path = self.base_dir / "experiments" / "runs" / run_id / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def load_experiment_run(self, run_id: str) -> Dict:
        """Load all data for an experiment run"""
        conn = sqlite3.connect(self.db_path)
        
        # Load experiment info
        exp_df = pd.read_sql_query(
            "SELECT * FROM experiment_runs WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        # Load results
        results_df = pd.read_sql_query(
            "SELECT * FROM model_results WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        # Load prompts
        prompts_df = pd.read_sql_query(
            "SELECT * FROM prompts WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        # Load cross-evaluations
        cross_eval_df = pd.read_sql_query(
            "SELECT * FROM cross_evaluations WHERE run_id = ?",
            conn, params=(run_id,)
        )
        
        conn.close()
        
        return {
            'experiment': exp_df.to_dict('records')[0] if not exp_df.empty else {},
            'results': results_df,
            'prompts': prompts_df,
            'cross_evaluations': cross_eval_df
        }
    
    def get_all_experiments(self) -> pd.DataFrame:
        """Get list of all experiment runs"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM experiment_runs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    
    def cache_result(self, key: str, data: Any, ttl_hours: int = 24):
        """Cache intermediate results"""
        cache_path = self.base_dir / "cache" / f"{key}.pkl"
        
        cache_data = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'ttl_hours': ttl_hours
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        if self.compress:
            self._compress_file(cache_path)
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result if still valid"""
        cache_path = self.base_dir / "cache" / f"{key}.pkl"
        
        # Check compressed version
        if not cache_path.exists():
            cache_path = self.base_dir / "cache" / f"{key}.pkl.gz"
            if cache_path.exists():
                cache_path = self._decompress_file(cache_path)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check TTL
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        if age_hours > cache_data['ttl_hours']:
            return None
        
        return cache_data['data']
    
    def _compress_file(self, file_path: Path) -> Path:
        """Compress a file using gzip"""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        file_path.unlink()
        
        return compressed_path
    
    def _decompress_file(self, file_path: Path) -> Path:
        """Decompress a gzipped file"""
        decompressed_path = file_path.with_suffix('')
        
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return decompressed_path
    
    def _update_output_path(self, run_id: str, key: str, path: str):
        """Update output paths for an experiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current paths
        cursor.execute("SELECT output_paths FROM experiment_runs WHERE run_id = ?", (run_id,))
        result = cursor.fetchone()
        
        if result:
            paths = json.loads(result[0]) if result[0] else {}
            paths[key] = path
            
            cursor.execute("""
                UPDATE experiment_runs
                SET output_paths = ?
                WHERE run_id = ?
            """, (json.dumps(paths), run_id))
            
            conn.commit()
        
        conn.close()
    
    def cleanup_old_runs(self, days: int = 30):
        """Clean up old experiment runs"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get old runs
        cursor.execute("""
            SELECT run_id, output_paths 
            FROM experiment_runs 
            WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        old_runs = cursor.fetchall()
        
        for run_id, paths_json in old_runs:
            # Delete files
            if paths_json:
                paths = json.loads(paths_json)
                for path in paths.values():
                    file_path = Path(path)
                    if file_path.exists():
                        file_path.unlink()
            
            # Delete run directory
            run_dir = self.base_dir / "experiments" / "runs" / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            
            # Delete from database
            cursor.execute("DELETE FROM model_results WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM prompts WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM cross_evaluations WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM experiment_runs WHERE run_id = ?", (run_id,))
        
        conn.commit()
        conn.close()
        
        return len(old_runs)


# Example usage
if __name__ == "__main__":
    # Initialize storage manager
    storage = DataStorageManager()
    
    # Start an experiment
    run_id = storage.start_experiment_run(
        models=['gpt-4o', 'claude-3.5-sonnet'],
        config={'sample_size': 100, 'methods': ['logprob', 'cot']}
    )
    
    print(f"Started experiment run: {run_id}")
    
    # Example: Save a model result
    result = ModelResult(
        model_name='gpt-4o',
        country='United States',
        topic='abortion',
        method='cot',
        score=0.3,
        reasoning='Complex cultural factors...',
        confidence=0.8,
        tokens_used=150,
        latency_ms=1200
    )
    
    storage.save_model_result(result)
    
    # Complete the run
    storage.complete_experiment_run()
    
    print("Data storage system initialized and tested")