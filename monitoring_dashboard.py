#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Comprehensive Evaluation
Shows progress, errors, and statistics during execution
"""

import os
import sys
import time
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import json
import psutil

# Configure Streamlit
st.set_page_config(
    page_title="Moral Alignment Evaluation Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EvaluationMonitor:
    """Real-time monitoring for evaluation progress"""
    
    def __init__(self, db_path: str = "outputs/comprehensive/results.db"):
        self.db_path = Path(db_path)
        self.refresh_interval = 5  # seconds
        
        # Initialize session state
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now()
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def get_progress_data(self):
        """Get current progress from database"""
        if not self.db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get overall progress
            progress_query = """
                SELECT 
                    model,
                    model_type,
                    COUNT(*) as completed,
                    AVG(score) as avg_score,
                    AVG(inference_time) as avg_time,
                    SUM(cost) as total_cost,
                    MAX(timestamp) as last_update
                FROM model_results
                GROUP BY model, model_type
            """
            progress_df = pd.read_sql_query(progress_query, conn)
            
            # Get recent errors
            errors_query = """
                SELECT 
                    model,
                    COUNT(*) as error_count
                FROM model_results
                WHERE response LIKE '%error%' OR score IS NULL
                GROUP BY model
            """
            errors_df = pd.read_sql_query(errors_query, conn)
            
            # Get checkpoints
            checkpoint_query = """
                SELECT * FROM checkpoints
                ORDER BY last_checkpoint DESC
            """
            checkpoints_df = pd.read_sql_query(checkpoint_query, conn)
            
            conn.close()
            
            return {
                'progress': progress_df,
                'errors': errors_df,
                'checkpoints': checkpoints_df
            }
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.title("🔬 Moral Alignment Evaluation Monitor")
        
        with col2:
            elapsed = datetime.now() - st.session_state.start_time
            st.metric("Elapsed Time", str(elapsed).split('.')[0])
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
    
    def render_system_stats(self):
        """Render system resource statistics"""
        st.sidebar.header("💻 System Resources")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.sidebar.metric("CPU Usage", f"{cpu_percent}%")
        
        # Memory usage
        mem = psutil.virtual_memory()
        mem_used_gb = (mem.total - mem.available) / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_percent = mem.percent
        
        st.sidebar.metric("Memory Usage", 
                         f"{mem_used_gb:.1f}/{mem_total_gb:.1f} GB",
                         f"{mem_percent:.1f}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        st.sidebar.metric("Disk Free", f"{disk_free_gb:.1f} GB")
        
        # Network status
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            st.sidebar.success("✅ Internet Connected")
        except:
            st.sidebar.error("❌ Internet Disconnected")
    
    def render_progress_overview(self, data):
        """Render overall progress overview"""
        if data is None or data['progress'].empty:
            st.info("No evaluation data yet. Waiting for results...")
            return
        
        progress_df = data['progress']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_completed = progress_df['completed'].sum()
            st.metric("Total Completed", f"{total_completed:,}")
        
        with col2:
            api_cost = progress_df[progress_df['model_type'] == 'api']['total_cost'].sum()
            st.metric("API Cost", f"${api_cost:.2f}")
        
        with col3:
            avg_score = progress_df['avg_score'].mean()
            st.metric("Avg Score", f"{avg_score:.3f}")
        
        with col4:
            active_models = len(progress_df)
            st.metric("Active Models", active_models)
    
    def render_model_progress(self, data):
        """Render per-model progress bars"""
        if data is None or data['progress'].empty:
            return
        
        st.header("📊 Model Progress")
        
        progress_df = data['progress']
        errors_df = data['errors'] if 'errors' in data else pd.DataFrame()
        
        # Separate API and local models
        api_models = progress_df[progress_df['model_type'] == 'api']
        local_models = progress_df[progress_df['model_type'] == 'local']
        
        # API Models
        if not api_models.empty:
            st.subheader("☁️ API Models")
            for _, row in api_models.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    # Progress bar
                    progress_pct = min(row['completed'] / 1000 * 100, 100)  # Assume 1000 target
                    st.progress(progress_pct / 100)
                    st.text(f"{row['model']}: {row['completed']} samples")
                
                with col2:
                    st.metric("Avg Score", f"{row['avg_score']:.3f}")
                
                with col3:
                    st.metric("Cost", f"${row['total_cost']:.2f}")
                
                with col4:
                    error_count = errors_df[errors_df['model'] == row['model']]['error_count'].sum()
                    if error_count > 0:
                        st.metric("Errors", error_count, delta_color="inverse")
        
        # Local Models
        if not local_models.empty:
            st.subheader("💻 Local Models")
            for _, row in local_models.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    progress_pct = min(row['completed'] / 1000 * 100, 100)
                    st.progress(progress_pct / 100)
                    st.text(f"{row['model']}: {row['completed']} samples")
                
                with col2:
                    st.metric("Avg Score", f"{row['avg_score']:.3f}")
                
                with col3:
                    st.metric("Avg Time", f"{row['avg_time']:.2f}s")
                
                with col4:
                    error_count = errors_df[errors_df['model'] == row['model']]['error_count'].sum()
                    if error_count > 0:
                        st.metric("Errors", error_count, delta_color="inverse")
    
    def render_live_charts(self, data):
        """Render live updating charts"""
        if data is None or data['progress'].empty:
            return
        
        st.header("📈 Live Statistics")
        
        progress_df = data['progress']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison bar chart
            fig_bar = px.bar(
                progress_df,
                x='model',
                y='completed',
                color='model_type',
                title='Samples Completed by Model',
                labels={'completed': 'Samples', 'model': 'Model'},
                color_discrete_map={'api': '#1f77b4', 'local': '#ff7f0e'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Score distribution
            fig_score = px.box(
                progress_df,
                x='model_type',
                y='avg_score',
                color='model_type',
                title='Score Distribution by Type',
                labels={'avg_score': 'Average Score', 'model_type': 'Type'},
                color_discrete_map={'api': '#1f77b4', 'local': '#ff7f0e'}
            )
            fig_score.update_layout(height=400)
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Time series of completion rate (if we have timestamp data)
        if 'last_update' in progress_df.columns:
            st.subheader("⏱️ Completion Rate Over Time")
            # This would need more detailed time series data
            st.info("Detailed time series data will be available after more samples complete")
    
    def render_error_log(self, data):
        """Render error log"""
        if data is None:
            return
        
        errors_df = data.get('errors', pd.DataFrame())
        
        if not errors_df.empty:
            with st.expander("⚠️ Error Log", expanded=False):
                st.dataframe(errors_df, use_container_width=True)
    
    def render_checkpoints(self, data):
        """Render checkpoint information"""
        if data is None:
            return
        
        checkpoints_df = data.get('checkpoints', pd.DataFrame())
        
        if not checkpoints_df.empty:
            st.sidebar.header("💾 Checkpoints")
            st.sidebar.text("Latest checkpoints:")
            for _, row in checkpoints_df.head(5).iterrows():
                st.sidebar.text(f"• {row['model']}: {row['completed_samples']}")
    
    def run(self):
        """Main dashboard loop"""
        # Render header
        self.render_header()
        
        # Get data
        data = self.get_progress_data()
        
        # Render components
        self.render_system_stats()
        self.render_progress_overview(data)
        self.render_model_progress(data)
        self.render_live_charts(data)
        self.render_error_log(data)
        self.render_checkpoints(data)
        
        # Auto-refresh
        st.sidebar.header("🔄 Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=60,
                value=10
            )
            
            # Add JavaScript for auto-refresh
            st.markdown(
                f"""
                <script>
                    setTimeout(function() {{
                        window.location.reload();
                    }}, {refresh_interval * 1000});
                </script>
                """,
                unsafe_allow_html=True
            )
        
        # Status message
        st.sidebar.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation monitoring dashboard")
    parser.add_argument('--db-path', type=str, 
                       default="outputs/comprehensive/results.db",
                       help='Path to results database')
    
    # For Streamlit, we need to handle args differently
    if 'streamlit' in sys.modules:
        # Running in Streamlit
        monitor = EvaluationMonitor()
        monitor.run()
    else:
        # Command line info
        print("Start the monitoring dashboard with:")
        print("  streamlit run monitoring_dashboard.py")
        print("\nOr specify a custom database:")
        print("  streamlit run monitoring_dashboard.py -- --db-path outputs/my_eval/results.db")


if __name__ == "__main__":
    main()