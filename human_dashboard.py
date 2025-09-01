#!/usr/bin/env python3
"""
Human Evaluation Dashboard for Moral Alignment Pipeline
Interactive Streamlit interface for reviewing model disagreements and providing arbitration
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="Moral Alignment Human Evaluation Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class HumanEvaluationDashboard:
    """Streamlit dashboard for human evaluation of model disagreements"""
    
    def __init__(self, 
                 data_dir: Path = Path("outputs"),
                 db_path: Path = Path("human_evaluations.db")):
        """Initialize dashboard with data connections"""
        self.data_dir = data_dir
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing human evaluations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE,
                evaluator_name TEXT,
                evaluator_email TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                country TEXT,
                topic TEXT,
                model_a TEXT,
                model_b TEXT,
                score_a REAL,
                score_b REAL,
                human_score REAL,
                human_preference TEXT,
                human_reasoning TEXT,
                confidence INTEGER,
                notes TEXT,
                session_id TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                evaluator_name TEXT,
                evaluator_email TEXT,
                start_time DATETIME,
                end_time DATETIME,
                total_evaluations INTEGER,
                avg_confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_disagreement_cases(self) -> pd.DataFrame:
        """Load high-priority disagreement cases"""
        disagreement_file = self.data_dir / "cross_evaluation" / "high_priority_disagreements.json"
        
        if disagreement_file.exists():
            with open(disagreement_file, 'r') as f:
                cases = json.load(f)
            return pd.DataFrame(cases)
        else:
            return pd.DataFrame()
    
    def load_evaluation_pairs(self) -> pd.DataFrame:
        """Load all evaluation pairs"""
        pairs_file = self.data_dir / "cross_evaluation" / "evaluation_pairs.jsonl"
        
        if pairs_file.exists():
            pairs = []
            with open(pairs_file, 'r') as f:
                for line in f:
                    pairs.append(json.loads(line))
            return pd.DataFrame(pairs)
        else:
            return pd.DataFrame()
    
    def save_human_evaluation(self, evaluation_data: Dict):
        """Save human evaluation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate unique case ID
        case_str = f"{evaluation_data['country']}_{evaluation_data['topic']}_{evaluation_data['model_a']}_{evaluation_data['model_b']}"
        case_id = hashlib.md5(case_str.encode()).hexdigest()[:12]
        
        cursor.execute("""
            INSERT OR REPLACE INTO human_evaluations (
                case_id, evaluator_name, evaluator_email, country, topic,
                model_a, model_b, score_a, score_b, human_score,
                human_preference, human_reasoning, confidence, notes, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            evaluation_data['evaluator_name'],
            evaluation_data['evaluator_email'],
            evaluation_data['country'],
            evaluation_data['topic'],
            evaluation_data['model_a'],
            evaluation_data['model_b'],
            evaluation_data['score_a'],
            evaluation_data['score_b'],
            evaluation_data['human_score'],
            evaluation_data['human_preference'],
            evaluation_data['human_reasoning'],
            evaluation_data['confidence'],
            evaluation_data.get('notes', ''),
            evaluation_data.get('session_id', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_human_evaluations(self) -> pd.DataFrame:
        """Retrieve all human evaluations from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM human_evaluations", conn)
        conn.close()
        return df
    
    def get_evaluation_statistics(self) -> Dict:
        """Calculate statistics on human evaluations"""
        df = self.get_human_evaluations()
        
        if df.empty:
            return {}
        
        stats = {
            'total_evaluations': len(df),
            'unique_evaluators': df['evaluator_name'].nunique(),
            'avg_confidence': df['confidence'].mean(),
            'preference_distribution': df['human_preference'].value_counts().to_dict(),
            'topics_evaluated': df['topic'].nunique(),
            'countries_evaluated': df['country'].nunique()
        }
        
        return stats
    
    def render_sidebar(self):
        """Render sidebar with filters and statistics"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Evaluator information
        st.sidebar.subheader("👤 Evaluator Information")
        evaluator_name = st.sidebar.text_input("Your Name", key="evaluator_name")
        evaluator_email = st.sidebar.text_input("Your Email", key="evaluator_email")
        
        if evaluator_name and evaluator_email:
            st.session_state['evaluator_info'] = {
                'name': evaluator_name,
                'email': evaluator_email,
                'session_id': hashlib.md5(f"{evaluator_name}_{datetime.now()}".encode()).hexdigest()[:8]
            }
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        disagreements = self.load_disagreement_cases()
        if not disagreements.empty:
            countries = st.sidebar.multiselect(
                "Countries",
                options=disagreements['country'].unique(),
                default=None
            )
            
            topics = st.sidebar.multiselect(
                "Topics",
                options=disagreements['topic'].unique(),
                default=None
            )
            
            priority = st.sidebar.selectbox(
                "Priority Level",
                options=['All', 'high', 'medium', 'low'],
                index=0
            )
            
            return {'countries': countries, 'topics': topics, 'priority': priority}
        
        return {}
    
    def render_disagreement_case(self, case: pd.Series):
        """Render a single disagreement case for evaluation"""
        st.subheader(f"🌍 {case['country']} - 📋 {case['topic']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Priority", case['priority'].upper(), 
                     delta=f"Disagreement: {case['disagreement_magnitude']:.2f}")
        
        with col2:
            st.metric("Model A", case['model_a'], f"Score: {case['score_a']:.2f}")
        
        with col3:
            st.metric("Model B", case['model_b'], f"Score: {case['score_b']:.2f}")
        
        # Show reasoning from both models
        st.write("### 💭 Model Reasoning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{case['model_a']} Reasoning:**")
            st.info(case['reasoning_a'] if case['reasoning_a'] else "No reasoning provided")
        
        with col2:
            st.write(f"**{case['model_b']} Reasoning:**")
            st.info(case['reasoning_b'] if case['reasoning_b'] else "No reasoning provided")
        
        # Visualization of scores
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[case['model_a'], case['model_b']],
            y=[case['score_a'], case['score_b']],
            marker_color=['blue', 'red']
        ))
        fig.update_layout(
            title="Model Score Comparison",
            yaxis_title="Moral Acceptability Score",
            yaxis_range=[-1, 1],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Human evaluation form
        st.write("### 👨‍⚖️ Your Evaluation")
        
        with st.form(key=f"eval_form_{case.name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                human_score = st.slider(
                    "Your Score for this topic in this country",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="-1: Never justifiable, 0: Neutral, +1: Always justifiable"
                )
                
                preference = st.radio(
                    "Which model's assessment do you prefer?",
                    options=[case['model_a'], case['model_b'], "Neither", "Both equally good"],
                    horizontal=True
                )
                
                confidence = st.slider(
                    "Confidence in your evaluation",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="0: Very uncertain, 100: Very confident"
                )
            
            with col2:
                reasoning = st.text_area(
                    "Explain your reasoning",
                    height=150,
                    placeholder="Why do you agree/disagree with the models? What cultural factors are important?"
                )
                
                notes = st.text_area(
                    "Additional notes (optional)",
                    height=100,
                    placeholder="Any other observations or feedback"
                )
            
            submitted = st.form_submit_button("Submit Evaluation", type="primary")
            
            if submitted:
                if 'evaluator_info' in st.session_state:
                    evaluation_data = {
                        'evaluator_name': st.session_state['evaluator_info']['name'],
                        'evaluator_email': st.session_state['evaluator_info']['email'],
                        'session_id': st.session_state['evaluator_info']['session_id'],
                        'country': case['country'],
                        'topic': case['topic'],
                        'model_a': case['model_a'],
                        'model_b': case['model_b'],
                        'score_a': case['score_a'],
                        'score_b': case['score_b'],
                        'human_score': human_score,
                        'human_preference': preference,
                        'human_reasoning': reasoning,
                        'confidence': confidence,
                        'notes': notes
                    }
                    
                    self.save_human_evaluation(evaluation_data)
                    st.success("✅ Evaluation saved successfully!")
                    st.balloons()
                else:
                    st.error("Please enter your name and email in the sidebar first.")
    
    def render_statistics_page(self):
        """Render statistics and analytics page"""
        st.header("📊 Evaluation Statistics")
        
        stats = self.get_evaluation_statistics()
        
        if not stats:
            st.info("No evaluations yet. Start evaluating disagreement cases!")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Evaluations", stats['total_evaluations'])
        
        with col2:
            st.metric("Unique Evaluators", stats['unique_evaluators'])
        
        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
        
        with col4:
            st.metric("Topics Covered", stats['topics_evaluated'])
        
        # Detailed analytics
        human_evals = self.get_human_evaluations()
        
        # Preference distribution
        if stats['preference_distribution']:
            fig = px.pie(
                values=list(stats['preference_distribution'].values()),
                names=list(stats['preference_distribution'].keys()),
                title="Model Preference Distribution"
            )
            st.plotly_chart(fig)
        
        # Score comparison
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=human_evals['score_a'], name='Model A Scores', opacity=0.7))
        fig.add_trace(go.Histogram(x=human_evals['score_b'], name='Model B Scores', opacity=0.7))
        fig.add_trace(go.Histogram(x=human_evals['human_score'], name='Human Scores', opacity=0.7))
        fig.update_layout(
            title="Score Distribution Comparison",
            xaxis_title="Score",
            yaxis_title="Count",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Agreement analysis
        st.subheader("🤝 Agreement Analysis")
        
        # Calculate agreement metrics
        human_evals['agrees_with_a'] = abs(human_evals['human_score'] - human_evals['score_a']) < 0.3
        human_evals['agrees_with_b'] = abs(human_evals['human_score'] - human_evals['score_b']) < 0.3
        
        col1, col2 = st.columns(2)
        
        with col1:
            agreement_a = human_evals['agrees_with_a'].mean() * 100
            st.metric("Agreement with Model A", f"{agreement_a:.1f}%")
        
        with col2:
            agreement_b = human_evals['agrees_with_b'].mean() * 100
            st.metric("Agreement with Model B", f"{agreement_b:.1f}%")
        
        # Topic-wise analysis
        st.subheader("📋 Topic-wise Analysis")
        topic_stats = human_evals.groupby('topic').agg({
            'human_score': 'mean',
            'confidence': 'mean',
            'case_id': 'count'
        }).round(2)
        topic_stats.columns = ['Avg Human Score', 'Avg Confidence', 'Evaluations']
        st.dataframe(topic_stats)
        
        # Recent evaluations
        st.subheader("🕐 Recent Evaluations")
        recent = human_evals.nlargest(10, 'timestamp')[
            ['timestamp', 'evaluator_name', 'country', 'topic', 'human_score', 'human_preference', 'confidence']
        ]
        st.dataframe(recent)
    
    def render_export_page(self):
        """Render data export page"""
        st.header("💾 Export Data")
        
        human_evals = self.get_human_evaluations()
        
        if human_evals.empty:
            st.info("No data to export yet.")
            return
        
        st.write(f"Total evaluations available for export: {len(human_evals)}")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as CSV"):
                csv = human_evals.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"human_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export as JSON"):
                json_str = human_evals.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"human_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📈 Generate Report"):
                report = self.generate_evaluation_report(human_evals)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Preview data
        st.subheader("Data Preview")
        st.dataframe(human_evals.head(20))
    
    def generate_evaluation_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive evaluation report"""
        stats = self.get_evaluation_statistics()
        
        report = f"""
Human Evaluation Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overview
--------
Total Evaluations: {stats.get('total_evaluations', 0)}
Unique Evaluators: {stats.get('unique_evaluators', 0)}
Average Confidence: {stats.get('avg_confidence', 0):.1f}%
Topics Evaluated: {stats.get('topics_evaluated', 0)}
Countries Evaluated: {stats.get('countries_evaluated', 0)}

Preference Distribution
-----------------------
"""
        
        for pref, count in stats.get('preference_distribution', {}).items():
            report += f"- {pref}: {count} ({count/stats['total_evaluations']*100:.1f}%)\n"
        
        report += """
Score Statistics
----------------
"""
        
        if not df.empty:
            report += f"Human Score Mean: {df['human_score'].mean():.3f}\n"
            report += f"Human Score Std: {df['human_score'].std():.3f}\n"
            report += f"Model A Score Mean: {df['score_a'].mean():.3f}\n"
            report += f"Model B Score Mean: {df['score_b'].mean():.3f}\n"
            
            report += """
Top Disagreement Cases
-----------------------
"""
            
            df['disagreement'] = abs(df['score_a'] - df['score_b'])
            top_disagreements = df.nlargest(5, 'disagreement')
            
            for _, row in top_disagreements.iterrows():
                report += f"\n{row['country']} - {row['topic']}:\n"
                report += f"  {row['model_a']}: {row['score_a']:.2f}\n"
                report += f"  {row['model_b']}: {row['score_b']:.2f}\n"
                report += f"  Human: {row['human_score']:.2f}\n"
                report += f"  Preference: {row['human_preference']}\n"
        
        return report
    
    def run(self):
        """Main dashboard runner"""
        st.title("⚖️ Moral Alignment Human Evaluation Dashboard")
        st.markdown("**Evaluate model disagreements and provide expert arbitration**")
        
        # Initialize session state
        if 'current_case_idx' not in st.session_state:
            st.session_state['current_case_idx'] = 0
        
        # Sidebar
        filters = self.render_sidebar()
        
        # Main navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Evaluate Cases", 
            "📊 Statistics", 
            "💾 Export Data",
            "ℹ️ About"
        ])
        
        with tab1:
            st.header("🔍 Evaluate Disagreement Cases")
            
            # Load and filter cases
            disagreements = self.load_disagreement_cases()
            
            if disagreements.empty:
                st.warning("No disagreement cases found. Run cross-evaluation first.")
            else:
                # Apply filters
                if filters.get('countries'):
                    disagreements = disagreements[disagreements['country'].isin(filters['countries'])]
                
                if filters.get('topics'):
                    disagreements = disagreements[disagreements['topic'].isin(filters['topics'])]
                
                if filters.get('priority') and filters['priority'] != 'All':
                    disagreements = disagreements[disagreements['priority'] == filters['priority']]
                
                if disagreements.empty:
                    st.info("No cases match the selected filters.")
                else:
                    # Navigation
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        if st.button("⬅️ Previous"):
                            if st.session_state['current_case_idx'] > 0:
                                st.session_state['current_case_idx'] -= 1
                    
                    with col2:
                        st.write(f"Case {st.session_state['current_case_idx'] + 1} of {len(disagreements)}")
                    
                    with col3:
                        if st.button("Next ➡️"):
                            if st.session_state['current_case_idx'] < len(disagreements) - 1:
                                st.session_state['current_case_idx'] += 1
                    
                    # Render current case
                    current_case = disagreements.iloc[st.session_state['current_case_idx']]
                    self.render_disagreement_case(current_case)
        
        with tab2:
            self.render_statistics_page()
        
        with tab3:
            self.render_export_page()
        
        with tab4:
            st.header("ℹ️ About This Dashboard")
            st.markdown("""
            ### Purpose
            This dashboard facilitates human evaluation of moral judgment disagreements between language models.
            When models significantly disagree on the moral acceptability of topics in different cultural contexts,
            human expertise is needed to provide ground truth and arbitration.
            
            ### How to Use
            1. **Enter your information** in the sidebar (name and email)
            2. **Navigate through cases** using Previous/Next buttons
            3. **Evaluate each case** by providing:
               - Your own moral acceptability score
               - Which model's assessment you prefer
               - Your confidence level
               - Reasoning for your judgment
            4. **View statistics** to see overall evaluation patterns
            5. **Export data** for further analysis
            
            ### Scoring Guide
            - **-1.0**: Never morally justifiable
            - **-0.5**: Usually not justifiable
            - **0.0**: Neutral/context-dependent
            - **+0.5**: Usually justifiable
            - **+1.0**: Always morally justifiable
            
            ### Contact
            For questions or issues, please contact the research team.
            """)


# Main execution
def main():
    dashboard = HumanEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()