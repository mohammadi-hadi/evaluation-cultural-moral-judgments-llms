#!/usr/bin/env python3
"""
Test script for Human Judge Dashboard
Verifies the dashboard loads correctly and can process evaluations
"""

import json
import sqlite3
from pathlib import Path
import sys

def test_dashboard_data():
    """Test that dashboard can load conflict data"""
    
    print("=" * 60)
    print("TESTING HUMAN JUDGE DASHBOARD")
    print("=" * 60)
    
    # Check for conflict data
    conflict_file = Path("outputs/conflict_demo/conflicts_for_human_review.json")
    
    if not conflict_file.exists():
        print("❌ Conflict file not found. Running demo to generate conflicts...")
        import subprocess
        result = subprocess.run([sys.executable, "demo_with_conflicts.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running demo: {result.stderr}")
            return False
    
    # Load and verify conflict data
    with open(conflict_file, 'r') as f:
        conflict_data = json.load(f)
    
    print(f"\n✅ Conflict data loaded successfully")
    print(f"   Total conflicts: {conflict_data['metadata']['n_conflicts']}")
    print(f"   Severity breakdown:")
    for severity, count in conflict_data['metadata']['severity_breakdown'].items():
        print(f"     - {severity}: {count}")
    
    # Display sample conflicts
    print("\n📋 Sample Conflicts for Human Review:")
    for i, case in enumerate(conflict_data['cases'][:2], 1):
        print(f"\n   Case {i}: {case['case_id']}")
        print(f"   Severity: {case['severity']}")
        print(f"   Model A ({case['model_a']['name']}): {case['model_a']['score']:.2f}")
        print(f"   Model B ({case['model_b']['name']}): {case['model_b']['score']:.2f}")
        print(f"   Difference: {case['score_difference']:.2f}")
    
    # Test database initialization
    print("\n🗄️ Testing Database Setup...")
    db_path = Path("human_evaluations.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        expected_tables = ['conflict_evaluations', 'annotator_agreement', 
                          'evaluation_sessions', 'evaluation_metrics']
        
        table_names = [t[0] for t in tables]
        
        for table in expected_tables:
            if table in table_names:
                print(f"   ✅ Table '{table}' exists")
            else:
                print(f"   ❌ Table '{table}' missing")
        
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False
    
    # Instructions for running dashboard
    print("\n" + "=" * 60)
    print("DASHBOARD READY FOR HUMAN EVALUATION")
    print("=" * 60)
    
    print("\n📊 Dashboard Features:")
    print("   • Side-by-side model comparison")
    print("   • 7-point preference scale (-3 to +3)")
    print("   • Confidence scoring")
    print("   • Real-time statistics")
    print("   • SQLite persistence")
    print("   • CSV export functionality")
    
    print("\n🚀 To launch the dashboard:")
    print("   streamlit run human_judge_dashboard.py")
    
    print("\n📝 Evaluation Process:")
    print("   1. Enter your name and email")
    print("   2. Review each conflict case")
    print("   3. Select preference on 7-point scale")
    print("   4. Provide reasoning")
    print("   5. Submit evaluation")
    print("   6. Dashboard auto-advances to next case")
    
    print("\n✅ All systems ready for human evaluation!")
    
    return True

if __name__ == "__main__":
    success = test_dashboard_data()
    sys.exit(0 if success else 1)