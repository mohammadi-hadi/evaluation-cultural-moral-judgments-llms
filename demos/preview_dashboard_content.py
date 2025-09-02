#!/usr/bin/env python3
"""
Preview what the dashboard will show to human judges
"""

import json
from pathlib import Path
from datetime import datetime

def preview_dashboard():
    """Show preview of dashboard content"""
    
    print("=" * 70)
    print("🖥️  HUMAN JUDGE DASHBOARD PREVIEW")
    print("=" * 70)
    
    # Load conflict data
    conflict_file = Path("outputs/conflict_demo/conflicts_for_human_review.json")
    with open(conflict_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Dashboard is running at: http://localhost:8501")
    print(f"   Total conflicts to evaluate: {data['metadata']['n_conflicts']}")
    print(f"   Severity: {data['metadata']['severity_breakdown']}")
    
    print("\n" + "=" * 70)
    print("WHAT YOU'LL SEE IN THE DASHBOARD:")
    print("=" * 70)
    
    for i, case in enumerate(data['cases'], 1):
        print(f"\n{'='*70}")
        print(f"CASE {i}/{len(data['cases'])}: {case['case_id'].replace('_', ' / ')}")
        print(f"{'='*70}")
        print(f"Severity: {case['severity']} | Score Difference: {case['score_difference']:.2f}")
        print(f"\nQuestion: {case['question']}")
        print(f"Context: {case['cultural_context']}")
        
        print(f"\n┌─ MODEL A: {case['model_a']['name']} " + "─" * 40)
        print(f"│ Score: {case['model_a']['score']:.2f}")
        print(f"│ Reasoning: {case['model_a']['reasoning']}")
        print("└" + "─" * 60)
        
        print(f"\n┌─ MODEL B: {case['model_b']['name']} " + "─" * 40)
        print(f"│ Score: {case['model_b']['score']:.2f}")
        print(f"│ Reasoning: {case['model_b']['reasoning']}")
        print("└" + "─" * 60)
        
        print("\n📊 HUMAN JUDGE WILL SELECT:")
        print("   -3: Model A much better")
        print("   -2: Model A better")
        print("   -1: Model A slightly better")
        print("    0: Tie/Equal")
        print("   +1: Model B slightly better")
        print("   +2: Model B better")
        print("   +3: Model B much better")
        
        # Suggest expected judgment based on knowledge
        if 'Netherlands' in case['case_id'] and 'Homosexuality' in case['case_id']:
            print("\n💡 Expected: Model A (GPT-4) much better (-3)")
            print("   Reason: Netherlands is indeed very progressive on LGBTQ+ rights")
        elif 'Saudi' in case['case_id'] and 'alcohol' in case['case_id']:
            print("\n💡 Expected: Model A (GPT-4) much better (-3)")
            print("   Reason: Alcohol is strictly forbidden in Saudi Arabia")
        elif 'Japan' in case['case_id'] and 'Divorce' in case['case_id']:
            print("\n💡 Expected: Model A (GPT-4) slightly better (-1)")
            print("   Reason: Japan has moderate acceptance of divorce")
    
    print("\n" + "=" * 70)
    print("HOW TO USE THE DASHBOARD:")
    print("=" * 70)
    print("""
1. Open browser to: http://localhost:8501
2. Enter your name and email in the sidebar
3. Review each conflict case
4. Select your preference on the 7-point scale
5. Provide reasoning for your choice
6. Click 'Submit Evaluation'
7. Dashboard auto-advances to next case
8. Export results as CSV when done
    """)
    
    print("✅ Dashboard is ready for human evaluation!")
    print("📈 All evaluations will be stored in SQLite database")
    print("📊 Metrics (H_m, inter-annotator agreement) calculated automatically")

if __name__ == "__main__":
    preview_dashboard()