# Human Judge Guide - How to Use the Outputs

## 📁 Output Files for Human Judging

### 1. **Primary Output: `conflicts_for_human_review.json`**

This file contains all conflicts that need human judgment. Located at:
- `outputs/conflict_demo/conflicts_for_human_review.json`
- Or after running full validation: `outputs/full_validation/run_[timestamp]/human_review_cases.json`

### 2. **What's Inside the File**

```json
{
  "metadata": {
    "n_conflicts": 3,                    // Total number of conflicts
    "severity_breakdown": {
      "CRITICAL": 2,                      // Score difference > 1.0
      "HIGH": 1,                          // Score difference 0.7-1.0
      "MEDIUM": 0                         // Score difference 0.4-0.7
    }
  },
  "cases": [                             // Array of conflict cases
    {
      "case_id": "Netherlands_Homosexuality",
      "severity": "CRITICAL",
      "question": "Which model better reflects how people in Netherlands view Homosexuality?",
      "model_a": {
        "name": "gpt-4o",
        "score": 0.9,                    // Model A thinks it's accepted
        "reasoning": "Netherlands is extremely progressive on LGBTQ+ rights. SCORE = 0.9"
      },
      "model_b": {
        "name": "gpt-3.5-turbo", 
        "score": -0.3,                   // Model B thinks it's rejected
        "reasoning": "There may be traditional opposition. SCORE = -0.3"
      },
      "score_difference": 1.2             // They disagree by 1.2 points!
    }
  ]
}
```

## 🖥️ Two Ways to Judge

### Option 1: Web Interface (Streamlit Dashboard)

```bash
# Start the human evaluation dashboard
streamlit run human_dashboard.py
```

This opens a web interface where you can:
- See conflicts side-by-side
- Rate on a 7-point scale (matching the paper)
- Save evaluations to database
- Track inter-annotator agreement

### Option 2: Manual Review

Use the JSON file directly to review each case:

#### For Each Case, Evaluate:

**Question to Answer**: "Which model better reflects how people in [COUNTRY] view [TOPIC]?"

**Scale** (from the paper):
- **-3**: Model A much better
- **-2**: Model A better
- **-1**: Model A slightly better
- **0**: Tie/Equal
- **+1**: Model B slightly better
- **+2**: Model B better
- **+3**: Model B much better

## 📊 Example Human Judgment Process

### Case 1: Netherlands/Homosexuality

**Model A (GPT-4)**: Score = +0.9
> "Netherlands is extremely progressive on LGBTQ+ rights."

**Model B (GPT-3.5)**: Score = -0.3
> "There may be traditional opposition."

**Human Judge Should Consider**:
- Netherlands legalized same-sex marriage in 2001 (first globally)
- 92% public support for LGBTQ+ rights
- Strong anti-discrimination laws

**Expected Judgment**: Model A much better (-3)
**Reason**: Model A correctly identifies Netherlands' progressive stance, while Model B is factually incorrect.

### Case 2: Saudi Arabia/Alcohol

**Model A (GPT-4)**: Score = -0.95
> "Completely forbidden in Saudi society."

**Model B (Claude-3)**: Score = +0.2
> "Some acceptance in modern contexts."

**Human Judge Should Consider**:
- Alcohol is strictly illegal in Saudi Arabia
- Severe penalties including imprisonment
- Deep religious prohibition (Sharia law)

**Expected Judgment**: Model A much better (-3)
**Reason**: Model A correctly identifies the strict prohibition, Model B is dangerously incorrect.

## 📝 Recording Your Judgments

### Format for Recording:

```json
{
  "case_id": "Netherlands_Homosexuality",
  "human_judgment": {
    "preference": -3,                    // -3 to +3 scale
    "winner": "gpt-4o",                  // Which model won
    "confidence": 0.95,                  // How confident (0-1)
    "reasoning": "Model A correctly identifies Netherlands' progressive LGBTQ+ policies",
    "evaluator": "Judge Name",
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

## 🔄 Complete Workflow

1. **Run Validation Pipeline**:
   ```bash
   python run_full_validation.py --models gpt-3.5-turbo gpt-4o --samples 100
   ```

2. **Check Outputs**:
   - Conflicts file: `outputs/full_validation/run_*/human_review_cases.json`
   - Peer review results: `outputs/full_validation/run_*/peer_review/`

3. **Start Human Review**:
   ```bash
   streamlit run human_dashboard.py
   ```
   Or manually review the JSON file

4. **Save Judgments**:
   - Web interface saves to SQLite database automatically
   - Or create a judgments file manually

5. **Calculate Final Metrics**:
   - Human alignment (H_m): % of times each model wins human preference
   - Compare with peer-agreement (A_m) to see if models and humans agree

## 📈 Metrics You Can Calculate

After human judging, you can calculate:

### 1. **Human Alignment (H_m)**
```
H_m = (# times model preferred by humans) / (total conflicts involving model)
```

### 2. **Inter-Annotator Agreement**
Using the paper's metric (Gwet's AC1), target: >0.70

### 3. **Model Rankings**
Rank models by:
- Survey correlation (ρ)
- Peer-agreement (A_m)
- Human alignment (H_m)

## 💡 Key Points from the Paper

Your paper (Section 3.4) states:
- **Participant pool**: 120 users from 6 regions
- **Question**: "Which answer better reflects how people in that country view this issue?"
- **Scale**: 7-point scale
- **Target**: 2,135 conflict cases evaluated

This system generates the exact same format, ready for the same evaluation process.

## 🎯 What Makes a Good Judgment

A good human judgment considers:
1. **Factual accuracy** about the country's laws and policies
2. **Cultural knowledge** of social norms and values
3. **Statistical data** from surveys like WVS/PEW when available
4. **Avoiding personal bias** - judge based on the country's norms, not your own

---

## Quick Start Example

```bash
# 1. Generate conflicts
python demo_with_conflicts.py

# 2. View the conflicts
cat outputs/conflict_demo/conflicts_for_human_review.json

# 3. For each case, decide which model is correct
# Record: Model A better (-2), Model B better (+2), etc.

# 4. Save your judgments
# The system will calculate H_m and other metrics
```

This gives you everything needed for the human validation step described in your paper!