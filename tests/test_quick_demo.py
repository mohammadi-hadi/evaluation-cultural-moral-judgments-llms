#!/usr/bin/env python3
"""
Quick Demo Test for Moral Alignment Pipeline
Tests with minimal samples to demonstrate functionality
"""

import logging
from wvs_processor import WVSProcessor
from moral_alignment_tester import MoralAlignmentTester
from validation_suite import ValidationSuite
from paper_outputs import PaperOutputGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_quick_demo():
    """Run quick demo with minimal samples"""
    
    print("=" * 60)
    print("MORAL ALIGNMENT PIPELINE - QUICK DEMO")
    print("=" * 60)
    
    # 1. Test WVS Processor
    print("\n1️⃣ Testing WVS Data Processor...")
    wvs = WVSProcessor()
    wvs.load_data()
    wvs.process_moral_scores()
    
    # Create small evaluation dataset
    eval_data = wvs.create_evaluation_dataset(n_samples=10)
    print(f"✅ Created evaluation dataset: {len(eval_data)} samples")
    print(f"   Countries: {eval_data['country'].nunique()}")
    print(f"   Topics: {eval_data['topic'].nunique()}")
    
    # Show sample data
    print("\nSample evaluation data:")
    print(eval_data[['country', 'topic', 'normalized_score']].head(3))
    
    # 2. Calculate human baseline
    print("\n2️⃣ Calculating Human Baseline...")
    baseline = wvs.calculate_human_baseline()
    print(f"✅ Human baseline calculated")
    print(f"   Overall mean: {baseline['overall_mean']:.3f}")
    print(f"   Overall std: {baseline['overall_std']:.3f}")
    print(f"   Topics analyzed: {len(baseline['by_topic'])}")
    
    # 3. Create prompts
    print("\n3️⃣ Creating Evaluation Prompts...")
    prompts = wvs.create_prompts_for_evaluation(eval_data.head(5))
    print(f"✅ Created {len(prompts)} prompts")
    print("\nExample prompt:")
    print(f"Country: {prompts[0]['country']}")
    print(f"Topic: {prompts[0]['topic']}")
    print(f"Type: {prompts[0]['type']}")
    print(f"Ground truth: {prompts[0]['ground_truth']:.3f}")
    
    # 4. Save processed data
    print("\n4️⃣ Saving Processed Data...")
    wvs.save_processed_data()
    print("✅ Data saved to outputs/wvs_processed/")
    
    # 5. Generate country-topic means
    print("\n5️⃣ Analyzing Country-Topic Patterns...")
    means = wvs.get_country_topic_means()
    print(f"✅ Calculated means for {len(means)} country-topic pairs")
    
    # Show top controversial topics
    topic_variance = wvs.processed_data.groupby('topic')['normalized_score'].var().sort_values(ascending=False)
    print("\nMost controversial topics (highest variance):")
    for topic, var in topic_variance.head(3).items():
        print(f"  - {topic}: variance = {var:.3f}")
    
    # Show countries with most extreme views
    country_means = wvs.processed_data.groupby('country')['normalized_score'].mean().sort_values()
    print("\nCountries with most conservative views:")
    for country in country_means.head(3).index:
        print(f"  - {country}: mean = {country_means[country]:.3f}")
    
    print("\nCountries with most liberal views:")
    for country in country_means.tail(3).index:
        print(f"  - {country}: mean = {country_means[country]:.3f}")
    
    # 6. Test validation suite
    print("\n6️⃣ Testing Validation Suite...")
    validator = ValidationSuite()
    
    # Create mock results for validation demo
    mock_results = {
        'scores': [
            {'model_score': 0.5, 'ground_truth': 0.6, 'method': 'direct', 
             'country': 'USA', 'topic': 'Abortion'},
            {'model_score': -0.3, 'ground_truth': -0.4, 'method': 'direct',
             'country': 'Japan', 'topic': 'Divorce'},
            {'model_score': 0.2, 'ground_truth': 0.1, 'method': 'logprob',
             'country': 'Germany', 'topic': 'Homosexuality'}
        ]
    }
    
    validation = validator.validate_model_results(mock_results, 'demo_model')
    print("✅ Validation complete")
    
    if 'statistical_validity' in validation:
        if 'correlations' in validation['statistical_validity']:
            corr = validation['statistical_validity']['correlations']
            if 'pearson' in corr:
                print(f"   Pearson correlation: {corr['pearson']['r']:.3f}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - READY FOR FULL EVALUATION")
    print("=" * 60)
    
    print("\n📊 Data Summary:")
    print(f"  Total WVS samples: {len(wvs.data):,}")
    print(f"  Processed judgments: {len(wvs.processed_data):,}")
    print(f"  Countries: {wvs.processed_data['country'].nunique()}")
    print(f"  Topics: {len(wvs.WVS_TOPICS)}")
    
    print("\n🚀 Next Steps:")
    print("1. Run full evaluation: python run_full_evaluation.py --mode standard")
    print("2. Test specific models: python run_full_evaluation.py --models gpt-4o gpt-4o-mini")
    print("3. Generate paper outputs: python paper_outputs.py")
    
    print("\n💡 Tips for Paper Representation:")
    print("• Table 1: Model correlations with WVS (ρ^LP and ρ^Dir)")
    print("• Figure 2: Country-wise correlation heatmap")
    print("• Figure 3: Error density plots by method")
    print("• Figure 4: Topic-specific error heatmap")
    print("• Figure 5: Regional preference analysis")
    
    return wvs, eval_data, baseline


if __name__ == "__main__":
    wvs, eval_data, baseline = run_quick_demo()