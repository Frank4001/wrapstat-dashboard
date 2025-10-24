import pandas as pd
import numpy as np

def calculate_dashboard_stats():
    """
    Calculate actual statistics from the enhanced dataset for the dashboard
    """
    
    # Read the enhanced dataset
    df = pd.read_csv('enhanced_wrapstat_training_data.csv')
    
    print("=== DASHBOARD DATA VERIFICATION ===\n")
    
    # Knowledge Assessment Distribution
    print("1. KNOWLEDGE ASSESSMENT BEFORE TRAINING:")
    knowledge_before_counts = df['knowledge_before'].value_counts().sort_index()
    print(knowledge_before_counts)
    
    print("\n2. KNOWLEDGE ASSESSMENT AFTER TRAINING:")
    knowledge_after_counts = df['knowledge_after'].value_counts().sort_index()
    print(knowledge_after_counts)
    
    # Learning Effectiveness Ratings
    print("\n3. LEARNING EFFECTIVENESS RATINGS (Average):")
    learning_columns = [
        'improved_user_access_numeric',
        'improved_care_coordinator_mgmt_numeric', 
        'improved_youth_roster_numeric',
        'improved_followup_numeric'
    ]
    
    for col in learning_columns:
        avg = df[col].mean()
        print(f"{col.replace('_numeric', '').replace('_', ' ').title()}: {avg:.2f}")
    
    # Content Quality Ratings
    print("\n4. CONTENT QUALITY RATINGS (Average):")
    content_columns = [
        'content_engaging_numeric',
        'content_relevant_numeric',
        'content_understandable_numeric',
        'content_interactive_numeric',
        'content_visual_support_numeric',
        'content_user_friendly_numeric'
    ]
    
    for col in content_columns:
        avg = df[col].mean()
        print(f"{col.replace('_numeric', '').replace('content_', '').replace('_', ' ').title()}: {avg:.2f}")
    
    # Technology Experience Ratings
    print("\n5. TECHNOLOGY EXPERIENCE RATINGS (Average):")
    tech_columns = [
        'tech_easy_access_numeric',
        'tech_easy_navigate_numeric',
        'tech_functioning_properly_numeric'
    ]
    
    for col in tech_columns:
        avg = df[col].mean()
        print(f"{col.replace('_numeric', '').replace('tech_', '').replace('_', ' ').title()}: {avg:.2f}")
    
    # Geographic Distribution
    print("\n6. GEOGRAPHIC DISTRIBUTION:")
    region_counts = df['region'].value_counts()
    print(region_counts)
    
    # Knowledge Improvement Distribution
    print("\n7. KNOWLEDGE IMPROVEMENT DISTRIBUTION:")
    improvement_counts = df['knowledge_improvement'].value_counts().sort_index()
    print(improvement_counts)
    
    # Training Duration Distribution
    print("\n8. TRAINING DURATION DISTRIBUTION:")
    # Create duration bins
    bins = [0, 1, 2, 3, 4, float('inf')]
    labels = ['<1 min', '1-2 min', '2-3 min', '3-4 min', '4+ min']
    df['duration_bins'] = pd.cut(df['training_duration_minutes'], bins=bins, labels=labels, right=False)
    duration_counts = df['duration_bins'].value_counts()
    print(duration_counts)
    
    # Monthly Response Distribution
    print("\n9. MONTHLY RESPONSE DISTRIBUTION:")
    monthly_counts = df['training_month'].value_counts().sort_index()
    print(monthly_counts)
    
    # Recommendation and Technical Issues
    print("\n10. SATISFACTION METRICS:")
    recommend_rate = df['would_recommend'].mean() * 100
    tech_issues_rate = df['had_tech_issues'].mean() * 100
    print(f"Would Recommend Rate: {recommend_rate:.1f}%")
    print(f"Technical Issues Rate: {tech_issues_rate:.1f}%")
    
    # Overall Satisfaction Averages
    print("\n11. OVERALL SATISFACTION AVERAGES:")
    print(f"Content Quality: {df['avg_content_satisfaction'].mean():.2f}")
    print(f"Technology Experience: {df['avg_tech_satisfaction'].mean():.2f}")
    print(f"Learning Effectiveness: {df['avg_improvement_rating'].mean():.2f}")
    
    # All unique feedback comments
    print("\n12. ALL PARTICIPANT FEEDBACK:")
    feedback_comments = df[df['final_comments'].notna()]['final_comments'].unique()
    for i, comment in enumerate(feedback_comments, 1):
        print(f"{i:2d}. \"{comment}\"")
    
    # Reasons for not taking another course
    print("\n13. REASONS FOR NOT TAKING ANOTHER COURSE:")
    no_reasons = df[df['would_take_another_course_reason'].notna()]['would_take_another_course_reason'].unique()
    for reason in no_reasons:
        print(f"- \"{reason}\"")
    
    # Technical difficulties explanations
    print("\n14. TECHNICAL DIFFICULTIES EXPLANATIONS:")
    tech_explanations = df[df['technical_difficulties_explanation'].notna()]['technical_difficulties_explanation'].unique()
    for explanation in tech_explanations:
        print(f"- \"{explanation}\"")

if __name__ == "__main__":
    calculate_dashboard_stats()