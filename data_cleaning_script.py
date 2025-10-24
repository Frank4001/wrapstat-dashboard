import pandas as pd
import numpy as np
from datetime import datetime
import re

def clean_wrapstat_data():
    """
    Clean and prepare WrapStat training survey data for dashboard creation
    """
    
    # Read the cleaned CSV file
    df = pd.read_csv('cleaned_wrapstat_training_data.csv')
    
    print("Original dataset shape:", df.shape)
    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Convert date columns to datetime
    date_columns = ['start_date', 'end_date', 'recorded_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Create additional useful columns
    df['training_month'] = df['start_date'].dt.strftime('%Y-%m')
    df['training_year'] = df['start_date'].dt.year
    df['training_duration_minutes'] = df['duration_seconds'] / 60
    
    # Map Likert scale responses to numeric values for analysis
    likert_mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2, 
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5
    }
    
    knowledge_mapping = {
        'Extremely bad': 1,
        'Somewhat bad': 2,
        'Neither good nor bad': 3,
        'Somewhat good': 4,
        'Extremely good': 5
    }
    
    # Apply mappings to create numeric versions
    likert_columns = [
        'improved_user_access', 'improved_care_coordinator_mgmt', 
        'improved_youth_roster', 'improved_followup', 'content_engaging',
        'content_relevant', 'content_understandable', 'content_interactive',
        'content_visual_support', 'content_user_friendly', 'tech_easy_access',
        'tech_easy_navigate', 'tech_functioning_properly'
    ]
    
    for col in likert_columns:
        df[f'{col}_numeric'] = df[col].map(likert_mapping)
    
    # Map knowledge columns
    knowledge_columns = ['knowledge_before', 'knowledge_after']
    for col in knowledge_columns:
        df[f'{col}_numeric'] = df[col].map(knowledge_mapping)
    
    # Calculate knowledge improvement
    df['knowledge_improvement'] = df['knowledge_after_numeric'] - df['knowledge_before_numeric']
    
    # Create binary columns for analysis
    df['would_recommend'] = df['would_take_another_course'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['had_tech_issues'] = df['had_technical_difficulties'].apply(lambda x: 1 if 'Yes' in str(x) else 0)
    df['provided_comments'] = df['final_comments'].notna().astype(int)
    
    # Calculate average satisfaction scores
    content_cols = [col for col in df.columns if col.startswith('content_') and col.endswith('_numeric')]
    tech_cols = [col for col in df.columns if col.startswith('tech_') and col.endswith('_numeric')]
    improvement_cols = [col for col in df.columns if col.startswith('improved_') and col.endswith('_numeric')]
    
    df['avg_content_satisfaction'] = df[content_cols].mean(axis=1)
    df['avg_tech_satisfaction'] = df[tech_cols].mean(axis=1)
    df['avg_improvement_rating'] = df[improvement_cols].mean(axis=1)
    
    # Create location regions based on coordinates (rough Illinois regions)
    def assign_region(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return 'Unknown'
        
        # These are approximate boundaries for Illinois regions
        if lat > 42.0:
            return 'Northern Illinois'
        elif lat > 40.0:
            return 'Central Illinois'
        else:
            return 'Southern Illinois'
    
    df['region'] = df.apply(lambda row: assign_region(row['latitude'], row['longitude']), axis=1)
    
    # Save the enhanced dataset
    df.to_csv('enhanced_wrapstat_training_data.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_responses': len(df),
        'date_range': f"{df['start_date'].min().strftime('%Y-%m-%d')} to {df['start_date'].max().strftime('%Y-%m-%d')}",
        'avg_duration_minutes': round(df['training_duration_minutes'].mean(), 1),
        'completion_rate': f"{(df['completed'] == True).sum() / len(df) * 100:.1f}%",
        'would_recommend_rate': f"{df['would_recommend'].mean() * 100:.1f}%",
        'avg_knowledge_before': round(df['knowledge_before_numeric'].mean(), 2),
        'avg_knowledge_after': round(df['knowledge_after_numeric'].mean(), 2),
        'avg_knowledge_improvement': round(df['knowledge_improvement'].mean(), 2),
        'tech_issues_rate': f"{df['had_tech_issues'].mean() * 100:.1f}%",
        'avg_content_satisfaction': round(df['avg_content_satisfaction'].mean(), 2),
        'avg_tech_satisfaction': round(df['avg_tech_satisfaction'].mean(), 2)
    }
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for key, value in summary_stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Create a simplified dataset for basic dashboards
    dashboard_columns = [
        'start_date', 'training_month', 'training_year', 'duration_seconds', 'training_duration_minutes',
        'region', 'knowledge_before', 'knowledge_after', 'knowledge_improvement',
        'would_take_another_course', 'would_recommend', 'had_tech_issues',
        'avg_content_satisfaction', 'avg_tech_satisfaction', 'avg_improvement_rating',
        'final_comments', 'provided_comments'
    ]
    
    dashboard_df = df[dashboard_columns].copy()
    dashboard_df.to_csv('dashboard_ready_wrapstat_data.csv', index=False)
    
    print(f"\n✅ Created 3 files:")
    print(f"   1. cleaned_wrapstat_training_data.csv - Clean structured data")
    print(f"   2. enhanced_wrapstat_training_data.csv - Full analysis-ready dataset")
    print(f"   3. dashboard_ready_wrapstat_data.csv - Simplified for dashboards")
    
    return df, summary_stats

if __name__ == "__main__":
    df, stats = clean_wrapstat_data()