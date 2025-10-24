#!/usr/bin/env python3
"""
Content Quality Ratings Calculation Script
Shows how the values in the Content Quality radar chart were derived
"""

import pandas as pd
import numpy as np

def main():
    # Load the enhanced dataset
    df = pd.read_csv('enhanced_wrapstat_training_data.csv')
    
    print("="*60)
    print("CONTENT QUALITY RATINGS CALCULATION")
    print("="*60)
    
    # The content quality ratings come from these survey columns:
    content_columns = {
        'content_engaging_numeric': 'Engaging/Held Attention',
        'content_relevant_numeric': 'Relevant to Job', 
        'content_understandable_numeric': 'Easy to Understand',
        'content_interactive_numeric': 'Appropriately Interactive',
        'content_visual_support_numeric': 'Supported by Visuals',
        'content_user_friendly_numeric': 'User-Friendly'
    }
    
    print("\n📊 ORIGINAL SURVEY QUESTIONS:")
    print("Participants rated each aspect on a 5-point Likert scale:")
    print("1 = Strongly Disagree")
    print("2 = Somewhat Disagree") 
    print("3 = Neither Agree nor Disagree")
    print("4 = Somewhat Agree")
    print("5 = Strongly Agree")
    
    print("\n📈 CALCULATED AVERAGES:")
    print("-" * 50)
    
    chart_data = []
    for col, label in content_columns.items():
        # Calculate average, excluding NaN values
        avg_rating = df[col].mean()
        chart_data.append(avg_rating)
        
        # Show detailed stats
        total_responses = df[col].count()
        print(f"{label:25} | Avg: {avg_rating:.2f} | Responses: {total_responses}")
        
        # Show distribution
        value_counts = df[col].value_counts().sort_index()
        print(f"                         | Distribution: {dict(value_counts)}")
        print()
    
    print("="*60)
    print("CHART DATA ARRAY (used in dashboard):")
    print(f"data: {[round(x, 2) for x in chart_data]}")
    
    print("\n🔍 DETAILED BREAKDOWN:")
    print("-" * 50)
    
    for i, (col, label) in enumerate(content_columns.items()):
        print(f"\n{i+1}. {label}:")
        print(f"   Column: {col}")
        
        # Show actual responses
        responses = df[col].dropna()
        print(f"   Raw scores: {list(responses)}")
        print(f"   Average: {responses.mean():.4f}")
        print(f"   Rounded: {round(responses.mean(), 2)}")
    
    print("\n" + "="*60)
    print("RADAR CHART INTERPRETATION:")
    print("• Higher values (closer to 5) = Better ratings")
    print("• Lower values (closer to 1) = Poorer ratings") 
    print("• The radar chart shows relative strengths/weaknesses")
    print("• 'Supported by Visuals' scores highest (4.46)")
    print("• 'Appropriately Interactive' scores lowest (3.58)")

if __name__ == "__main__":
    main()