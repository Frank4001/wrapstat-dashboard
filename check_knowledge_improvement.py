import pandas as pd

# Read the enhanced dataset
df = pd.read_csv('enhanced_wrapstat_training_data.csv')

print("=== KNOWLEDGE IMPROVEMENT CALCULATION ===\n")

print("=== STEP 1: KNOWLEDGE MAPPING SCALE ===")
knowledge_mapping = {
    'Extremely bad': 1,
    'Somewhat bad': 2,
    'Neither good nor bad': 3,
    'Somewhat good': 4,
    'Extremely good': 5
}

print("Knowledge Scale Used:")
for text, number in knowledge_mapping.items():
    print(f"  '{text}' = {number}")

print(f"\n=== STEP 2: CALCULATION METHOD ===")
print("Knowledge Improvement = Knowledge After - Knowledge Before")
print("Formula: knowledge_after_numeric - knowledge_before_numeric")

print(f"\n=== STEP 3: SAMPLE CALCULATIONS ===")
sample_data = df[['response_id', 'knowledge_before', 'knowledge_before_numeric', 
                 'knowledge_after', 'knowledge_after_numeric', 'knowledge_improvement']].head(10)

for _, row in sample_data.iterrows():
    print(f"Response {row['response_id']}:")
    print(f"  Before: '{row['knowledge_before']}' = {row['knowledge_before_numeric']}")
    print(f"  After:  '{row['knowledge_after']}' = {row['knowledge_after_numeric']}")
    print(f"  Improvement: {row['knowledge_after_numeric']} - {row['knowledge_before_numeric']} = {row['knowledge_improvement']}")
    print()

print(f"=== STEP 4: OVERALL STATISTICS ===")
improvement_counts = df['knowledge_improvement'].value_counts().sort_index()
print("Knowledge Improvement Distribution:")
for improvement, count in improvement_counts.items():
    print(f"  {improvement:+2d} points: {count} participants")

print(f"\nAverage Knowledge Improvement:")
print(f"  Sum of all improvements: {df['knowledge_improvement'].sum()}")
print(f"  Number of participants: {len(df)}")
print(f"  Average = {df['knowledge_improvement'].sum()} ÷ {len(df)} = {df['knowledge_improvement'].mean():.4f}")
print(f"  Rounded = {df['knowledge_improvement'].mean():.2f} points")

print(f"\n=== STEP 5: INTERPRETATION ===")
avg_improvement = df['knowledge_improvement'].mean()
if avg_improvement > 1:
    interpretation = "EXCELLENT - Strong knowledge gain"
elif avg_improvement > 0.5:
    interpretation = "GOOD - Moderate knowledge gain"
elif avg_improvement > 0:
    interpretation = "FAIR - Some knowledge gain"
else:
    interpretation = "NEEDS IMPROVEMENT - Little to no gain"

print(f"Average improvement of {avg_improvement:.2f} points = {interpretation}")

print(f"\n=== EXAMPLES OF IMPROVEMENT LEVELS ===")
print("Examples from the data:")
for improvement_level in sorted(df['knowledge_improvement'].unique()):
    example = df[df['knowledge_improvement'] == improvement_level].iloc[0]
    print(f"  {improvement_level:+2d} points: {example['knowledge_before']} → {example['knowledge_after']}")