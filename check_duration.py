import pandas as pd

# Read the enhanced dataset
df = pd.read_csv('enhanced_wrapstat_training_data.csv')

print("=== AVERAGE DURATION CALCULATION ===\n")

print("First 10 duration values:")
print(df[['response_id', 'duration_seconds', 'training_duration_minutes']].head(10))

print(f"\n=== DURATION STATISTICS ===")
print(f"Total responses: {len(df)}")
print(f"Average duration: {df['training_duration_minutes'].mean():.4f} minutes")
print(f"Median duration: {df['training_duration_minutes'].median():.4f} minutes")
print(f"Min duration: {df['training_duration_minutes'].min():.4f} minutes")
print(f"Max duration: {df['training_duration_minutes'].max():.4f} minutes")

print(f"\n=== HOW CALCULATION WORKS ===")
print("1. Original data has 'duration_seconds' column")
print("2. I converted seconds to minutes: duration_seconds ÷ 60")
print("3. Then calculated the mean of all training_duration_minutes")

print(f"\n=== ALL DURATION VALUES ===")
durations = df[['response_id', 'duration_seconds', 'training_duration_minutes']].sort_values('duration_seconds')
for _, row in durations.iterrows():
    print(f"Response {row['response_id']}: {row['duration_seconds']} seconds = {row['training_duration_minutes']:.2f} minutes")

print(f"\nCalculation verification:")
print(f"Sum of all durations: {df['training_duration_minutes'].sum():.2f} minutes")
print(f"Number of responses: {len(df)}")
print(f"Average = {df['training_duration_minutes'].sum():.2f} ÷ {len(df)} = {df['training_duration_minutes'].mean():.2f} minutes")