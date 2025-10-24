import pandas as pd

# Read the enhanced dataset
df = pd.read_csv('enhanced_wrapstat_training_data.csv')

print("=== IMPROVED USER ACCESS MANAGEMENT RATING CALCULATION ===\n")

print("=== STEP 1: ORIGINAL SURVEY QUESTION ===")
print("Question: 'Taking this course has improved my ability to manage WrapStat user access within my CCSO.'")
print("Response Scale: 5-point Likert scale")

print(f"\n=== STEP 2: LIKERT SCALE MAPPING ===")
likert_mapping = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5
}

print("Likert Scale Used:")
for text, number in likert_mapping.items():
    print(f"  '{text}' = {number}")

print(f"\n=== STEP 3: RAW RESPONSES ===")
print("All participant responses to the User Access Management question:")

# Show the original text responses and their numeric values
user_access_data = df[['response_id', 'improved_user_access', 'improved_user_access_numeric']].copy()

# Count responses
response_counts = df['improved_user_access'].value_counts()
print("\nResponse Distribution:")
for response, count in response_counts.items():
    numeric_value = likert_mapping.get(response, 'Unknown')
    print(f"  '{response}' (={numeric_value}): {count} participants")

print(f"\n=== STEP 4: SAMPLE INDIVIDUAL RESPONSES ===")
sample_data = user_access_data.head(10)
for _, row in sample_data.iterrows():
    print(f"Response {row['response_id']}: '{row['improved_user_access']}' = {row['improved_user_access_numeric']}")

print(f"\n=== STEP 5: CALCULATION ===")
all_numeric_values = df['improved_user_access_numeric'].dropna()
total_responses = len(all_numeric_values)
sum_values = all_numeric_values.sum()
average = all_numeric_values.mean()

print(f"Sum of all numeric values: {sum_values}")
print(f"Number of valid responses: {total_responses}")
print(f"Average = {sum_values} ÷ {total_responses} = {average:.4f}")
print(f"Rounded to 2 decimals = {average:.2f}")

print(f"\n=== STEP 6: INTERPRETATION ===")
if average >= 4.5:
    interpretation = "EXCELLENT - Strong agreement with improvement"
elif average >= 4.0:
    interpretation = "GOOD - General agreement with improvement"
elif average >= 3.5:
    interpretation = "MODERATE - Mild agreement with improvement"
elif average >= 3.0:
    interpretation = "NEUTRAL - No strong opinion"
else:
    interpretation = "POOR - Disagreement with improvement"

print(f"Rating of {average:.2f} = {interpretation}")

print(f"\n=== STEP 7: DASHBOARD VALUE ===")
print(f"This {average:.2f} rating appears in the 'Learning Effectiveness Ratings' chart")
print(f"as 'Improved User Access Management: {average:.2f}'")

print(f"\n=== VERIFICATION WITH ALL VALUES ===")
print("All individual numeric responses:")
all_values = df['improved_user_access_numeric'].dropna().tolist()
print(all_values)
print(f"Manual calculation: sum({all_values}) / {len(all_values)} = {sum(all_values) / len(all_values):.4f}")