import pandas as pd

# Load the CSV file into a DataFrame
input_file = 'new_rmppo_running_deviation.csv'
df = pd.read_csv(input_file)

# Define the threshold magnitude
threshold = 0.06

# Filter out rows where the magnitude is less than the threshold
df_filtered = df[df['Magnitude'] >= threshold]

# Save the filtered DataFrame to a new CSV file
output_file = 'new_vanilla_running_deviation.csv'
df_filtered.to_csv(output_file, index=False)

print(f"Rows with magnitude less than {threshold} have been removed. Filtered data saved to {output_file}.")