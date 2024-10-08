import pandas as pd

# Load your dataset
df = pd.read_csv('/home/berend/Code/inno-ml-orchestration/ml-models/journal_2/_data/MS_11349_MS_11349_POD_0_utilization.csv')

# Convert timestamp to datetime (assuming it's in milliseconds)
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

# Drop the original timestamp column
df = df.drop('timestamp', axis=1)

# Define the desired column order
column_order = ['date', 'cpu_utilization', 'memory_utilization']

# Reorder the dataframe columns
df = df[column_order]

# Verify the column order
print("Column order:", df.columns.tolist())

# Save the processed data
df.to_csv('processed_data.csv', index=False)

# Verify the saved data
saved_df = pd.read_csv('processed_data.csv')
print("Saved data column order:", saved_df.columns.tolist())

# Display the first few rows of the saved data
print(saved_df.head())