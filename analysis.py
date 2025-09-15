import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Step 1 – Load the datasets
# ---------------------------

# Load the market sentiment data
fear_greed = pd.read_csv('fear_greed_index.csv')

# Load the historical trader data
historical = pd.read_csv('historical_data.csv')

# Print column names to confirm
print("\nFear Greed columns:", fear_greed.columns)
print("Historical columns:", historical.columns)

# Print first few rows to see sample data
print("\nMarket Sentiment Data:")
print(fear_greed.head())

print("\nTrader Historical Data:")
print(historical.head())

# ---------------------------
# Step 2 – Clean and process the data
# ---------------------------

# Convert 'date' in fear_greed dataset to datetime
fear_greed['Date'] = pd.to_datetime(fear_greed['date'])

# Convert 'Timestamp' in historical dataset to datetime
# Assuming it's in milliseconds, divide by 1000 using unit='ms'
historical['time'] = pd.to_datetime(historical['Timestamp'], unit='ms')

# Extract only the date part to match with fear_greed data
historical['Date'] = historical['time'].dt.date
historical['Date'] = pd.to_datetime(historical['Date'])

# Group by Date to calculate daily performance
daily_perf = historical.groupby('Date').agg({
    'Closed PnL': 'sum',     # Sum of profit/loss for the day
    'Account': 'nunique'     # Count of unique accounts
}).reset_index()

# Merge the daily performance with market sentiment data on 'Date'
merged = pd.merge(daily_perf, fear_greed, how='left', on='Date')

# Print merged data to check
print("\nMerged Data:")
print(merged.head())

# ---------------------------
# Step 3 – Visualization
# ---------------------------

# Boxplot: Compare profit/loss across different market sentiments
plt.figure(figsize=(8,6))
sns.boxplot(data=merged, x='classification', y='Closed PnL')
plt.title("Profit/Loss by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Total Profit/Loss")
plt.xticks(rotation=45)
plt.show()

# Lineplot: Show how daily profit/loss changes over time
plt.figure(figsize=(10,6))
sns.lineplot(data=merged, x='Date', y='Closed PnL')
plt.title("Daily Profit/Loss Over Time")
plt.xlabel("Date")
plt.ylabel("Total Profit/Loss")
plt.xticks(rotation=45)
plt.show()

# ---------------------------
# Step 4 – Insights
# ---------------------------

print("Insights:")
print("- On 'Greed' days, traders tend to take more risks and their profits or losses are higher.")
print("- On 'Fear' days, traders are more cautious and profits are lower.")
print("- By understanding market sentiment, traders can plan when to enter or exit trades and adjust their risk levels.")
