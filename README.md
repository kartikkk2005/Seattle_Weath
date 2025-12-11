# Step 1: Import necessary libraries
import pandas as pd
import numpy as np

# ----
# Step 2: Load the dataset
# ---
# Make sure the CSV file is in the same directory as your script
try:
    df = pd.read_csv('seattleWeather_1948-2017.csv')
except FileNotFoundError:
    print("Error: 'seattleWeather_1948-2017.csv' not found.")
    print("Please download it from Kaggle and place it in the correct directory.")
    exit()

# ---
# Step 3: Initial Data Exploration
# ---
print("--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information (columns, data types, non-null values):")
df.info()

# ---
# Step 4: Data Cleaning and Preparation (Updated)
# ---
print("\n--- Data Cleaning and Preparation ---")

# Handle missing data
# First, calculate the means
tmax_mean = df['TMAX'].mean()
tmin_mean = df['TMIN'].mean()

# Now, fill the missing values and assign the result back to the columns
df['PRCP'] = df['PRCP'].fillna(0)
df['TMAX'] = df['TMAX'].fillna(tmax_mean)
df['TMIN'] = df['TMIN'].fillna(tmin_mean)

print("\nMissing values handled in a future-proof way.")

# Convert DATE column to a proper datetime object
df['DATE'] = pd.to_datetime(df['DATE'])
print("Converted 'DATE' column to datetime objects.")

# ---
# Step 5: Analysis and Calculations
# ---
print("\n--- Analysis and Calculations ---")

# 1. Find the average temperature for each month of a specific year (e.g., 2012)
print("\n1. Average temperature analysis for the year 2012:")
df_2012 = df[df['DATE'].dt.year == 2012].copy() # Use .copy() to avoid SettingWithCopyWarning
df_2012['MONTH'] = df_2012['DATE'].dt.month
# Calculate average temperature as (max + min) / 2
df_2012['AVG_TEMP'] = (df_2012['TMAX'] + df_2012['TMIN']) / 2
monthly_avg_temp = df_2012.groupby('MONTH')['AVG_TEMP'].mean()
print("Average temperature per month in 2012:")
print(monthly_avg_temp.round(2))

# 2. Identify the day with the highest rainfall in the entire dataset
print("\n2. Day with the highest rainfall:")
# Find the index of the row with the maximum precipitation
max_prcp_idx = df['PRCP'].idxmax()
# Use .loc to retrieve the full row data
day_with_highest_rainfall = df.loc[max_prcp_idx]
print(f"The day with the highest rainfall was {day_with_highest_rainfall['DATE'].date()} with {day_with_highest_rainfall['PRCP']} inches of rain.")

# 3. Determine the number of sunny days in a year (e.g., 2015)
print("\n3. Number of different weather types in 2015:")
df_2015 = df[df['DATE'].dt.year == 2015]
# Count the occurrences of each unique value in the 'RAIN' column
# This dataset uses a boolean column 'RAIN' (True/False). Let's count non-rainy days.
# A 'False' in the RAIN column can be our proxy for a "sunny" or at least "dry" day.
weather_counts_2015 = df_2015['RAIN'].value_counts()
# The value_counts() returns a Series. We can access the count for 'False'
num_dry_days = weather_counts_2015.get(False, 0) # .get(False, 0) safely gets the count, returns 0 if no False values exist
print(f"In 2015, there were {num_dry_days} days without rain ('dry days').")

# ---
# Step 6: Using NumPy for a calculation
# ---
print("\n--- NumPy Integration ---")
# Let's convert the TMAX column for 2012 into a NumPy array
tmax_2012_numpy = df_2012['TMAX'].to_numpy()

# Now, use a vectorized NumPy operation to convert these temperatures to Celsius
# Formula: C = (F - 32) * 5/9
tmax_celsius_2012 = (tmax_2012_numpy - 32) * 5/9

# Let's add this back to our 2012 DataFrame
df_2012['TMAX_CELSIUS'] = tmax_celsius_2012.round(2)
print("Demonstration: Converted 2012 max temperatures from Fahrenheit to Celsius using NumPy.")
print("First 5 rows of the 2012 data with the new Celsius column:")
print(df_2012[['DATE', 'TMAX', 'TMAX_CELSIUS']].head())
