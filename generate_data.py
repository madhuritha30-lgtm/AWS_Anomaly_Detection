import pandas as pd
import numpy as np

# Make the generated data reproducible
np.random.seed(42)

# Number of weather records
rows = 1000

# Create hourly timestamps
timestamps = pd.date_range(
    start="2026-01-01",
    periods=rows,
    freq="h"
)

# Generate normal weather readings
temperature = np.random.normal(30, 3, rows)
humidity = np.random.normal(65, 10, rows)
pressure = np.random.normal(1010, 5, rows)
wind_speed = np.random.normal(15, 5, rows)
rainfall = np.random.exponential(1, rows)

# Create the dataset
data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure,
    "wind_speed": wind_speed,
    "rainfall": rainfall
})

# Add abnormal readings for testing
anomaly_indices = [100, 250, 400, 550, 700, 850]

data.loc[anomaly_indices, "temperature"] = [55, 5, 60, 2, 58, 0]
data.loc[anomaly_indices, "humidity"] = [15, 98, 10, 95, 12, 99]
data.loc[anomaly_indices, "pressure"] = [950, 1060, 945, 1070, 948, 1065]
data.loc[anomaly_indices, "wind_speed"] = [80, 2, 90, 1, 85, 3]
data.loc[anomaly_indices, "rainfall"] = [0, 80, 0, 100, 0, 70]

# Save as CSV
data.to_csv("aws_weather_data.csv", index=False)

print("AWS weather dataset created successfully!")
print()
print("Number of records:", len(data))
print()
print(data.head())