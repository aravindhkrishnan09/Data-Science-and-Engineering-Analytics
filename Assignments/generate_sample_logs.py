import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Directory to save the log files
log_dir = "sample_logs"
os.makedirs(log_dir, exist_ok=True)

# Function to generate random timestamps
def generate_timestamps(start, end, n):
    start_time = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    return [start_time + (end_time - start_time) * np.random.random() for _ in range(n)]

# Generate 10 sample log files
for i in range(1, 11):
    num_rows = np.random.randint(50, 100)  # Random number of rows per file
    data = {
        "timestamp": generate_timestamps("2025-01-01 00:00:00", "2025-01-10 23:59:59", num_rows),
        "sensor_id": np.random.randint(1, 10, size=num_rows),
        "value": np.random.uniform(10.0, 100.0, size=num_rows),
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(log_dir, f"log_file_{i}.csv")
    df.to_csv(file_path, index=False)
    print(f"Generated: {file_path}")

print(f"All log files have been generated in the '{log_dir}' directory.")