import pandas as pd
import numpy as np
from pathlib import Path

'''
KALO BELUM PUNYA DATASET ATAU MAU NGULANG 
DATASET AJA BARU JALANKAN KODE INI
'''

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset (1000 rows)
n = 10000

# Simulate flood height (in mm)
height = np.random.uniform(100, 500, n)

# Simulate rainfall (in mm/h)
rainfall = np.random.uniform(0, 50, n)

# Simulate receding speed (mm/min), could be negative if not receding
speed = np.random.normal(loc=-(height / 100) + 2, scale=1.5, size=n)

# Ensure speed doesn't go too extreme
speed = np.clip(speed, -5, 10)

# Calculate estimated time to recede based on current height and speed
# If speed <= 0 (not receding), we set time_to_recede as -1 (cannot be predicted)
time_to_recede = np.where(speed > 0, height / speed, -1)
time_to_recede = np.round(time_to_recede, 2)

# Create DataFrame
df = pd.DataFrame({
    "height_mm": np.round(height, 2),
    "rainfall_mm_per_h": np.round(rainfall, 2),
    "speed_mm_per_min": np.round(speed, 2),
    "time_to_recede_min": time_to_recede
})

# Save to CSV
csv_path = Path("AI3/datasets/flood_dataset.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False)

csv_path
