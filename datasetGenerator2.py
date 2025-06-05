import pandas as pd
import numpy as np
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Feature generation based on realistic assumptions
# Curah hujan (mm/jam): 0 - 100 (heavy rainfall range)
curah_hujan = np.clip(np.random.normal(loc=30, scale=15, size=n_samples), 0, 100)

# Tinggi saat ini (cm): 10 - 500 (from shallow to deep flood level)
tinggi_saat_ini = np.clip(np.random.normal(loc=200, scale=100, size=n_samples), 10, 500)

# Kecepatan surut (mm/menit): 1 - 20 (slow to fast drainage rate)
kecepatan_surut = np.clip(np.random.normal(loc=10, scale=4, size=n_samples), 1, 20)

# Estimasi waktu surut (menit), target:
# Asumsi sederhana:
# waktu = tinggi_saat_ini / kecepatan_surut + tambahan waktu karena curah hujan tinggi
estimasi_waktu = (tinggi_saat_ini * 10 / kecepatan_surut) + (curah_hujan * 1.5)
estimasi_waktu = np.round(estimasi_waktu, 2)

# Create DataFrame
df = pd.DataFrame({
    "height_mm": np.round(tinggi_saat_ini, 2),
    "rainfall_mm_per_h": np.round(curah_hujan, 2),
    "speed_mm_per_min": np.round(kecepatan_surut, 2),
    "time_to_recede_min": estimasi_waktu
})

# Save to CSV
csv_path = Path("AI3/datasets/flood_dataset500data.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False)

print( csv_path)
