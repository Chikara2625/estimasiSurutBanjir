from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.metrics import accuracy_score


import joblib
import pandas as pd
from pathlib import Path

# Load the dataset
# df = pd.read_csv("AI3/datasets/flood_dataset2.csv")
df = pd.read_csv("AI3/datasets/flood_dataset500data.csv")

# Filter out invalid training data (where water is not receding)
train_df = df[df["time_to_recede_min"] > 0]

# Features and target
X = train_df[["height_mm", "rainfall_mm_per_h", "speed_mm_per_min"]]
y = train_df["time_to_recede_min"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate
y_pred = model_rf.predict(X_test)
# acc = accuracy_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
'''
Rata-rata kesalahan kuadrat dari prediksi. Karena nilai ini menggunakan kuadrat dari error, maka kesalahan besar dihukum lebih berat.

✳️ Biasanya digunakan untuk membandingkan antar model (makin kecil makin baik).
📌 Tapi nilainya sulit diinterpretasikan langsung karena satuannya menit².
'''



r2 = r2_score(y_test, y_pred)
'''
Model menjelaskan sekitar 94.91% dari variansi data asli.

✅ Ini sangat bagus, karena:

Skor R² berkisar dari 0 hingga 1 (kadang bisa negatif jika model jelek).

Nilai > 0.9 berarti model punya akurasi sangat tinggi dalam memprediksi target.
'''


mae = mean_absolute_error(y_test, y_pred)
'''
Secara rata-rata, prediksi model kamu meleset sebesar 16.63 menit dari nilai sebenarnya (time_to_recede_min).

✅ Baik jika dibandingkan dengan nilai-nilai target (y) yang mungkin berkisar dari puluhan hingga ratusan menit.
'''



# Save model
model_path = Path("AI3/models/aiModel1.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_rf, model_path)

print(f"MARGIN OF ERROR: {mae}\nMEAN SQUARED ERRROR:{mse}\nR2:{r2}\nPATH: {model_path}")
