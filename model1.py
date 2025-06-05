from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
from pathlib import Path

# Load the dataset
# df = pd.read_csv("AI3/datasets/flood_dataset2.csv")
df = pd.read_csv("datasets/flood_dataset500data.csv")

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
mae = mean_absolute_error(y_test, y_pred)

# Save model
model_path = Path("models/aiModel1.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_rf, model_path)

print(f"MARGIN OF ERROR: {mae}\nPATH: {model_path}")
