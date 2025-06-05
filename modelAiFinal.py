import numpy as np
import joblib
import pandas as pd
import math

# Simulate real-time input with rolling data to calculate speed from previous 3 minutes
def calculate_speed_from_history(heights, timestamps):
    # heights: list of last N heights (mm)
    # timestamps: list of same length, in minutes
    if len(heights) < 2:
        return None  # not enough data
    diffs = np.diff(heights)
    times = np.diff(timestamps)
    speeds = -diffs / times  # negative if decreasing height
    avg_speed = np.mean(speeds)
    return round(avg_speed, 2)

# Simulate current live data + last 3 minutes
# New data: height = 250, rainfall = 12, current time = minute 8
current_data = {
    "height_mm": 50,

    #Curah hujan biasanya perjam
    "rainfall_mm_per_h": 12, #CARI TAHU MENGENAI INI
    "timestamps": [5, 6, 7, 8],  # minutes

    

    #jika banjir TURUN TERUS
    #CASE 1
    # "heights": [275, 265, 258, 250] #mm
    
    #CASE 2
    # "heights" : [155, 153, 152, 150] #mm
    
    #CASE 3
    "heights" : [50, 40, 30, 20] #mm


    #dalam mm jika banjir malah NAIK
    # "heights" : [150, 155, 153, 160]
}

# Calculate speed from history
calculated_speed = calculate_speed_from_history(current_data["heights"], current_data["timestamps"])

# Load model from previous training
model_rf_loaded = joblib.load("models/aiModel1.pkl")

# Make prediction or handle if not receding
if calculated_speed is not None and calculated_speed > 0:
    input_features = pd.DataFrame([[
        current_data["height_mm"],
        current_data["rainfall_mm_per_h"],
        calculated_speed
    ]], columns=["height_mm", "rainfall_mm_per_h", "speed_mm_per_min"])
    
    predicted_time = model_rf_loaded.predict(input_features)[0]
    predicted_minutes = math.ceil(predicted_time)  # Bulatkan ke atas
    
    # Ubah ke format jam-menit
    hours = predicted_minutes // 60
    minutes = predicted_minutes % 60
    
    if hours > 0:
        result_message = f"⏳ Prediksi waktu surut: {hours} jam {minutes} menit"
    else:
        result_message = f"⏳ Prediksi waktu surut: {minutes} menit"
else:
    result_message = "⚠️ Air belum surut. Tidak bisa prediksi waktu surut sekarang."

print(result_message)
