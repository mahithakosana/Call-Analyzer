# train_model.py
# RUN THIS SCRIPT FIRST to create the model file.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Generate synthetic training data (Replace this with loading your real data from a CSV)
print("Generating training data...")
np.random.seed(42)
n_samples = 5000

data = {
    'latency': np.random.normal(50, 20, n_samples),
    'jitter': np.random.normal(10, 5, n_samples),
    'packet_loss': np.random.uniform(0, 5, n_samples)
}

# Synthesize a QoS score (lower metrics = higher score)
data['qos_score'] = (100 - 
                    0.5 * np.clip(data['latency'], 0, 100) - 
                    0.3 * np.clip(data['jitter'], 0, 20) * 5 - 
                    0.2 * data['packet_loss'] * 10 +
                    np.random.normal(0, 5, n_samples)
                    )
df_train = pd.DataFrame(data)
df_train['qos_score'] = np.clip(df_train['qos_score'], 0, 100) # Keep score between 0-100

print("Data sample:")
print(df_train.head())

# 2. Train the Model
X = df_train[['latency', 'jitter', 'packet_loss']]
y = df_train['qos_score']

model = RandomForestRegressor(n_estimators=100, random_state=42)
print("\nTraining the Random Forest model...")
model.fit(X, y)

# 3. Save the Model
model_filename = 'qos_model.pkl'
joblib.dump(model, model_filename)
print(f"\n✅ Model successfully trained and saved as '{model_filename}'")
print("You can now run the Streamlit dashboard (qos_dashboard.py).")