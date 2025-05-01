import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_file):
    # ---- load telemetry data
    print("Loading data...")
    data = pd.read_csv(csv_file)
    print(f"Data shape: {data.shape}")

    # --- define input (sensors) and output (actions) columns
    input_cols = ['speedX', 'speedY', 'speedZ', 'angle', 'trackPos', 'accel']
    output_cols = ['steer', 'brake', 'accel']

    # --- inputs and outputs
    X = data[input_cols].values
    y = data[output_cols].values

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Splitting done.")

    # ---Normalize inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Preprocessing done.")

    return X_train, X_test, y_train, y_test

