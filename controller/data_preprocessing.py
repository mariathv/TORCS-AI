import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_file, return_scaler=False):
    # ---- load telemetry data
    print("Loading data...")
    data = pd.read_csv(csv_file)
    print(f"Data shape: {data.shape}")

    # --- define input (sensors) and output (actions) columns - REDUCED FEATURE SET
    input_cols = [
        'speedX', 'speedY', 'speedZ',       # Speed components (3)
        'angle', 'trackPos',                # Position on track (2)
        'rpm', 'gear'                       # Engine state (2)
    ]
    
    # Add more track sensors for better coverage
    # These correspond to sensors in front, sides, and rear of the car
    track_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # Full 180-degree coverage
    
    # Try both naming conventions for track sensors
    track_cols_format1 = [f'track_{i}' for i in track_indices]  # track_0, track_1, etc.
    track_cols_format2 = [f'track[{i}]' for i in track_indices]  # track[0], track[1], etc.
    
    # Check which format exists in the dataset
    if any(col in data.columns for col in track_cols_format1):
        track_cols = [col for col in track_cols_format1 if col in data.columns]
    elif any(col in data.columns for col in track_cols_format2):
        track_cols = [col for col in track_cols_format2 if col in data.columns]
    else:
        print("WARNING: Track sensor columns not found in expected format.")
        track_cols = []
    
    # Add available track columns to inputs
    input_cols.extend(track_cols)
    
    # Output columns remain the same
    output_cols = ['steer', 'brake', 'accel']

    # Check if columns exist in the dataset
    for col in input_cols + output_cols:
        if col not in data.columns:
            print(f"Warning: Column {col} not found in dataset.")

    # Filter to only include columns that exist in the dataset
    input_cols = [col for col in input_cols if col in data.columns]
    output_cols = [col for col in output_cols if col in data.columns]

    print(f"Using REDUCED input columns ({len(input_cols)} features): {input_cols}")
    print(f"Using output columns: {output_cols}")

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

    print("Preprocessing done with REDUCED feature set (12 features).")

    if return_scaler:
        return X_train, X_test, y_train, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test

