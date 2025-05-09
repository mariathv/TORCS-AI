import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class TelemetryDataProcessor:
    """
    Handles preprocessing of telemetry data for multi-model training
    """
    
    def __init__(self, data_dir='telemetry_logs', scaler_dir='model_scalers'):
        self.data_dir = data_dir
        self.scaler_dir = scaler_dir
        os.makedirs(scaler_dir, exist_ok=True)
        
        # Define feature groups for different models
        self.feature_groups = {
            'high_level': [
                'trackPos', 'racePos', 'distRaced', 'distFromStart',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
                'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
                'track_15', 'track_16', 'track_17', 'track_18'
            ],
            'tactical': [
                'speedX', 'speedY', 'speedZ', 'angle', 'trackPos',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9'
            ],
            'low_level': [
                'speedX', 'speedY', 'speedZ', 'angle', 'rpm',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
                'wheelSpinVel_0', 'wheelSpinVel_1', 'wheelSpinVel_2', 'wheelSpinVel_3'
            ],
            'gear_selection': [
                'speedX', 'rpm', 'gear', 'track_0', 'track_1', 'track_2',
                'track_3', 'track_4', 'track_5', 'track_6', 'track_7',
                'track_8', 'track_9'
            ],
            'corner_handling': [
                'speedX', 'angle', 'trackPos',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9'
            ]
        }
        
        # Define target variables for each model
        self.target_groups = {
            'high_level': ['trackPos'],  # Target racing line
            'tactical': ['speedX'],      # Target speed
            'low_level': ['accel', 'brake', 'steer', 'clutch'],
            'gear_selection': ['gear'],
            'corner_handling': ['steer', 'brake']
        }
        
        self.scalers = {}
    
    def load_data(self, filename='telemetry.csv'):
        """Load telemetry data from CSV file"""
        file_path = os.path.join(self.data_dir, filename)
        return pd.read_csv(file_path)
    
    def preprocess_data(self, data):
        """Basic preprocessing of the data"""
        # Remove any rows with missing values
        data = data.dropna()
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        return data
    
    def create_sequences(self, data, sequence_length=10):
        """Create sequences for time series data"""
        sequences = {}
        
        for model_name, features in self.feature_groups.items():
            X = data[features].values
            y = data[self.target_groups[model_name]].values
            
            X_seq, y_seq = [], []
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:(i + sequence_length)])
                y_seq.append(y[i + sequence_length])
            
            sequences[model_name] = {
                'X': np.array(X_seq),
                'y': np.array(y_seq)
            }
        
        return sequences
    
    def scale_data(self, sequences):
        """Scale the data using StandardScaler"""
        scaled_sequences = {}
        
        for model_name, sequence_data in sequences.items():
            # Create a single scaler for both features and targets
            scaler = StandardScaler()
            
            # Flatten sequence data for fitting the scaler
            X_flat = sequence_data['X'].reshape(-1, sequence_data['X'].shape[-1])
            scaler.fit(X_flat)
            
            # Scale features
            X_scaled = scaler.transform(X_flat).reshape(sequence_data['X'].shape)
            
            # Scale targets using the same scaler (optional)
            # If you want separate scaling for targets, you can use a different scaler
            # In this case, we're using the same scaler for simplicity
            y_scaled = sequence_data['y']  # Not scaling targets
            
            # Save scaler
            self.scalers[model_name] = scaler
            
            scaled_sequences[model_name] = {
                'X': X_scaled,
                'y': y_scaled
            }
        
        return scaled_sequences
    
    def save_scalers(self):
        """Save the scalers for later use"""
        for model_name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.scaler_dir, f'{model_name}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler for {model_name}")
    
    def prepare_training_data(self, filename='telemetry.csv', sequence_length=10):
        """Prepare all data for training"""
        # Load and preprocess data
        data = self.load_data(filename)
        data = self.preprocess_data(data)
        
        # Create sequences
        sequences = self.create_sequences(data, sequence_length)
        
        # Scale data
        scaled_sequences = self.scale_data(sequences)
        
        # Save scalers
        self.save_scalers()
        
        # Split data into training and validation sets
        train_val_splits = {}
        for model_name, sequence_data in scaled_sequences.items():
            X_train, X_val, y_train, y_val = train_test_split(
                sequence_data['X'],
                sequence_data['y'],
                test_size=0.2,
                random_state=42
            )
            
            train_val_splits[model_name] = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val
            }
        
        return train_val_splits 