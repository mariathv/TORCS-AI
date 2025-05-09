import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
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
        
        # Define target variables for each model - all models now predict control actions
        self.target_groups = {
            'high_level': ['steer', 'accel', 'brake'],  # Strategic control
            'tactical': ['steer', 'accel', 'brake'],    # Tactical control
            'low_level': ['steer', 'accel', 'brake'],   # Precise control
            'gear_selection': ['gear'],                 # Gear selection
            'corner_handling': ['steer', 'accel', 'brake']  # Corner-specific control
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
            # Pre-allocate arrays for better performance
            X = data[features].values
            y = data[self.target_groups[model_name]].values
            
            n_samples = len(X) - sequence_length
            X_seq = np.zeros((n_samples, sequence_length, X.shape[1]))
            y_seq = np.zeros((n_samples, y.shape[1]))
            
            # Use vectorized operations
            for i in range(n_samples):
                X_seq[i] = X[i:(i + sequence_length)]
                y_seq[i] = y[i + sequence_length]
            
            sequences[model_name] = {
                'X': X_seq,
                'y': y_seq
            }
        
        return sequences
    
    def scale_data(self, sequences):
        """Scale the data using StandardScaler"""
        scaled_sequences = {}
        
        for model_name, sequence_data in sequences.items():
            # Reshape for scaling
            X_reshaped = sequence_data['X'].reshape(-1, sequence_data['X'].shape[-1])
            
            # Scale features
            feature_scaler = StandardScaler()
            X_scaled = feature_scaler.fit_transform(X_reshaped)
            
            # Reshape back
            X_scaled = X_scaled.reshape(sequence_data['X'].shape)
            
            # Scale targets
            target_scaler = StandardScaler()
            y_scaled = target_scaler.fit_transform(sequence_data['y'])
            
            # Save scalers
            self.scalers[model_name] = {
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler
            }
            
            scaled_sequences[model_name] = {
                'X': X_scaled,
                'y': y_scaled
            }
        
        return scaled_sequences
    
    def save_scalers(self):
        """Save the scalers for later use"""
        for model_name, scalers in self.scalers.items():
            joblib.dump(
                scalers['feature_scaler'],
                os.path.join(self.scaler_dir, f'{model_name}_feature_scaler.pkl')
            )
            joblib.dump(
                scalers['target_scaler'],
                os.path.join(self.scaler_dir, f'{model_name}_target_scaler.pkl')
            )
    
    def prepare_training_data(self, filename='telemetry.csv', sequence_length=10):
        """Prepare all data for training"""
        # Load and preprocess data
        print("Loading data...")
        data = self.load_data(filename)
        
        print("Preprocessing data...")
        data = self.preprocess_data(data)
        
        print("Creating sequences...")
        sequences = self.create_sequences(data, sequence_length)
        
        print("Scaling data...")
        scaled_sequences = self.scale_data(sequences)
        
        print("Saving scalers...")
        self.save_scalers()
        
        print("Splitting data...")
        # Split data into training and validation sets
        train_val_splits = {}
        for model_name, sequence_data in scaled_sequences.items():
            X_train, X_val, y_train, y_val = train_test_split(
                sequence_data['X'],
                sequence_data['y'],
                test_size=0.2,
                random_state=42,
                shuffle=True  # Enable shuffling for better training
            )
            
            train_val_splits[model_name] = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val
            }
        
        return train_val_splits 