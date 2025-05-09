import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from sklearn.preprocessing import StandardScaler
import pickle

class ModelCoordinator:
    def __init__(self, models_dir='models', scalers_dir='model_scalers'):
        """Initialize the model coordinator with trained models and scalers"""
        self.models_dir = models_dir
        self.scalers_dir = scalers_dir
        
        # Load all models
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        
        # Model types we have
        self.model_types = [
            'high_level',
            'tactical',
            'low_level',
            'gear_selection',
            'corner_handling'
        ]
        
        # Load each model and its scaler
        for model_type in self.model_types:
            try:
                # Load model
                model_path = os.path.join(models_dir, f'{model_type}_model.h5')
                if os.path.exists(model_path):
                    self.models[model_type] = keras.models.load_model(model_path)
                    print(f"Loaded {model_type} model")
                else:
                    print(f"Warning: {model_type} model not found at {model_path}")
                
                # Load scaler
                scaler_path = os.path.join(scalers_dir, f'{model_type}_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_type] = pickle.load(f)
                    print(f"Loaded {model_type} scaler")
                else:
                    print(f"Warning: {model_type} scaler not found at {scaler_path}")
                
                # Load model config
                config_path = os.path.join('model_info', f'{model_type}_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.model_configs[model_type] = json.load(f)
                    print(f"Loaded {model_type} config")
                
            except Exception as e:
                print(f"Error loading {model_type} model/scaler: {e}")
    
    def prepare_state_data(self, state_data):
        """Prepare state data for model input"""
        # Extract track sensors
        track_sensors = state_data['track']
        
        # Track sensors array contains both track and track edge information
        # First 19 values are track sensors, next 19 are track edge sensors
        track = track_sensors[:19]
        track_edge = track_sensors[19:38] if len(track_sensors) >= 38 else [0] * 19
        
        # Check if advanced attributes exist and provide defaults if not
        roll = state_data.get('roll', 0.0)
        pitch = state_data.get('pitch', 0.0)
        yaw = state_data.get('yaw', 0.0)
        speedGlobalX = state_data.get('speedGlobalX', 0.0)
        speedGlobalY = state_data.get('speedGlobalY', 0.0)
        speedGlobalZ = state_data.get('speedGlobalZ', 0.0)
        
        # Convert state data to numpy array
        features = np.array([
            state_data['angle'],
            state_data['trackPos'],
            state_data['speedX'],
            state_data['speedY'],
            state_data['speedZ'],
            state_data['rpm'],
            state_data['gear'],
            *track,  # Unpack track sensors
            *track_edge,  # Unpack track edge sensors
            *state_data['focus'],  # Unpack focus sensors
            state_data['fuel'],
            state_data['distRaced'],
            state_data['distFromStart'],
            state_data['racePos'],
            state_data['z'],
            roll,
            pitch,
            yaw,
            speedGlobalX,
            speedGlobalY,
            speedGlobalZ
        ]).reshape(1, -1)
        
        return features
    
    def get_control_actions(self, state_data):
        """Get control actions from all models"""
        try:
            # Prepare state data
            features = self.prepare_state_data(state_data)
            
            # Get predictions from each model
            predictions = {}
            
            # High-level model for overall strategy
            if 'high_level' in self.models:
                if 'high_level' in self.scalers:
                    features_scaled = self.scalers['high_level'].transform(features)
                else:
                    features_scaled = features
                predictions['high_level'] = self.models['high_level'].predict(features_scaled, verbose=0)[0]
            
            # Tactical model for immediate decisions
            if 'tactical' in self.models:
                if 'tactical' in self.scalers:
                    features_scaled = self.scalers['tactical'].transform(features)
                else:
                    features_scaled = features
                predictions['tactical'] = self.models['tactical'].predict(features_scaled, verbose=0)[0]
            
            # Low-level model for precise control
            if 'low_level' in self.models:
                if 'low_level' in self.scalers:
                    features_scaled = self.scalers['low_level'].transform(features)
                else:
                    features_scaled = features
                predictions['low_level'] = self.models['low_level'].predict(features_scaled, verbose=0)[0]
            
            # Corner handling model
            if 'corner_handling' in self.models:
                if 'corner_handling' in self.scalers:
                    features_scaled = self.scalers['corner_handling'].transform(features)
                else:
                    features_scaled = features
                predictions['corner_handling'] = self.models['corner_handling'].predict(features_scaled, verbose=0)[0]
            
            # Combine predictions
            controls = self.combine_predictions(predictions, state_data)
            
            return controls
            
        except Exception as e:
            print(f"Error getting control actions: {e}")
            # Return safe default controls
            return {
                'steer': 0.0,
                'accel': 0.5,
                'brake': 0.0
            }
    
    def combine_predictions(self, predictions, state_data):
        """Combine predictions from different models"""
        # Initialize default controls
        controls = {
            'steer': 0.0,
            'accel': 0.5,
            'brake': 0.0
        }
        
        # Combine predictions based on model type
        if 'high_level' in predictions:
            # High-level model provides overall strategy
            controls['steer'] = predictions['high_level'][0] * 0.3  # 30% weight
            controls['accel'] = predictions['high_level'][1] * 0.3
            controls['brake'] = predictions['high_level'][2] * 0.3
        
        if 'tactical' in predictions:
            # Tactical model provides immediate decisions
            controls['steer'] += predictions['tactical'][0] * 0.3  # 30% weight
            controls['accel'] += predictions['tactical'][1] * 0.3
            controls['brake'] += predictions['tactical'][2] * 0.3
        
        if 'low_level' in predictions:
            # Low-level model provides precise control
            controls['steer'] += predictions['low_level'][0] * 0.2  # 20% weight
            controls['accel'] += predictions['low_level'][1] * 0.2
            controls['brake'] += predictions['low_level'][2] * 0.2
        
        if 'corner_handling' in predictions:
            # Corner handling model provides specialized control
            controls['steer'] += predictions['corner_handling'][0] * 0.2  # 20% weight
            controls['accel'] += predictions['corner_handling'][1] * 0.2
            controls['brake'] += predictions['corner_handling'][2] * 0.2
        
        # Ensure controls are within valid ranges
        controls['steer'] = max(-1.0, min(1.0, controls['steer']))
        controls['accel'] = max(0.0, min(1.0, controls['accel']))
        controls['brake'] = max(0.0, min(1.0, controls['brake']))
        
        return controls 