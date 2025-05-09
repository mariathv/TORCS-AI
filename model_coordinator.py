import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from sklearn.preprocessing import StandardScaler
import pickle
from collections import deque

def custom_mse():
    """Custom MSE function that can be properly serialized"""
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    return mse

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
        
        # Define feature counts for each model based on check_models.py output
        self.feature_counts = {
            'high_level': 23,
            'tactical': 15,
            'low_level': 19,
            'gear_selection': 13,
            'corner_handling': 13
        }
        
        # Keep sequence history for each feature set
        self.sequence_length = 10  # Based on model input shapes
        self.feature_history = {model_type: deque(maxlen=self.sequence_length) 
                                for model_type in self.model_types}
        
        # Load each model and its scaler
        for model_type in self.model_types:
            try:
                # Try different model file formats (h5 or SavedModel directory)
                model_path_h5 = os.path.join(models_dir, f'{model_type}_model.h5')
                model_path_dir = os.path.join(models_dir, f'{model_type}_model')
                
                if os.path.exists(model_path_h5):
                    # Load H5 format model
                    self.models[model_type] = keras.models.load_model(
                        model_path_h5, 
                        custom_objects={'mse': custom_mse()}
                    )
                    print(f"Loaded {model_type} model from H5 file")
                elif os.path.exists(model_path_dir) and os.path.isdir(model_path_dir):
                    # Load SavedModel format
                    self.models[model_type] = keras.models.load_model(
                        model_path_dir,
                        custom_objects={'mse': custom_mse()}
                    )
                    print(f"Loaded {model_type} model from SavedModel directory")
                else:
                    print(f"Warning: {model_type} model not found at {model_path_h5} or {model_path_dir}")
                
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
        
        print("Successfully loaded all models and scalers")
    
    def prepare_features_for_model(self, state_data, model_type):
        """Prepare features specifically for a given model type"""
        # Get required feature count for this model
        feature_count = self.feature_counts.get(model_type, 23)  # Default to high_level if unknown
        
        # Extract base features that are common to all models
        base_features = [
            state_data['angle'],
            state_data['trackPos'],
            state_data['speedX'],
            state_data['speedY'],
            state_data['speedZ'],
            state_data['rpm'],
            state_data['gear']
        ]
        
        # Extract track sensors
        track_sensors = state_data['track']
        track = track_sensors[:19]  # First 19 values are track sensors
        
        # Create model-specific feature vectors
        if model_type == 'high_level':
            # High level uses more track sensors, position data
            features = base_features + track[:18]  # 7 base + 18 track = 25, will trim to 23
            features = features[:feature_count]  # Trim to exact feature count needed
            
        elif model_type == 'tactical':
            # Tactical model focuses on immediate surroundings and speed
            features = base_features + track[:10]  # 7 base + 10 track = 17, will trim to 15
            features = features[:feature_count]
            
        elif model_type == 'low_level':
            # Low level needs more immediate control information
            features = base_features + track[:15]  # 7 base + 15 track = 22, will trim to 19
            features = features[:feature_count]
            
        elif model_type == 'gear_selection':
            # Gear selection mostly cares about speed, rpm, and some track data
            features = base_features + track[:8]  # 7 base + 8 track = 15, will trim to 13
            features = features[:feature_count]
            
        elif model_type == 'corner_handling':
            # Corner handling focuses on angle, speed, track position
            features = base_features + track[:8]  # 7 base + 8 track = 15, will trim to 13
            features = features[:feature_count]
        
        # Store the features in history for this model
        self.feature_history[model_type].append(features)
        
        # Only proceed if we have enough history
        if len(self.feature_history[model_type]) < self.sequence_length:
            # If not enough history, return None
            return None
        
        # Create sequence for LSTM input
        sequence = np.array(list(self.feature_history[model_type])).reshape(1, self.sequence_length, -1)
        
        return sequence
    
    def get_control_actions(self, state_data):
        """Get control actions from all models"""
        try:
            # Get predictions from each model
            predictions = {}
            
            # Process each model with its specific feature set
            for model_type in self.model_types:
                if model_type in self.models:
                    try:
                        # Get features specifically for this model
                        features = self.prepare_features_for_model(state_data, model_type)
                        
                        # Skip prediction if we don't have enough history yet
                        if features is None:
                            print(f"Not enough history for {model_type} model yet, skipping")
                            continue
                        
                        # Apply scaling if scaler is available
                        if model_type in self.scalers:
                            # For sequence data, we need to reshape, transform, and reshape back
                            original_shape = features.shape
                            features_flat = features.reshape(-1, features.shape[-1])
                            features_scaled = self.scalers[model_type].transform(features_flat)
                            features = features_scaled.reshape(original_shape)
                        
                        # Make prediction
                        prediction = self.models[model_type].predict(features, verbose=0)
                        
                        # Store the prediction
                        predictions[model_type] = prediction[0]
                        
                    except Exception as e:
                        print(f"Error predicting with {model_type} model: {e}")
            
            # If we got any predictions, combine them
            if predictions:
                controls = self.combine_predictions(predictions, state_data)
                return controls
            else:
                print("No model predictions available, using default controls")
                return {
                    'steer': 0.0,
                    'accel': 0.5,
                    'brake': 0.0
                }
            
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
            high_level_pred = predictions['high_level']
            if isinstance(high_level_pred, np.ndarray) and high_level_pred.size >= 1:
                controls['steer'] = float(high_level_pred[0]) * 0.3  # 30% weight
        
        if 'tactical' in predictions:
            # Tactical model provides immediate decisions
            tactical_pred = predictions['tactical']
            if isinstance(tactical_pred, np.ndarray) and tactical_pred.size >= 1:
                controls['accel'] = float(tactical_pred[0]) * 0.3  # 30% weight
        
        if 'low_level' in predictions:
            # Low-level model provides precise control
            low_level_pred = predictions['low_level']
            if isinstance(low_level_pred, np.ndarray) and low_level_pred.size >= 3:
                controls['steer'] += float(low_level_pred[0]) * 0.2  # 20% weight
                controls['accel'] += float(low_level_pred[1]) * 0.2
                controls['brake'] += float(low_level_pred[2]) * 0.2
        
        if 'corner_handling' in predictions:
            # Corner handling model provides specialized control
            corner_pred = predictions['corner_handling']
            if isinstance(corner_pred, np.ndarray) and corner_pred.size >= 2:
                controls['steer'] += float(corner_pred[0]) * 0.2  # 20% weight
                controls['brake'] += float(corner_pred[1]) * 0.2
        
        # Ensure controls are within valid ranges
        controls['steer'] = max(-1.0, min(1.0, controls['steer']))
        controls['accel'] = max(0.0, min(1.0, controls['accel']))
        controls['brake'] = max(0.0, min(1.0, controls['brake']))
        
        # Output control values for debugging
        print(f"Final controls: steer={controls['steer']:.2f}, accel={controls['accel']:.2f}, brake={controls['brake']:.2f}")
        
        return controls 