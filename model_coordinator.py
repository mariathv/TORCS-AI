import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import joblib
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging at module level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCoordinator:
    def __init__(self, models_dir: str = 'models', scalers_dir: str = 'model_scalers'):
        """Initialize the model coordinator with trained models and scalers"""
        # Configure logging first
        self.logger = logging.getLogger(__name__)
        
        self.model_dir = models_dir
        self.scaler_dir = scalers_dir
        self.models = {}
        self.scalers = {}
        self.configs = {}
        self.state_history = {
            'high_level': [],
            'tactical': [],
            'low_level': [],
            'corner_handling': []
        }
        self.sequence_length = 10
        self.last_controls = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0}
        self.control_smoothing = 0.3  # Smoothing factor for control transitions
        
        # Model types we have
        self.model_types = [
            'high_level',
            'tactical',
            'low_level',
            'corner_handling'
        ]
        
        # Define feature groups for different models (matching data_processor.py)
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
            'corner_handling': [
                'speedX', 'angle', 'trackPos',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9'
            ]
        }
        
        try:
            # Load models and scalers
            self._load_models()
            self._load_scalers()
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise
    
    def _load_models(self):
        """Load all trained models"""
        model_types = ['high_level', 'tactical', 'low_level', 'corner_handling']
        
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f'{model_type}_model.keras')
            if os.path.exists(model_path):
                try:
                    # Load model with custom_objects to handle optimizer
                    self.models[model_type] = keras.models.load_model(
                        model_path,
                        custom_objects={'optimizer': tf.keras.optimizers.Adam()}
                    )
                    self.logger.info(f"Loaded {model_type} model from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_type} model: {str(e)}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
    
    def _load_scalers(self):
        """Load all scalers"""
        model_types = ['high_level', 'tactical', 'low_level', 'corner_handling']
        
        for model_type in model_types:
            feature_scaler_path = os.path.join(self.scaler_dir, f'{model_type}_feature_scaler.pkl')
            target_scaler_path = os.path.join(self.scaler_dir, f'{model_type}_target_scaler.pkl')
            
            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                try:
                    self.scalers[model_type] = {
                        'feature_scaler': joblib.load(feature_scaler_path),
                        'target_scaler': joblib.load(target_scaler_path)
                    }
                    self.logger.info(f"Loaded scalers for {model_type}")
                except Exception as e:
                    self.logger.error(f"Error loading scalers for {model_type}: {str(e)}")
            else:
                self.logger.warning(f"Scaler files not found for {model_type}")
    
    def prepare_state_data(self, state_data: Dict, model_type: str) -> np.ndarray:
        """Prepare state data for model input"""
        features = []
        
        # Extract features based on model type
        for feature in self.feature_groups[model_type]:
            if feature in state_data:
                value = state_data[feature]
                # Handle track sensors specifically
                if feature.startswith('track_'):
                    # Ensure track sensors are non-negative and have reasonable values
                    value = max(0.0, min(value, 200.0))  # Track sensors typically range 0-200
                features.append(value)
            else:
                # Handle missing features
                if feature.startswith('track_'):
                    features.append(0.0)  # Default track sensor value
                elif feature == 'wheelSpinVel_0':
                    features.append(state_data.get('wheelSpinVel', 0.0))
                else:
                    features.append(0.0)  # Default value for other features
        
        # Convert to numpy array and reshape for sequence
        features = np.array(features, dtype=np.float32)
        
        # Update state history
        self.state_history[model_type].append(features)
        if len(self.state_history[model_type]) > self.sequence_length:
            self.state_history[model_type].pop(0)
        
        # Create sequence
        if len(self.state_history[model_type]) < self.sequence_length:
            # Pad with zeros if not enough history
            padding = np.zeros((self.sequence_length - len(self.state_history[model_type]), len(features)))
            sequence = np.vstack([padding, np.array(self.state_history[model_type])])
        else:
            sequence = np.array(self.state_history[model_type])
        
        # Reshape for model input: (1, sequence_length, features)
        return sequence.reshape(1, self.sequence_length, len(features))
    
    def get_control_actions(self, state_data):
        """Get control actions from all models and combine them"""
        try:
            # Prepare input data for each model
            high_level_input = self.prepare_state_data(state_data, 'high_level')
            tactical_input = self.prepare_state_data(state_data, 'tactical')
            low_level_input = self.prepare_state_data(state_data, 'low_level')
            corner_input = self.prepare_state_data(state_data, 'corner_handling')
            
            # Get predictions from each model
            high_level_pred = self.models['high_level'].predict(high_level_input, verbose=0)
            tactical_pred = self.models['tactical'].predict(tactical_input, verbose=0)
            low_level_pred = self.models['low_level'].predict(low_level_input, verbose=0)
            corner_pred = self.models['corner_handling'].predict(corner_input, verbose=0)
            
            # Log predictions for debugging
            self.logger.info(f"High-level prediction shape: {high_level_pred.shape}")
            self.logger.info(f"High-level prediction mean: {np.mean(high_level_pred):.4f}")
            self.logger.info(f"Tactical prediction shape: {tactical_pred.shape}")
            self.logger.info(f"Tactical prediction mean: {np.mean(tactical_pred):.4f}")
            self.logger.info(f"Low-level prediction shape: {low_level_pred.shape}")
            self.logger.info(f"Low-level prediction mean: {np.mean(low_level_pred):.4f}")
            self.logger.info(f"Corner handling prediction shape: {corner_pred.shape}")
            self.logger.info(f"Corner handling prediction mean: {np.mean(corner_pred):.4f}")
            
            # Combine predictions with weighted average
            weights = {
                'high_level': 0.35,    # Increased from 0.3 due to best overall performance
                'tactical': 0.35,      # Increased from 0.3 due to excellent performance
                'low_level': 0.15,     # Decreased from 0.2 due to mixed performance
                'corner_handling': 0.15 # Decreased from 0.2 due to poor brake performance
            }
            
            combined_pred = (
                weights['high_level'] * high_level_pred +
                weights['tactical'] * tactical_pred +
                weights['low_level'] * low_level_pred +
                weights['corner_handling'] * corner_pred
            )
            
            # Extract control values and scale them for more pronounced response
            steer = float(combined_pred[0][0]) * 1.5  # Amplify steering response
            accel = float(combined_pred[0][1])
            brake = float(combined_pred[0][2])
            
            # Get current state values
            speed = state_data.get('speedX', 0.0)
            rpm = state_data.get('rpm', 0.0)
            track_pos = state_data.get('trackPos', 0.0)
            track_sensors = [state_data.get(f'track_{i}', -1.0) for i in range(19)]
            track_sensor_mean = np.mean(track_sensors)
            
            # Log state for debugging
            self.logger.info(f"Current speed: {speed:.2f}")
            self.logger.info(f"Current RPM: {rpm:.2f}")
            self.logger.info(f"Track position: {track_pos:.2f}")
            self.logger.info(f"Track sensors: {track_sensors[:5]}")  # Show first 5 sensors
            self.logger.info(f"Track sensor mean: {track_sensor_mean:.2f}")
            
            # Check if car is off track
            is_off_track = abs(track_pos) > 1.0 or track_sensor_mean < 0.0
            self.logger.info(f"Off track: {is_off_track}")
            
            # Adjust controls based on state
            if is_off_track:
                # When off track, prioritize getting back on track
                if track_pos > 0:  # Off track to the right
                    steer = -0.5  # Steer left
                else:  # Off track to the left
                    steer = 0.5   # Steer right
                
                # Reduce speed when off track
                accel = 0.3
                brake = 0.2
            else:
                # Normal driving adjustments
                # Limit steering based on speed
                max_steer = 0.5
                if speed > 50:
                    max_steer = 0.3
                elif speed > 30:
                    max_steer = 0.4
                steer = np.clip(steer, -max_steer, max_steer)
                
                # Add track position correction
                track_correction = -track_pos * 0.3  # Gentle correction towards center
                steer = np.clip(steer + track_correction, -max_steer, max_steer)
                
                # Adjust acceleration based on speed and RPM
                if speed < 5.0:  # Very slow
                    accel = 0.5  # More aggressive acceleration
                    brake = 0.0
                elif speed > 100:  # Very fast
                    accel = 0.3
                    brake = 0.1
                else:
                    # More conservative brake control due to poor model performance
                    accel = np.clip(accel, 0.0, 0.5)
                    brake = np.clip(brake, 0.0, 0.2)  # Reduced max brake from 0.3 to 0.2
            
            # Apply control smoothing
            steer = self.apply_control_smoothing(steer, 'steer')
            accel = self.apply_control_smoothing(accel, 'accel')
            brake = self.apply_control_smoothing(brake, 'brake')
            
            # Log final controls
            self.logger.info(f"Final controls: steer={steer:.4f}, accel={accel:.4f}, brake={brake:.4f}")
            
            return {
                'steer': steer,
                'accel': accel,
                'brake': brake
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_control_actions: {e}")
            # Return safe default controls
            return {
                'steer': 0.0,
                'accel': 0.0,
                'brake': 1.0
            }
    
    def apply_control_smoothing(self, value, control_name):
        """Apply smoothing to prevent sudden control changes"""
        smoothed_value = (1 - self.control_smoothing) * value + self.control_smoothing * self.last_controls[control_name]
        self.last_controls[control_name] = smoothed_value
        return smoothed_value