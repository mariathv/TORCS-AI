import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import joblib
import time
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging at module level - reduce logging overhead
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Force CPU usage to avoid GPU-related overhead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ModelCoordinator:
    def __init__(self, models_dir: str = 'models', scalers_dir: str = 'model_scalers'):
        """Initialize the model coordinator with trained models and scalers"""
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        self.model_dir = models_dir
        self.scaler_dir = scalers_dir
        self.models = {}
        self.scalers = {}
        self.configs = {}
        self.last_controls = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0}
        self.control_smoothing = 0.2
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.last_report_time = time.time()
        
        # Output logging
        self.log_frequency = 30  # Log every 30 frames
        self.log_counter = 0
        
        # Model weights for combining predictions
        self.model_weights = {
            'high_level': 0.5,  # Equal weight for high-level model
            'tactical': 0.5     # Equal weight for tactical model
        }
        
        # Sequence handling
        self.sequence_length = 10
        self.state_history = {
            'high_level': [],
            'tactical': []
        }
        
        # Pre-allocate feature arrays for all models
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
            ]
        }
        
        self.feature_arrays = {}
        self.sequence_arrays = {}
        
        for model_type in ['high_level', 'tactical']:
            feature_count = len(self.feature_groups[model_type])
            self.feature_arrays[model_type] = np.zeros((1, feature_count), dtype=np.float32)
            self.sequence_arrays[model_type] = np.zeros((1, self.sequence_length, feature_count), dtype=np.float32)
        
        try:
            # Load both models for combined predictions
            self._load_models(['high_level', 'tactical'])
            self._load_scalers(['high_level', 'tactical'])
            
            # Create optimized prediction functions
            @tf.function(experimental_compile=True)
            def predict_high_level(x_input):
                return self.models['high_level'](x_input, training=False)
            
            @tf.function(experimental_compile=True)
            def predict_tactical(x_input):
                return self.models['tactical'](x_input, training=False)
            
            self.predict_fns = {
                'high_level': predict_high_level,
                'tactical': predict_tactical
            }
            
            # Pre-warm the models with dummy predictions
            for model_type in ['high_level', 'tactical']:
                feature_count = len(self.feature_groups[model_type])
                dummy_input = np.zeros((1, self.sequence_length, feature_count), dtype=np.float32)
                _ = self.predict_fns[model_type](dummy_input)
            
            print("MODEL COORDINATOR: Using high_level + tactical models")
            print(f"MODEL WEIGHTS: high_level={self.model_weights['high_level']}, tactical={self.model_weights['tactical']}")
            
            for model_type in ['high_level', 'tactical']:
                if model_type in self.models:
                    print(f"{model_type.upper()} MODEL: {self.models[model_type].input_shape} → {self.models[model_type].output_shape}")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise
    
    def _load_models(self, model_types):
        """Load specified models"""
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f'{model_type}_model.keras')
            if os.path.exists(model_path):
                try:
                    # Load model with custom_objects to handle optimizer
                    self.models[model_type] = keras.models.load_model(
                        model_path,
                        custom_objects={'optimizer': tf.keras.optimizers.Adam()},
                        compile=False  # Skip compilation for faster loading
                    )
                    print(f"MODEL LOADED: {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_type} model: {str(e)}")
                    raise
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                print(f"WARNING: Model file not found: {model_path}")
                # Don't raise an exception, just remove from weights
                if model_type in self.model_weights:
                    del self.model_weights[model_type]
    
    def _load_scalers(self, model_types):
        """Load scalers for specified models"""
        for model_type in model_types:
            # Skip if model wasn't loaded
            if model_type not in self.models:
                continue
                
            feature_scaler_path = os.path.join(self.scaler_dir, f'{model_type}_feature_scaler.pkl')
            target_scaler_path = os.path.join(self.scaler_dir, f'{model_type}_target_scaler.pkl')
            
            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                try:
                    self.scalers[model_type] = {
                        'feature_scaler': joblib.load(feature_scaler_path),
                        'target_scaler': joblib.load(target_scaler_path)
                    }
                    print(f"SCALERS LOADED: {feature_scaler_path}")
                except Exception as e:
                    self.logger.error(f"Error loading scalers for {model_type}: {str(e)}")
            else:
                self.logger.warning(f"Scaler files not found for {model_type}")
    
    def prepare_input(self, state_data: Dict, model_type: str) -> np.ndarray:
        """Prepare state data for model input with sequence handling"""
        # Skip if model wasn't loaded
        if model_type not in self.models:
            return None
            
        # Reuse pre-allocated array
        features = self.feature_arrays[model_type][0]
        
        # Extract features for the specified model
        for i, feature in enumerate(self.feature_groups[model_type]):
            if feature in state_data:
                value = state_data[feature]
                # Handle track sensors specifically
                if feature.startswith('track_'):
                    # Ensure track sensors are non-negative and have reasonable values
                    value = max(0.0, min(value, 200.0))
                features[i] = value
            else:
                # Handle missing features
                features[i] = 0.0
        
        # Apply scaling if available
        if model_type in self.scalers and 'feature_scaler' in self.scalers[model_type]:
            try:
                features_scaled = self.scalers[model_type]['feature_scaler'].transform(self.feature_arrays[model_type])[0]
                # Copy scaled features back to our array
                for i in range(len(features)):
                    features[i] = features_scaled[i]
            except Exception as e:
                pass
        
        # Update state history for sequence
        self.state_history[model_type].append(features.copy())
        if len(self.state_history[model_type]) > self.sequence_length:
            self.state_history[model_type].pop(0)
        
        # Create sequence
        sequence = self.sequence_arrays[model_type][0]
        if len(self.state_history[model_type]) < self.sequence_length:
            # Pad with zeros if not enough history
            padding_length = self.sequence_length - len(self.state_history[model_type])
            # Fill padding with zeros
            for i in range(padding_length):
                sequence[i].fill(0.0)
            # Fill the rest with actual history
            for i, hist_features in enumerate(self.state_history[model_type]):
                sequence[padding_length + i] = hist_features
        else:
            # Fill with history
            for i, hist_features in enumerate(self.state_history[model_type]):
                sequence[i] = hist_features
        
        return self.sequence_arrays[model_type]
    
    def get_control_actions(self, state_data):
        """Get control actions using all available models"""
        start_time = time.time()
        self.frame_count += 1
        self.log_counter += 1
        
        try:
            # Prepare input data for all models
            model_inputs = {}
            model_predictions = {}
            
            # Get available models (those that were successfully loaded)
            available_models = [model_type for model_type in self.model_weights.keys() 
                               if model_type in self.models]
            
            # Normalize weights for available models
            total_weight = sum(self.model_weights[model_type] for model_type in available_models)
            normalized_weights = {model_type: self.model_weights[model_type] / total_weight 
                                for model_type in available_models}
            
            # Prepare inputs and get predictions for each model
            for model_type in available_models:
                model_inputs[model_type] = self.prepare_input(state_data, model_type)
                if model_inputs[model_type] is not None:
                    model_pred = self.predict_fns[model_type](model_inputs[model_type])
                    
                    # Convert to numpy arrays
                    if hasattr(model_pred, 'numpy'):
                        model_predictions[model_type] = model_pred.numpy()[0]
                    else:
                        model_predictions[model_type] = tf.keras.backend.get_value(model_pred)[0]
            
            # Extract individual control values from each model
            model_controls = {}
            for model_type in model_predictions:
                steer, accel, brake = model_predictions[model_type]
                model_controls[model_type] = {
                    'steer': steer,
                    'accel': accel,
                    'brake': brake
                }
            
            # Combine predictions with weighted average
            steer = 0.0
            accel = 0.0
            brake = 0.0
            
            for model_type in model_controls:
                steer += normalized_weights[model_type] * model_controls[model_type]['steer']
                accel += normalized_weights[model_type] * model_controls[model_type]['accel']
                brake += normalized_weights[model_type] * model_controls[model_type]['brake']
            
            # Log raw neural network outputs
            if self.log_counter >= self.log_frequency:
                print("\n===== NEURAL NETWORK OUTPUTS =====")
                for model_type in model_controls:
                    controls = model_controls[model_type]
                    print(f"{model_type.upper()}: steer={controls['steer']:.4f}, accel={controls['accel']:.4f}, brake={controls['brake']:.4f}")
                print(f"COMBINED:   steer={steer:.4f}, accel={accel:.4f}, brake={brake:.4f}")
            
            # Get basic state variables for logging
            speed = state_data.get('speedX', 0.0)
            track_pos = state_data.get('trackPos', 0.0)
            
            # Ensure acceleration is in valid range
            accel = max(0.0, min(1.0, accel))  # Clip between 0.0 and 1.0
            
            # Apply control smoothing
            steer_final = self.apply_control_smoothing(steer, 'steer')
            accel_final = self.apply_control_smoothing(accel, 'accel')
            brake_final = self.apply_control_smoothing(brake, 'brake')
            
            # Log detailed output parameters
            if self.log_counter >= self.log_frequency:
                print("\n----- Input State -----")
                print(f"Speed: {speed:.2f}, Track Position: {track_pos:.2f}")
                print(f"Track Sensors: {[state_data.get(f'track_{i}', -1.0) for i in range(5)]}")
                
                print("\n----- Processing Steps -----")
                print(f"1. Raw NN Outputs:    (see above)")
                print(f"2. Combined Output:   steer={steer:.4f}, accel={accel:.4f}, brake={brake:.4f}")
                print(f"3. Final Smoothed:    steer={steer_final:.4f}, accel={accel_final:.4f}, brake={brake_final:.4f}")
                
                # Reset counter
                self.log_counter = 0
            
            # Log performance stats occasionally
            current_time = time.time()
            if current_time - self.last_report_time > 10.0:  # Report every 10 seconds
                elapsed = current_time - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                frame_time = (time.time() - start_time) * 1000
                
                model_names = "+".join(available_models)
                print(f"\nPERF: {frame_time:.1f}ms | FPS: {fps:.1f} | Using {model_names} models")
                self.last_report_time = current_time
            
            return {
                'steer': steer_final,
                'accel': accel_final,
                'brake': brake_final
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_control_actions: {e}")
            # Return safe default controls on error
            return {
                'steer': 0.0,
                'accel': 0.5,
                'brake': 0.0
            }
    
    def apply_control_smoothing(self, value, control_name):
        """Apply smoothing to prevent sudden control changes"""
        # Get previous value
        prev_value = self.last_controls[control_name]
        
        # Use different smoothing factors for different controls
        if control_name == 'steer':
            # Less smoothing for steering to be more responsive
            smoothing_factor = 0.15
            
            # Add extra logic for steering to address persistent bias issues
            # If we're turning right (negative) and have been turning right,
            # make it easier to return to center or turn left
            if value > prev_value and prev_value < 0:
                # Moving from right towards center/left - reduce smoothing to allow quicker correction
                smoothing_factor = 0.1
            # If we're turning left (positive) and have been turning left,
            # ensure we don't overcompensate
            elif value < prev_value and prev_value > 0:
                # Moving from left towards center/right - standard smoothing
                smoothing_factor = 0.15
            # If we're at center and trying to turn left, reduce smoothing further
            elif value > 0 and abs(prev_value) < 0.1:
                smoothing_factor = 0.1
        else:
            # Standard smoothing for acceleration and braking
            smoothing_factor = self.control_smoothing
            
        # Calculate smoothed value with time-based decay
        smoothed_value = (1 - smoothing_factor) * value + smoothing_factor * prev_value
        
        # Store for next iteration
        self.last_controls[control_name] = smoothed_value
        
        return smoothed_value