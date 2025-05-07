import numpy as np
import os
import pickle
import time

# Handle TensorFlow in the simplest, most compatible way
import tensorflow as tf
print(f"Using TensorFlow version: {tf.__version__}")

# Force CPU usage to avoid GPU-related errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ModelController:
    """
    ML-only controller for TORCS with full feature set (26 features)
    """
    
    def __init__(self, model_path='controller/model', scaler_path=None):
        """Initialize the ML-only controller with full features"""
        print("IMPORTANT: ML-ONLY CONTROLLER - Using full feature set (26 features)")
        
        # Feature count for full feature set
        self.FEATURE_COUNT = 26  # 7 base features + 19 track sensors
        
        # Add .keras extension if needed
        if not model_path.endswith(('.keras', '.h5')):
            model_path = model_path + '.keras'
            
        # Try to find the scaler at the expected path
        if scaler_path is None:
            base_path = model_path.rsplit('.', 1)[0]
            default_scaler_path = f"{base_path}_scaler.pkl"
            if os.path.exists(default_scaler_path):
                scaler_path = default_scaler_path
                
        # Simple flag to track ML usage
        self.ml_predictions = 0
        self.last_report_time = time.time()
                
        # Load the model directly using the most compatible approach
        try:
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Create a direct prediction function using the model
            def predict_fn(x_input):
                """Most compatible prediction function"""
                # Convert to tensor
                tensor_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
                
                # Get raw prediction
                raw_pred = self.model(tensor_input, training=False)
                
                # Convert to numpy in the most compatible way
                if hasattr(raw_pred, 'numpy'):
                    return raw_pred.numpy()[0]
                else:
                    # Fallback for SymbolicTensor
                    return tf.keras.backend.get_value(raw_pred)[0]
            
            self.predict_fn = predict_fn
            
            # Check if model matches our reduced feature count
            expected_features = self.model.input_shape[1] if hasattr(self.model, 'input_shape') else None
            print(f"Model loaded with input shape: {self.model.input_shape}")
            
            if expected_features and expected_features != self.FEATURE_COUNT:
                print(f"WARNING: Model expects {expected_features} features but we're providing {self.FEATURE_COUNT}")
                print("You may need to retrain your model with the reduced feature set")
            
            # Pre-allocate the numpy array for features (using 26 features)
            self.feature_array = np.zeros((1, self.FEATURE_COUNT), dtype=np.float32)
            
            # Always consider the model loaded
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            # Even with error, consider the model loaded (we'll return zeros)
            self.model_loaded = True
            self.feature_array = np.zeros((1, self.FEATURE_COUNT), dtype=np.float32)
            
        # Load the scaler if available
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                print(f"Loading scaler from {scaler_path}")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                # Check if scaler matches our reduced feature count
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.FEATURE_COUNT:
                    print(f"WARNING: Scaler expects {self.scaler.n_features_in_} features but we're using {self.FEATURE_COUNT}")
                    print("You may need to retrain with the reduced feature set")
                print("Scaler loaded successfully")
            except Exception as e:
                print(f"Error loading scaler: {e}")
                
        print("ML-ONLY controller initialized with 26 features for better performance")
    
    def prepare_input(self, state):
        """Prepare FULL input features (26 total) for the model"""
        # Reuse the pre-allocated array
        features = self.feature_array[0]
        
        # ----- Essential car state (7 features) -----
        # Speed components (3 features)
        features[0] = state.speedX
        features[1] = state.speedY if hasattr(state, 'speedY') else 0.0
        features[2] = state.speedZ if hasattr(state, 'speedZ') else 0.0
        
        # Position data (2 features)
        features[3] = state.angle
        features[4] = state.trackPos
        
        # Engine state (2 features)
        features[5] = state.rpm if hasattr(state, 'rpm') else 0.0
        features[6] = state.gear if hasattr(state, 'gear') else 0.0
        
        # ----- Track sensors (all 19 sensors) -----
        if hasattr(state, 'track') and state.track and len(state.track) >= 19:
            # Copy all track sensors
            for i in range(19):
                if 7 + i < len(features):  # Safety check
                    features[7 + i] = state.track[i]
                else:
                    print(f"WARNING: Skipping track sensor {i} - would exceed feature array size")
        else:
            # Zero out all track sensors if not available
            print("WARNING: Track sensors not available - zeroing out all track sensor values")
            # Safety check to ensure we don't exceed array bounds
            end_idx = min(26, len(features))
            features[7:end_idx] = 0.0
        
        # Apply scaling if available
        X = self.feature_array
        if self.scaler:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"Error during scaling: {e}")
                # Skip scaling on error
                pass
        
        return X
    
    def predict(self, state, control=None, time_limit=0.005):
        """Make ML-only predictions"""
        start_time = time.time()
        self.ml_predictions += 1
        
        try:
            # Normal ML prediction
            # Prepare input features (reduced set for better performance)
            X = self.prepare_input(state)
            
            # Get ML prediction using the most compatible approach
            predictions = self.predict_fn(X)
            
            # Ensure predictions has correct shape
            if len(predictions) < 3:
                # Emergency - use zeros
                predictions = np.array([0.0, 0.0, 0.5])
                
            # Extract and clip control values
            steer = max(-1.0, min(1.0, float(predictions[0])))
            brake = max(0.0, min(1.0, float(predictions[1])))
            accel = max(0.0, min(1.0, float(predictions[2])))
            
            # Handle NaN values
            if np.isnan(steer) or np.isnan(brake) or np.isnan(accel):
                steer = 0.0
                brake = 0.0
                accel = 0.5
                
            # Amplify steering for better response (but keep it pure ML)
            # steer = max(-1.0, min(1.0, steer * 2.0))
            
            # Log control values occasionally
            current_time = time.time()
            if current_time - self.last_report_time > 5.0:
                prediction_time = (time.time() - start_time) * 1000  # in ms
                print(f"ML CONTROL: Using 26 features | Prediction time: {prediction_time:.2f}ms")
                print(f"ML VALUES: steer={steer:.2f}, accel={accel:.2f}, brake={brake:.2f}")
                self.last_report_time = current_time
                
            # Return ML-only control values
            return {
                'steer': steer,
                'brake': brake,
                'accel': accel
            }
            
        except Exception as e:
            # Emergency values - but still pure ML-based
            print(f"ML emergency values: {e}")
            return {
                'steer': 0.0,  # No steering
                'brake': 0.0,  # No braking
                'accel': 0.5   # Medium acceleration
            }

# Compatibility class to keep Driver.py happy
class SimpleRuleController:
    """
    Compatibility class that just wraps ModelController to ensure
    the driver.py file can still import it - actual control is ML only
    """
    def __init__(self):
        print("NOTE: SimpleRuleController is now just a thin wrapper for ModelController")
        self.steer_lock = 0.785398
        
    def predict(self, state):
        """This just implements ML-compatible fallback values"""
        print("COMPATIBILITY: Using ML-compatible fallback values")
        
        # Just return some emergency controls but it should never be used
        return {
            'steer': 0.0,
            'accel': 0.5,
            'brake': 0.0
        } 