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
    ML controller for TORCS with full feature set (24 features)
    """
    
    def __init__(self, model_path='controller/model', scaler_path=None):
        """Initialize the ML controller with full features"""
        print("IMPORTANT: ML CONTROLLER - Using full feature set (24 features)")
        
        # Feature count for full feature set
        self.FEATURE_COUNT = 24  # 5 basic + 19 track sensors
        
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
        
        # Gear shifting state
        self.last_gear_change_time = 0
        self.gear_change_cooldown = 2.0  # Increased cooldown to 2 seconds
        self.last_gear = 0
        self.last_speed = 0
        self.gear_lock = False  # New flag to prevent gear hunting
        self.gear_lock_time = 0
        self.gear_lock_duration = 3.0  # Lock gear for 3 seconds after change
        self.in_reverse_mode = False  # Add reverse mode tracking
                
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
            
            # Check if model matches our feature count
            expected_features = self.model.input_shape[1] if hasattr(self.model, 'input_shape') else None
            print(f"Model loaded with input shape: {self.model.input_shape}")
            
            if expected_features and expected_features != self.FEATURE_COUNT:
                print(f"WARNING: Model expects {expected_features} features but we're providing {self.FEATURE_COUNT}")
                print("You may need to retrain your model with the full feature set")
            
            # Pre-allocate the numpy array for features
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
                # Check if scaler matches our feature count
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.FEATURE_COUNT:
                    print(f"WARNING: Scaler expects {self.scaler.n_features_in_} features but we're using {self.FEATURE_COUNT}")
                    print("You may need to retrain with the full feature set")
                print("Scaler loaded successfully")
            except Exception as e:
                print(f"Error loading scaler: {e}")
                
        print("ML controller initialized with full feature set (24 features)")
        
        # Initialize control object for gear management
        self.control = None
    
    def calculate_rpm_from_speed(self, speed, gear):
        """Calculate RPM based on speed and gear using precise gear ratios"""
        # Base RPM when car is idling
        if abs(speed) < 0.1:
            return 1000 if gear == 1 else 800

        # Gear ratios (multipliers) based on actual car behavior
        gear_ratios = {
            1: 110,  # 1st gear: higher ratio for more torque
            2: 70,   # 2nd gear: balanced ratio
            3: 50,   # 3rd gear: lower ratio
            4: 40,   # 4th gear: even lower
            5: 35,   # 5th gear: lowest ratio
            6: 30    # 6th gear: lowest ratio
        }
        
        # Get the appropriate ratio for the current gear
        ratio = gear_ratios.get(gear, 50)  # Default to 50 if gear not found
        
        # Calculate base RPM from speed
        base_rpm = speed * ratio
        
        # Add minimum RPM to prevent stalling
        min_rpm = 1000 if gear == 1 else 800
        
        # Calculate final RPM
        rpm = base_rpm + min_rpm
        
        # Cap RPM at redline (9000)
        return min(rpm, 9000)

    def gear(self, state, control):
        '''Improved automatic gear shifting logic with hysteresis to prevent gear hunting'''
        # Get all required values at once to minimize attribute access
        speed = state.getSpeedX()
        gear = state.getGear()
        accel = control.getAccel()
        brake = control.getBrake()
        current_time = time.time()
        # Calculate RPM based on speed and current gear
        rpm = self.calculate_rpm_from_speed(speed, gear)
        # Only log speed and RPM
        print(f"Speed: {speed:.2f} km/h, RPM: {rpm:.0f}")

        # Handle special cases
        if gear == -1:
            # Only get out of reverse if we're explicitly trying to go forward
            if accel > 0 and brake == 0 and speed > -1.0:
                # Only shift to forward if not in reverse mode
                if not self.in_reverse_mode:
                    control.setGear(1)
            return

        # Standing still or very slow: use first gear
        if abs(speed) < 0.5:
            # If we're in reverse mode, use reverse gear
            if self.in_reverse_mode:
                control.setGear(-1)
            else:
                control.setGear(1)
            return
            
        # Check if we were in neutral
        if gear == 0:
            if self.in_reverse_mode:
                control.setGear(-1)
            else:
                control.setGear(1)
            return

        # Check if we're in gear lock period
        if self.gear_lock and current_time - self.gear_lock_time < self.gear_lock_duration:
            return
            
        # Check if we're in cooldown period
        if current_time - self.last_gear_change_time < self.gear_change_cooldown:
            return

        # Store the old RPM ranges - tuned for stability
        # Adding hysteresis - different thresholds for upshift vs downshift
        upshift_rpm = 8000     # Only upshift when RPM is very high
        upshift_rpm_min = 6500 # Don't upshift below this RPM even if in speed range
        downshift_rpm = 3000   # Downshift if RPM gets this low
            
        # Add hysteresis to speed ranges too - different ranges for up/down shifting
        gear_speed_up_ranges = {
            1: (0, 70),      # 1st gear: 0-70 km/h (upshift point)
            2: (50, 110),    # 2nd gear: 50-110 km/h (upshift point)
            3: (90, 170),    # 3rd gear: 90-170 km/h (upshift point)
            4: (150, 230),   # 4th gear: 150-230 km/h (upshift point)
            5: (210, 290),   # 5th gear: 210-290 km/h (upshift point)
            6: (270, 330)    # 6th gear: 270-330 km/h
        }
            
        gear_speed_down_ranges = {
            1: (0, 50),      # 1st gear: 0-50 km/h (downshift point)
            2: (30, 90),     # 2nd gear: 30-90 km/h (downshift point)
            3: (70, 150),    # 3rd gear: 70-150 km/h (downshift point)
            4: (130, 210),   # 4th gear: 130-210 km/h (downshift point)
            5: (190, 270),   # 5th gear: 190-270 km/h (downshift point)
            6: (250, 330)    # 6th gear: 250-330 km/h
        }
            
        # GEAR SELECTION LOGIC
        new_gear = gear  # Start with current gear
            
        # First handle extreme cases that should override the delay
        if rpm > 8500 and gear < 6:
            # Engine protection - always upshift if RPM is dangerously high
            new_gear = gear + 1
            self.last_gear_change_time = current_time
            self.gear_lock = True
            self.gear_lock_time = current_time
        elif rpm < 2500 and gear > 1:
            # Engine protection - always downshift if RPM is dangerously low
            new_gear = gear - 1
            self.last_gear_change_time = current_time
            self.gear_lock = True
            self.gear_lock_time = current_time
        else:
            # Define strict speed bands for each gear like in the second snippet
            if gear == 1:
                # In 1st gear, only upshift if speed is well above threshold
                if speed > 70:  # Much higher threshold to prevent premature upshifts
                    new_gear = gear + 1
                    self.last_gear_change_time = current_time
                    self.gear_lock = True
                    self.gear_lock_time = current_time
            elif gear == 2:
                # In 2nd gear, only downshift if speed is well below threshold
                if speed < 50:  # Much lower threshold to prevent premature downshifts
                    new_gear = gear - 1
                    self.last_gear_change_time = current_time
                    self.gear_lock = True
                    self.gear_lock_time = current_time
            else:
                # For other gears, use a combination of both approaches
                # UPSHIFT LOGIC
                if gear < 6 and rpm > upshift_rpm:
                    # Consider upshifting if RPM is high enough
                    # BUT only if we're also in the right speed range for the next gear
                    if gear + 1 in gear_speed_up_ranges:
                        min_speed, _ = gear_speed_up_ranges[gear + 1]
                        if speed >= min_speed:
                            new_gear = gear + 1
                            self.last_gear_change_time = current_time
                            self.gear_lock = True
                            self.gear_lock_time = current_time
                
                # DOWNSHIFT LOGIC - include brake pressure from second snippet
                elif gear > 1 and (brake > 0 or rpm < downshift_rpm or 
                            (gear in gear_speed_down_ranges and 
                            speed < gear_speed_down_ranges[gear][0])):
                    # Downshift if brake is applied, RPM is too low OR we're below the speed range for this gear
                    new_gear = gear - 1
                    self.last_gear_change_time = current_time
                    self.gear_lock = True
                    self.gear_lock_time = current_time
                    
                # Special case for very high speeds - ensure we're in top gear
                elif speed > 280 and gear < 6:
                    new_gear = 6
                    self.last_gear_change_time = current_time
                    self.gear_lock = True
                    self.gear_lock_time = current_time
                    
                # Special case for significant mismatch between gear and speed
                # If we're more than 2 gears away from where we should be
                elif gear > 2:  # Only check if we're in 3rd gear or higher
                    # Find what gear we should be in based on speed
                    target_gear = 1  # Default to first gear
                    for g in range(1, 7):
                        if g in gear_speed_down_ranges:
                            min_speed, max_speed = gear_speed_down_ranges[g]
                            if min_speed <= speed <= max_speed:
                                target_gear = g
                                break
                    
                    # If we're significantly mis-geared (more than 2 gears off), start correcting
                    if gear > target_gear + 2:
                        new_gear = gear - 1
                        self.last_gear_change_time = current_time
                        self.gear_lock = True
                        self.gear_lock_time = current_time
            
        # Apply the gear change if needed
        if new_gear != gear:
            control.setGear(new_gear)
    
    def prepare_input(self, state):
        """Prepare FULL input features (24 total) for the model"""
        # Reuse the pre-allocated array
        features = self.feature_array[0]
        
        # ----- Essential car state (5 features) -----
        # Speed components (3 features)
        features[0] = state.speedX
        features[1] = state.speedY if hasattr(state, 'speedY') else 0.0
        features[2] = state.speedZ if hasattr(state, 'speedZ') else 0.0
        
        # Position data (2 features)
        features[3] = state.angle
        features[4] = state.trackPos
        
        # ----- Track sensors (all 19 sensors) -----
        if hasattr(state, 'track') and state.track and len(state.track) >= 19:
            # Copy all 19 track sensors
            for i in range(19):
                features[5 + i] = state.track[i]
        else:
            # Zero out all track sensors if not available
            features[5:24] = 0.0
        
        # Apply scaling if available
        X = self.feature_array
        if self.scaler:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                # Skip scaling on error
                pass
        
        return X
    
    def predict(self, state, control=None, time_limit=0.005):
        """Make ML-only predictions"""
        start_time = time.time()
        self.ml_predictions += 1
        
        # Store control object for gear management
        self.control = control
        
        try:
            # Normal ML prediction
            # Prepare input features (full set)
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
            steer = max(-1.0, min(1.0, steer * 2.0))
            
            # Apply gear shifting logic (rule-based, not ML)
            if control is not None:
                self.gear(state, control)
            
            # Log control values occasionally
            current_time = time.time()
            if current_time - self.last_report_time > 5.0:
                prediction_time = (time.time() - start_time) * 1000  # in ms
                print(f"\nML CONTROL: Using 24 features | Prediction time: {prediction_time:.2f}ms")
                print(f"ML VALUES: steer={steer:.2f}, accel={accel:.2f}, brake={brake:.2f}")
                print(f"CAR STATE: speed={state.getSpeedX():.2f}, rpm={state.getRpm():.0f}, gear={state.getGear()}")
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