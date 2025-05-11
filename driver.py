import msgParser
import carState
import carControl
import csv
import os
from datetime import datetime
import keyboard  
from telemetryLogger import TelemetryLogger
import time
import math
import numpy as np

# Import ML model controller if available (don't fail if not available)
try:
    from controller.model_controller import ModelController, SimpleRuleController
    ML_CONTROLLER_AVAILABLE = True
except ImportError:
    ML_CONTROLLER_AVAILABLE = False
    print("ML controller not available. Falling back to rule-based control.")

class Driver(object):
    '''
    A driver object for the SCRC with manual control and telemetry logging
    '''

    def __init__(self, stage, manual_mode=False, max_episodes=1, model_coordinator=None):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 330  # Updated max speed
        self.prev_rpm = None
        
        # Store last steering value for smoothing
        self.last_steer = 0
        # Store last acceleration value for smoothing
        self.last_accel = 0
        # Keep track of time in reverse
        self.reverse_time = 0
        # Track when we started braking
        self.brake_start_time = 0
        # Track if we're currently in reverse mode
        self.in_reverse_mode = False

        self.manual_mode = manual_mode
        self.max_episodes = max_episodes
        self.model_coordinator = model_coordinator
        
        # Initialize telemetry logger
        self.telemetry_logger = TelemetryLogger(
            stage=stage, 
            max_episodes=max_episodes
        )
        
        # Ask if user wants to log data in model mode
        self.log_in_model_mode = False
        if not manual_mode:
            user_input = input("Log data in model/autonomous mode? (y/n): ").strip().lower()
            self.log_in_model_mode = user_input == 'y'
        
        print(f"INFO: Telemetry logging {'enabled' if manual_mode or self.log_in_model_mode else 'disabled'} for {manual_mode and 'manual' or 'autonomous'} mode")
        
        # Initialize ML model controller if available
        self.ml_controller = None
        self.rule_controller = None
        
        if ML_CONTROLLER_AVAILABLE and not manual_mode:
            # Default to controller/model if not specified
            if model_coordinator is None:
                model_coordinator = 'controller/model'
            
            try:
                # Create the main ML controller
                self.ml_controller = ModelController(model_path=model_coordinator)
                print(f"ML Controller initialized with model: {model_coordinator}")
                
                # Also create a simple rule controller as a backup only for emergencies
                self.rule_controller = SimpleRuleController()
                print("Note: Simple rule controller will ONLY be used as emergency fallback")
            except Exception as e:
                print(f"Error initializing ML controller: {e}")
                
        # Performance tracking for drive method
        self.drive_times = []
        
        # Always use ML strategy as required
        self.use_ml_strategy = True
        print("IMPORTANT: Always using ML control strategy as required")
    
    def __del__(self):
        '''Destructor to close telemetry logger'''
        if hasattr(self, 'telemetry_logger'):
            self.telemetry_logger.close()
        pass
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        self.angles[9] = 0
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        # Start timing this method to catch timeouts
        start_time = time.time()
        
        try:
            # Process the message
            self.state.setFromMsg(msg)
            
            # Use an absolute short time limit to ensure we never time out
            MAX_PROCESS_TIME = 0.009  # 9ms, pushing it to the limit but leaving 1ms buffer
            
            if self.manual_mode:
                # Manual control mode
                self.manual_control()
                
                # Log data if time permits and in manual mode
                elapsed = time.time() - start_time
                if elapsed < MAX_PROCESS_TIME - 0.001:
                    self.telemetry_logger.log_data(self.state, self.control)
            else:
                # First, always apply gear shifting (critical)
                self.gear()
                
                # Always use ML control (as required)
                # Time left for control decisions
                elapsed = time.time() - start_time
                remaining_time = MAX_PROCESS_TIME - elapsed
                
                # Use ML control with maximum possible time budget
                self.autonomous_control(time_limit=remaining_time-0.001)
                
                # Log data if user requested and time permits in autonomous mode
                if self.log_in_model_mode:
                    elapsed = time.time() - start_time
                    if elapsed < MAX_PROCESS_TIME - 0.001:
                        self.telemetry_logger.log_data(self.state, self.control)
            
            # Track total processing time
            total_time = time.time() - start_time
            self.drive_times.append(total_time)
            
            # Occasionally print performance stats
            if len(self.drive_times) % 100 == 0:
                avg_time = sum(self.drive_times[-100:]) / 100
                max_time = max(self.drive_times[-100:])
                print(f"Drive method stats - avg: {avg_time*1000:.2f}ms, max: {max_time*1000:.2f}ms")
                # Clear old data
                if len(self.drive_times) > 500:
                    self.drive_times = self.drive_times[-100:]
            
            # Check if we're getting too close to timeout
            if total_time > 0.009:  # Over 9ms is dangerous
                print(f"WARNING: Drive method took {total_time*1000:.2f}ms, approaching timeout!")
            
            return self.control.toMsg()
            
        except Exception as e:
            print(f"Error in drive method: {e}")
            # Return simple control message as fallback for emergencies only
            self.control.setSteer(0)
            self.control.setAccel(0.1)
            self.control.setBrake(0)
            self.control.setGear(1)
            return self.control.toMsg()
    
    def smooth_value(self, current, previous, smoothing_factor=0.1):
        """Apply smoothing between current and previous value"""
        return previous + smoothing_factor * (current - previous)
    
    def manual_control(self):
        '''Manual control with automatic gear shifting'''
        current_speed = self.state.getSpeedX()
        current_gear = self.state.getGear()
        
        # Steering control with smoother transitions
        target_steer = 0
        if keyboard.is_pressed('right'):
            target_steer = -0.5  # Reduced magnitude for gentler steering
        elif keyboard.is_pressed('left'):
            target_steer = 0.5   # Reduced magnitude for gentler steering
        
        # Apply stronger smoothing to steering (lower value = smoother transition)
        smooth_steer = self.smooth_value(target_steer, self.last_steer, 0.15)
        self.control.setSteer(smooth_steer)
        self.last_steer = smooth_steer
        
        # Handle reverse mode tracking
        if keyboard.is_pressed('down') and current_speed < 1.0:
            # If down key is pressed and we're almost stopped, consider it reverse intent
            if not self.in_reverse_mode and current_gear != -1:
                self.in_reverse_mode = True
                
        if keyboard.is_pressed('up') and self.in_reverse_mode:
            # If up key is pressed while in reverse mode, exit reverse mode
            self.in_reverse_mode = False
            
        # Acceleration / Braking / Reversing with improved logic
        if keyboard.is_pressed('up'):
            if self.in_reverse_mode or current_gear == -1:
                # Exit reverse mode and shift to forward gear
                self.in_reverse_mode = False
                self.control.setGear(1)
                # Apply gentle acceleration from reverse
                target_accel = 0.3
            else:
                # Progressive acceleration based on speed
                max_accel = 1.0 if current_speed < self.max_speed else 0.3
                target_accel = min(max_accel, self.last_accel + 0.1)  # Smooth acceleration ramp-up
            
            self.control.setBrake(0)
            
        elif keyboard.is_pressed('down'):
            if current_speed > 1.0:  
                # If moving forward, apply progressive braking
                if self.brake_start_time == 0:
                    self.brake_start_time = time.time()
                
                # Calculate how long we've been braking
                brake_duration = time.time() - self.brake_start_time
                
                # Increase brake pressure over time, up to a maximum
                brake_pressure = min(1.0, 0.3 + brake_duration * 0.5)
                
                self.control.setAccel(0)
                self.control.setBrake(brake_pressure)
                target_accel = 0
            else:
                # Stopped or almost stopped, handle reverse
                self.brake_start_time = 0  # Reset brake timer
                
                # If we're in reverse mode, ensure the gear is -1 and apply acceleration
                if self.in_reverse_mode or current_gear == -1:
                    self.control.setGear(-1)
                    target_accel = min(0.7, self.last_accel + 0.05)  # Gentle acceleration in reverse
                else:
                    # Just stopping, not yet in reverse
                    target_accel = 0
                    self.control.setBrake(0.1)  # Light brake to ensure full stop
            
        else:
            # No pedal inputs, coast with minimal braking
            self.brake_start_time = 0  # Reset brake timer
            target_accel = max(0, self.last_accel - 0.1)  # Gradually reduce acceleration
            self.control.setBrake(0)
        
        # Apply smoothed acceleration
        smooth_accel = self.smooth_value(target_accel, self.last_accel, 0.3)
        self.control.setAccel(smooth_accel)
        self.last_accel = smooth_accel
        
        # Apply gear shifting
        self.gear()

    def direct_rule_control(self):
        """Direct rule-based control - ONLY used in emergencies when ML fails completely"""
        print("EMERGENCY: Using direct rule control due to critical ML failure")
        # Get control values from rule controller
        if self.rule_controller:
            # Use our optimized rule controller
            controls = self.rule_controller.predict(self.state)
        else:
            # Fallback to basic rules
            controls = self.get_basic_controls()
            
        # Apply control values
        self.control.setSteer(controls['steer'])
        self.control.setAccel(controls['accel'])
        self.control.setBrake(controls['brake'])
    
    def get_basic_controls(self):
        """Simplest possible rule-based controls for emergency fallback only"""
        angle = self.state.angle
        dist = self.state.trackPos
        speed = self.state.speedX
        
        # Simple steering calculation based on track position and angle
        steer = (angle - dist*0.5)/self.steer_lock
        steer = max(-1.0, min(1.0, steer))  # Clip to [-1, 1]
        
        # Basic acceleration/braking based on track position
        if abs(dist) > 0.8:  # Close to edge
            accel = 0.3
            brake = 0.3
        elif abs(angle) > 0.4:  # Sharp turn
            accel = 0.4
            brake = 0.0
        else:  # Clear track
            accel = 0.8
            brake = 0.0
        
        return {
            'steer': steer,
            'accel': accel,
            'brake': brake
        }

    def gear(self):
        '''Improved automatic gear shifting logic with hysteresis to prevent gear hunting'''
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()

        # Handle special cases
        if gear == -1:
            # Only get out of reverse if we're explicitly trying to go forward
            if self.control.getAccel() > 0 and self.control.getBrake() == 0 and speed > -1.0:
                # Only shift to forward if not in reverse mode
                if not self.in_reverse_mode:
                    self.control.setGear(1)
            return

        # Standing still or very slow: use first gear
        if abs(speed) < 0.5:
            # If we're in reverse mode, use reverse gear
            if self.in_reverse_mode:
                self.control.setGear(-1)
            else:
                self.control.setGear(1)
            return
        
        # Check if we were in neutral
        if gear == 0:
            if self.in_reverse_mode:
                self.control.setGear(-1)
            else:
                self.control.setGear(1)
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
        
        # Add gear change delay to prevent rapid shifting
        if not hasattr(self, 'last_gear_change_time'):
            self.last_gear_change_time = 0
        
        current_time = time.time()
        # Don't allow gear changes more frequently than every 0.5 seconds
        # Unless RPM is critically low or high
        time_since_last_change = current_time - self.last_gear_change_time
        gear_change_allowed = time_since_last_change > 0.5 or rpm < 2500 or rpm > 8500
        
        # GEAR SELECTION LOGIC
        new_gear = gear  # Start with current gear
        
        # First handle extreme cases that should override the delay
        if rpm > 8500 and gear < 6:
            # Engine protection - always upshift if RPM is dangerously high
            new_gear = gear + 1
            self.last_gear_change_time = current_time
        elif rpm < 2500 and gear > 1:
            # Engine protection - always downshift if RPM is dangerously low
            new_gear = gear - 1
            self.last_gear_change_time = current_time
        # Otherwise, only change gears if enough time has passed
        elif gear_change_allowed:
            # UPSHIFT LOGIC
            if gear < 6 and rpm > upshift_rpm:
                # Consider upshifting if RPM is high enough
                # BUT only if we're also in the right speed range for the next gear
                if gear + 1 in gear_speed_up_ranges:
                    min_speed, _ = gear_speed_up_ranges[gear + 1]
                    if speed >= min_speed:
                        new_gear = gear + 1
                        self.last_gear_change_time = current_time
            
            # DOWNSHIFT LOGIC
            elif gear > 1 and (rpm < downshift_rpm or 
                            (gear in gear_speed_down_ranges and 
                            speed < gear_speed_down_ranges[gear][0])):
                # Downshift if RPM is too low OR we're below the speed range for this gear
                new_gear = gear - 1
                self.last_gear_change_time = current_time
                
            # Special case for very high speeds - ensure we're in top gear
            elif speed > 280 and gear < 6:
                new_gear = 6
                self.last_gear_change_time = current_time
                
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
        
        # Apply the gear change if needed
        if new_gear != gear:
            self.control.setGear(new_gear)
    
    def autonomous_control(self, time_limit=None):
        '''Use ML controller for autonomous control'''
        try:
            if self.ml_controller:
                # Get predictions from ML controller
                predictions = self.ml_controller.predict(self.state, self.control, time_limit)
                
                # Apply the control actions with smoother transitions
                target_steer = predictions['steer']
                smooth_steer = self.smooth_value(target_steer, self.last_steer, 0.15)
                self.control.setSteer(smooth_steer)
                self.last_steer = smooth_steer
                
                # Apply acceleration with smoothing
                target_accel = predictions['accel']
                smooth_accel = self.smooth_value(target_accel, self.last_accel, 0.3)
                self.control.setAccel(smooth_accel)
                self.last_accel = smooth_accel
                
                # Apply brake value directly (usually more responsive)
                self.control.setBrake(predictions['brake'])
                
                # Gear selection is handled separately by the gear() method
            else:
                print("WARNING: No ML controller available, using fallback controls")
                self.direct_rule_control()
                
        except Exception as e:
            print(f"Error in autonomous control: {e}")
            print("EMERGENCY: Using direct rule control due to critical ML failure")
            self.direct_rule_control()
    
    def onShutDown(self):
        '''Clean up on shutdown'''
        if hasattr(self, 'telemetry_logger'):
            self.telemetry_logger.close()
        pass
    
    def onRestart(self):
        '''Handle restart event'''
        if hasattr(self, 'telemetry_logger'):
            self.telemetry_logger.start_new_episode()
        # Reset control variables
        self.last_steer = 0
        self.last_accel = 0
        self.in_reverse_mode = False
        self.brake_start_time = 0
        pass