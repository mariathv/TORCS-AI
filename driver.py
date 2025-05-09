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
        self.max_speed = 180
        self.prev_rpm = None

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
    
    def manual_control(self):
        '''Manual control with automatic gear shifting'''
        current_speed = self.state.getSpeedX()
        current_gear = self.state.getGear()
        
        # Steering control
        if keyboard.is_pressed('right'):
            self.control.setSteer(-0.75)  # Full left
        elif keyboard.is_pressed('left'):
            self.control.setSteer(0.75)   # Full right
        else:
            self.control.setSteer(0)     # No steering

        # Acceleration / Braking / Reversing
        if keyboard.is_pressed('up'):
            if current_gear == -1:
                # If in reverse, switch to 1st gear before accelerating
                self.control.setGear(1)

            self.control.setAccel(1.0 if current_speed < self.max_speed else 0)
            self.control.setBrake(0)

        elif keyboard.is_pressed('down'):
            if current_speed > 1.0:  
                # If moving forward, apply brakes
                self.control.setAccel(0)
                self.control.setBrake(1.0)
            else:
                # If stopped or moving backward, switch to reverse
                self.control.setGear(-1)
                self.control.setAccel(1.0)
                self.control.setBrake(0)

        else:
            self.control.setAccel(0)
            self.control.setBrake(0)

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
        '''Automatic gear shifting logic'''
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()

        # Handle special cases
        if gear == -1:
            # Only get out of reverse if we're explicitly trying to go forward
            if self.control.getAccel() > 0 and self.control.getBrake() == 0 and speed > -1.0:
                self.control.setGear(1)
            return

        # Standing still or very slow: use first gear
        if abs(speed) < 0.5:
            self.control.setGear(1)
            return

        # Regular shifting based on RPM
        if rpm > 7500 and gear < 6:
            # Upshift if RPM is high
            self.control.setGear(gear + 1)
        elif rpm < 3000 and gear > 1:
            # Downshift if RPM is low
            self.control.setGear(gear - 1)
        
        # Recovery from neutral gear
        if gear == 0:
            self.control.setGear(1)
    
    def autonomous_control(self, time_limit=None):
        '''Use ModelCoordinator for autonomous control'''
        try:
            # Prepare state data for the model
            state_data = {
                'angle': self.state.angle,
                'trackPos': self.state.trackPos,
                'speedX': self.state.speedX,
                'speedY': self.state.speedY,
                'speedZ': self.state.speedZ,
                'rpm': self.state.rpm,
                'gear': self.state.gear,
                'track': self.state.track,  # The track array contains both track and track edge data
                # No need for trackEdge as ModelCoordinator extracts it from track data
                'focus': self.state.focus,
                'fuel': self.state.fuel,
                'distRaced': self.state.distRaced,
                'distFromStart': self.state.distFromStart,
                'racePos': self.state.racePos,
                'z': self.state.z
                # Remove attributes that don't exist in CarState: roll, pitch, yaw, etc.
            }
            
            # Get control actions from model coordinator
            if self.model_coordinator:
                controls = self.model_coordinator.get_control_actions(state_data)
                
                # Apply the control actions
                self.control.setSteer(controls['steer'])
                self.control.setAccel(controls['accel'])
                self.control.setBrake(controls['brake'])
                
                # Gear selection is handled separately by the gear() method
            else:
                print("WARNING: No model coordinator available, using fallback controls")
                self.direct_rule_control()
                
        except Exception as e:
            print(f"Error in autonomous control: {e}")
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
        pass