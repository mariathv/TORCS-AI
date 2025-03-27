import msgParser
import carState
import carControl
import csv
import os
from datetime import datetime
import keyboard  
from telemetryLogger import TelemetryLogger

class Driver(object):
    '''
    A driver object for the SCRC with manual control and telemetry logging
    '''

    def __init__(self, stage, manual_mode=False, max_episodes=1):
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
        
        self.telemetry_logger = TelemetryLogger(
            stage=stage, 
            max_episodes=max_episodes
        )
    
    def __del__(self):
        '''Destructor to close telemetry logger'''
        if hasattr(self, 'telemetry_logger'):
            self.telemetry_logger.close()
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
    
        if self.manual_mode:
            self.manual_control()
        else:
            self.autonomous_control()
        
        # Always apply gear shifting logic (works for both modes)
        self.gear()
        
        self.telemetry_logger.log_data(self.state, self.control)
        
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

        self.gear()

    def gear(self):
        '''Automatic gear shifting logic'''
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if gear == -1:
            return  # Skip gear shifting in reverse

        if rpm > 7000 and gear < 6:
            # Upshift if RPM is high
            self.control.setGear(gear + 1)

        elif rpm < 3000 and gear > 1:
            # Downshift if RPM is low and car is slowing
            self.control.setGear(gear - 1)

        # Ensure first gear is engaged if the car is nearly stopped
        if abs(self.state.getSpeedX()) < 1.0 and gear > 1:
            self.control.setGear(1)


    
    def autonomous_control(self):
        '''Original autonomous control methods'''
        self.steer()
        self.gear()
        self.speed()
    
    def _log_telemetry(self):
        '''Log current state and control data to CSV'''
        telemetry_row = [
            self.state.angle, self.state.curLapTime, self.state.damage, 
            self.state.distFromStart, self.state.distRaced, self.state.fuel, 
            self.state.gear, self.state.lastLapTime, self.state.racePos, 
            self.state.rpm, self.state.speedX, self.state.speedY, 
            self.state.speedZ, self.state.trackPos, self.state.z,
            self.control.accel, self.control.brake, self.control.steer, 
            self.control.clutch
        ]
        
        track_values = self.state.track or [0] * 19
        telemetry_row.extend(track_values)
        
        wheel_spin_values = self.state.wheelSpinVel or [0] * 4
        telemetry_row.extend(wheel_spin_values)
        
        self.telemetry_writer.writerow(telemetry_row)
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    # def gear(self):
    #     rpm = self.state.getRpm()
    #     gear = self.state.getGear()
        
    #     if self.prev_rpm == None:
    #         up = True
    #     else:
    #         if (self.prev_rpm - rpm) < 0:
    #             up = True
    #         else:
    #             up = False
        
    #     if up and rpm > 7000:
    #         gear += 1
        
    #     if not up and rpm < 3000:
    #         gear -= 1
        
    #     self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        current_rpm = self.state.getRpm()
        current_gear = self.state.getGear()
        
        target_speed = self.max_speed  # Can be adjusted dynamically
        
        # Determine acceleration strategy based on current conditions
        if speed < target_speed:
            # Progressive acceleration
            if current_rpm < 7000:
                # Gradual acceleration, taking into account current RPM
                accel_increment = 0.05 + (current_rpm / 7000) * 0.1
                new_accel = self.control.getAccel() + accel_increment
                new_accel = min(new_accel, 1.0)  # Cap at 1.0
            else:
                # At high RPM, maintain or slightly reduce acceleration
                new_accel = 0.9
        else:
            # Speed is at or above target, gradually reduce acceleration
            new_accel = max(0.3, self.control.getAccel() - 0.1)
        
        # Additional speed control considerations
        if speed > target_speed * 1.1:  # If significantly over target speed
            new_accel = 0.0
            self.control.setBrake(0.3)  # Light braking
        else:
            self.control.setBrake(0.0)
        
        self.control.setAccel(new_accel)
        
        #gear shifting based on speed and RPM
        if current_rpm > 7500 and current_gear < 6:
            self.control.setGear(current_gear + 1)
        elif current_rpm < 3000 and current_gear > 1:
            self.control.setGear(current_gear - 1)
            
    def onShutDown(self):
        '''Close telemetry logger on shutdown'''
        self.telemetry_logger.close()
        print(f"Telemetry data saved to telemetry_logs/persistent_telemetry.csv")
    
    def onRestart(self):
        '''Start a new episode in the telemetry logger'''
        self.telemetry_logger.start_new_episode()