import msgParser
import carState
import carControl
import csv
import os
from datetime import datetime
import keyboard  

class Driver(object):
    '''
    A driver object for the SCRC with manual control and telemetry logging
    '''

    def __init__(self, stage, manual_mode=False):
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
        self.max_speed = 150
        self.prev_rpm = None

        self.manual_mode = manual_mode
        
        # -------------------------- TELEMETERY LOGGING ----------------------------- # 
        self.telemetry_dir = 'telemetry_logs'
        os.makedirs(self.telemetry_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_file = os.path.join(self.telemetry_dir, f'telemetry_{timestamp}.csv')
        
        self.telemetry_file_handle = open(self.telemetry_file, 'w', newline='')
        self.telemetry_writer = csv.writer(self.telemetry_file_handle)
        
        headers = [
            'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 
            'fuel', 'gear', 'lastLapTime', 'racePos', 'rpm', 
            'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
            'accel', 'brake', 'steer', 'clutch'
        ]
        
        headers.extend([f'track_{i}' for i in range(19)])
        headers.extend([f'wheelSpinVel_{i}' for i in range(4)])
        
        self.telemetry_writer.writerow(headers)
    
    def __del__(self):
        '''Destructor to ensure file is closed'''
        if hasattr(self, 'telemetry_file_handle'):
            self.telemetry_file_handle.close()
    
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
        
        # Log telemetry data
        self._log_telemetry()
        
        return self.control.toMsg()
    
    def manual_control(self):
        '''Manual control method using keyboard input'''
        # Steering
        if keyboard.is_pressed('right'):
            self.control.setSteer(-1.0)  # Full left
        elif keyboard.is_pressed('left'):
            self.control.setSteer(1.0)   # Full right
        else:
            self.control.setSteer(0)     # No steering
        
        if keyboard.is_pressed('up'):
            self.control.setAccel(1.0)   # Full acceleration
            self.control.setBrake(0)     # No braking
            self.control.setGear(1)      # Ensure forward gear
        elif keyboard.is_pressed('down'):
            current_speed = self.state.getSpeedX()
            current_gear = self.state.getGear()
            
            if current_speed <= 0 and current_gear > 0:
                self.control.setAccel(0)
                self.control.setBrake(1.0)
                self.control.setGear(-1)  # Shift to reverse
            elif current_gear == -1:
                self.control.setAccel(1.0)
                self.control.setBrake(0)
            else:
                self.control.setAccel(0)
                self.control.setBrake(1.0)
        else:
            self.control.setAccel(0)     # No acceleration
            self.control.setBrake(0)     # No braking
    
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
    
    def gear(self):
        rpm = self.state.getRpm()
        current_gear = self.state.getGear()
        speed = self.state.getSpeedX()
        
        if current_gear == 1:
            if speed > 30 or rpm > 5000:
                print("change gear > 2")
                current_gear = 2
        elif current_gear == 2:
            if speed > 60 or rpm > 6500:
                current_gear = 3
        elif current_gear == 3:
            if speed > 90 or rpm > 7000:
                current_gear = 4
        elif current_gear == 4:
            if speed > 120 or rpm > 7500:
                current_gear = 5
        elif current_gear == 5:
            if speed > 150 or rpm > 7800:
                current_gear = 6
        
        if current_gear > 1:
            if rpm < 3000:
                current_gear -= 1
        
        current_gear = max(1, min(current_gear, 6))
        
        self.control.setGear(current_gear)
        
        self.prev_rpm = rpm
    
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
        '''Close telemetry file on shutdown'''
        if hasattr(self, 'telemetry_file_handle'):
            self.telemetry_file_handle.close()
        print(f"Telemetry data saved to {self.telemetry_file}")
    
    def onRestart(self):
        pass