import csv
import os
from datetime import datetime
import threading

class TelemetryLogger:
    """
    A dedicated class for managing telemetry logging functionality with enhanced episode tracking
    """
    
    def __init__(self, stage, max_episodes, log_directory='telemetry_logs', filename='telemetry.csv'):
        """
        Initialize the telemetry logger
        
        Args:
            stage (int): Racing stage (Warm-up, Qualifying, Race)
            max_episodes (int): Maximum number of episodes to track
            log_directory (str): Directory to store telemetry logs
            filename (str): Name of the CSV file for logging
        """
        # Store stage and episode information
        self.stage_names = {
            0: 'Warm-Up',
            1: 'Qualifying', 
            2: 'Race', 
            3: 'Unknown'
        }
        self.current_stage = self.stage_names.get(stage, 'Unknown')
        self.max_episodes = max_episodes
        
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)
        
        self.log_file_path = os.path.join(self.log_directory, filename)
        
        # Thread-safe logging
        self._log_lock = threading.Lock()
        
        # headers
        self.headers = [
            'timestamp', 'stage', 'max_episodes', 'current_episode', 'session_id',
            'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 
            'fuel', 'gear', 'lastLapTime', 'racePos', 'rpm', 
            'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
            'accel', 'brake', 'steer', 'clutch'
        ]
        
        # Add track sensor and wheel spin velocity headers
        self.headers.extend([f'track_{i}' for i in range(19)])
        self.headers.extend([f'wheelSpinVel_{i}' for i in range(4)])
        
        # Unique session ID for this logging instance
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Episode counter
        self.current_episode = 0
        
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """
        Create the log file with headers if it doesn't exist
        """
        try:
            file_exists = os.path.isfile(self.log_file_path)
            
            if not file_exists:
                with open(self.log_file_path, 'w', newline='') as f:
                    csv.writer(f).writerow(self.headers)
        except Exception as e:
            print(f"Error initializing log file: {e}")
            self.log_file_path = os.path.join(
                self.log_directory, 
                f'persistent_telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            with open(self.log_file_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.headers)
    
    def log_data(self, state, control):
        """
        Log a single row of telemetry data with error handling
        
        Args:
            state (CarState): Current car state object
            control (CarControl): Current car control object
        """
        # Prepare telemetry row
        telemetry_row = [
            datetime.now().isoformat(),  # Timestamp
            self.current_stage,          # Current stage
            self.max_episodes,           # Maximum episodes
            self.current_episode,        # Current episode number
            self.session_id,             # Session ID
            state.angle, state.curLapTime, state.damage, 
            state.distFromStart, state.distRaced, state.fuel, 
            state.gear, state.lastLapTime, state.racePos, 
            state.rpm, state.speedX, state.speedY, 
            state.speedZ, state.trackPos, state.z,
            control.accel, control.brake, control.steer, 
            control.clutch
        ]
        
        # Add track sensor values (or zeros if None)
        track_values = state.track or [0] * 19
        telemetry_row.extend(track_values)
        
        # Add wheel spin velocity values (or zeros if None)
        wheel_spin_values = state.wheelSpinVel or [0] * 4
        telemetry_row.extend(wheel_spin_values)
        
        # Thread-safe logging with error handling
        try:
            with self._log_lock:
                with open(self.log_file_path, 'a', newline='') as f:
                    csv.writer(f).writerow(telemetry_row)
        except Exception as e:
            print(f"Error logging telemetry: {e}")
    
    def start_new_episode(self):
        """
        Increment episode counter when a new episode begins
        """
        self.current_episode += 1
        print(f"Starting new episode: {self.current_episode}/{self.max_episodes}")
    
    def close(self):
        """
        Close any open file handles (if applicable)
        """
        print(f"Telemetry logging completed. Total episodes: {self.current_episode}")