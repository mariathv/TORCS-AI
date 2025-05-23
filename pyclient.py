import sys
import argparse
import socket
import driver
import time

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=2,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--manual', action='store_true', dest='manual_mode',
                    help='Enable manual control mode')
parser.add_argument('--model', action='store', dest='model_path', default='controller/model',
                    help='Path to the trained ML model (default: controller/model)')


arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('Manual Mode:', arguments.manual_mode)
if not arguments.manual_mode:
    print('ML Model:', arguments.model_path)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# One second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = False

# Pass manual mode flag and model path to Driver constructor
d = driver.Driver(
    arguments.stage, 
    manual_mode=arguments.manual_mode,
    max_episodes=arguments.max_episodes,
    model_coordinator=arguments.model_path
)

if arguments.manual_mode:
    print("Manual Control Mode:")
    print("- Arrow keys: Steer and Accelerate/Brake")
    print("- Page Up: Shift Up")
    print("- Page Down: Shift Down")

while not shutdownClient:
    while True:
        print('Sending id to server:', arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:', buf)
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.timeout:
            print("Timeout waiting for server response. Retrying...")
            continue
        except socket.error as msg:
            print("Didn't get response from server...")
            continue

        if '***identified***' in buf:
            print('Received:', buf)
            break

    currentStep = 0
    
    while True:
        # Wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.timeout:
            print("Timeout waiting for server response. Skipping...")
            continue
        except socket.error as msg:
            print(f"Didn't get response from server... Socket error: {msg}")
            continue
        
        if verbose:
            print('Received:', buf)
        
        if buf is not None and '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf is not None and '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf is not None:
                start_drive = time.time()
                buf = d.drive(buf)
                drive_time = (time.time() - start_drive) * 1000
                if drive_time > 8:
                    print(f"[WARN] drive() took {drive_time:.2f} ms (step {currentStep})")
        else:
            buf = '(meta 1)'
        
        if verbose:
            print('Sending:', buf)
        
        if buf is not None:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True
        
sock.close()