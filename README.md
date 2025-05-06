# TORCS ML Car Controller

## Introduction

This project is about building a Machine Learning-based car racing controller using **TORCS (The Open Racing Car Simulator)**. The objective is to design a controller that can race and win against other cars on different tracks. The controller uses telemetry data collected from the simulator to learn driving actions through a Neural Network.

## TORCS Architecture

- **Client-Server Model:** Bots run as external processes and connect to the TORCS race server via UDP.
- **Real-time Processing:**
  - Every 20ms simulated time (a game tick), the server sends sensor data.
  - Bots have 10ms real time to respond with actions.
- **Abstraction Layer:** Separates driver code and server.
  - Programming language freedom (Python, Java, C++, C# clients available).
  - Controlled access to game information.

---

## Objective

Design and implement an ML-based controller that:

- Wins races by completing tracks fastest.
- Balances speed, obstacle avoidance, and track following.
- Uses telemetry sensors data.
- Rule-based controllers are **not** acceptable.

**Simplifications:**

- No noisy sensors.
- No car damage.
- Unlimited fuel.

---

## Project Breakdown

### ✅ 1. Data Collection (Done)

- Collected racing telemetry and saved in CSV.
  - **Inputs:** Sensors (speed, track position, angle, etc.)
  - **Outputs:** Actions (steering, throttle, brake)

### 2. ML & Neural Network

- Teach the car to map sensors ➡️ actions.
- Uses collected data to learn how to drive.
- Use Neural Network model

### Tools & Libraries

Install required packages:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

or better

```bash
pip install -r requirements.txt
```

---

## Get Python Virtual Env to overcome requirement mismatch

### Make sure to install python3.10 before executing the commands below

https://www.python.org/downloads/release/python-31010/

```bash
pip install virtualenv
virtualenv -p python3.10 venv
pip install -r requirements.txt
```

## Visual Workflow

```
CSV Data ➡️ Preprocessing ➡️ Neural Network ➡️ Training ➡️ Testing ➡️ TORCS Client Integration ➡️ Track Test
```

---

# SCRC Python Client

## Usage

To run the client in manual mode, use the following command:

`python pyclient.py --manual`

When running in manual mode, you'll be prompted whether to log telemetry data:
```
Do you want to log telemetry data? (y/n):
```
- Type `y` to enable telemetry logging (saved to a timestamped file in telemetry_logs/)
- Type `n` to disable telemetry logging

### Training with Manual Mode Data

To train a model using only data collected in manual mode:

```bash
python controller/main.py --train --data telemetry_logs/manual_telemetry_YYYYMMDD_HHMMSS.csv --save_scaler --manual_mode_only
```

The `--manual_mode_only` flag ensures only data points marked as collected in manual mode are used for training.

## Important Notes

Close the telemetry CSV file before running TORCS again!

If the CSV file remains open, it may cause issues with logging new data.

Requirements

Ensure you have the required dependencies installed. If necessary, install them using:

`pip install -r requirements.txt`

---

## 📦 Submission Checklist

- Telemetry Data (CSV) [Done]
- ML-based client code
- Self-contained archive with all code and execution files
- 2-page report explaining method

---

## 🔥 Notes

- Avoid rule-based logic; use ML predictions.
- Performance metric = race completion time.

---
