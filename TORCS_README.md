
# TORCS ML Car Controller

## 📚 Introduction
This project is about building a Machine Learning-based car racing controller using **TORCS (The Open Racing Car Simulator)**. The objective is to design a controller that can race and win against other cars on different tracks. The controller uses telemetry data collected from the simulator to learn driving actions through a Neural Network.

---

## 🚗 TORCS Architecture
- **Client-Server Model:** Bots run as external processes and connect to the TORCS race server via UDP.
- **Real-time Processing:** 
  - Every 20ms simulated time (a game tick), the server sends sensor data.
  - Bots have 10ms real time to respond with actions.
- **Abstraction Layer:** Separates driver code and server.
  - Programming language freedom (Python, Java, C++, C# clients available).
  - Controlled access to game information.

---

## 🎯 Objective
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

## 🧩 Project Breakdown

### ✅ 1. Data Collection (Done)
- Collected racing telemetry and saved in CSV.
  - **Inputs:** Sensors (speed, track position, angle, etc.)
  - **Outputs:** Actions (steering, throttle, brake)


### ✅ 2. ML & Neural Network

#### What is ML here?
- Teach the car to map sensors ➡️ actions.
- Uses collected data to learn how to drive.

#### What is Neural Network?
- A machine learning model with layers (neurons) that learns patterns.
- Input: Sensor values.
- Output: Actions (steer, throttle, brake).


### ✅ 3. Tools & Libraries
Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```


### ✅ 4. Steps to Build the Controller

#### Step 1: Prepare Data
- Load CSV.
- Ensure columns for sensors (inputs) and actions (outputs).

#### Step 2: Preprocess Data
- Normalize sensor values.
- Split into training (80%) and testing (20%).

#### Step 3: Build Neural Network (with Keras)
- Input layer matching sensor count.
- 2-3 hidden layers (e.g., 64-128 neurons).
- Output layer (3 neurons) ➡️ steer, throttle, brake.

#### Step 4: Train the Model
- Use Mean Squared Error (MSE) loss.
- Train until loss reduces.

#### Step 5: Test the Model
- Check accuracy on test data.
- Plot loss curves.

#### Step 6: Integrate with TORCS
- Load trained model in Python TORCS client.
- Replace manual action code with model predictions.

#### Step 7: Test on Track
- Run TORCS.
- See car drive with ML controller.
- Tune and retrain for better performance.

---

## 🚀 Visual Workflow
```
CSV Data ➡️ Preprocessing ➡️ Neural Network ➡️ Training ➡️ Testing ➡️ TORCS Client Integration ➡️ Track Test
```

---

## 📦 Submission Checklist
- ✅ Telemetry Data (CSV) [Done]
- ✅ ML-based client code
- ✅ Self-contained archive with all code and execution files
- ✅ 2-page report explaining method

---

## 🔥 Notes
- You are using the **Python TORCS client**.
- Avoid rule-based logic; use ML predictions.
- Performance metric = race completion time.

---

Happy Racing! 🏎️💨
