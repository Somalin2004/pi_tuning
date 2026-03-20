import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

N = 50000
data = []

def simulate(Kp, Ki, tau, K):
    dt = 0.01
    y = 0
    integral = 0
    total_error = 0

    for _ in range(1000):
        error = 1 - y
        integral += error * dt
        u = Kp * error + Ki * integral
        y += (-y + K*u)/tau * dt
        total_error += error**2

    return total_error

# Generate dataset
for _ in range(N):
    tau = np.random.uniform(0.5, 5)
    K = np.random.uniform(0.5, 2)

    best_score = 1e9
    best_Kp, best_Ki = 0, 0

    for _ in range(10):
        Kp = np.random.uniform(0, 10)
        Ki = np.random.uniform(0, 5)

        score = simulate(Kp, Ki, tau, K)

        if score < best_score:
            best_score = score
            best_Kp, best_Ki = Kp, Ki

    data.append([tau, K, best_Kp, best_Ki])

df = pd.DataFrame(data, columns=["tau","K","Kp","Ki"])

# Train ML model
X = df[["tau","K"]]
y = df[["Kp","Ki"]]

model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

pickle.dump(model, open("model.pkl","wb"))

# RL Model (Q-table)
Q = np.zeros((10,10))

for episode in range(500):
    state = np.random.randint(0,10)

    for _ in range(20):
        action = np.argmax(Q[state])
        reward = -abs(state - action)
        next_state = np.random.randint(0,10)

        Q[state, action] += 0.1 * (reward + 0.9*np.max(Q[next_state]) - Q[state,action])
        state = next_state

np.save("rl_model.npy", Q)

print("Training complete!")