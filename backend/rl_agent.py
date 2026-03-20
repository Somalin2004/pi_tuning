import numpy as np

Q = np.zeros((10,10))

alpha = 0.1
gamma = 0.9

def train_rl():
    for episode in range(500):
        state = np.random.randint(0,10)

        for step in range(20):
            action = np.argmax(Q[state])

            reward = -abs(state - action)

            next_state = np.random.randint(0,10)

            Q[state, action] += alpha * (reward + gamma*np.max(Q[next_state]) - Q[state,action])
            state = next_state

    return Q

if __name__ == "__main__":
    q = train_rl()
    print("RL trained")