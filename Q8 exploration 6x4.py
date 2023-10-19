import numpy as np
import random
n_rows, n_cols = 6, 4
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
epsilon = 0.2
state_values = np.zeros((n_rows, n_cols))
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        next_state_indices = [(state[0] + a[0], state[1] + a[1]) for a in actions]
        for i in range(len(next_state_indices)):
            next_state = (np.clip(next_state[0], 0, n_rows - 1), np.clip(next_state[1], 0, n_cols - 1))
        next_state_values = [state_values[index[0], index[1]] for index in next_state_indices]
        return np.argmax(next_state_values)
num_episodes = 1000
for _ in range(num_episodes):
    current_state = (0, 0)
    while True:
        action = choose_action(current_state)
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])
        if next_state == (5, 3):
            reward = 1
        else:
            reward = 0
        state_values[current_state] += 0.1 * (reward + 0.9 * state_values[next_state] - state_values[current_state])
        current_state = next_state
        if next_state == (5, 3):
            break
print("State Values with Exploration:")
print(state_values)
