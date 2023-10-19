import numpy as np
states = [(0, 0), (0, 1), (0, 2),
          (1, 0), (1, 1), (1, 2),
          (2, 0), (2, 1), (2, 2)]
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
def transition(state, action):
    if state in states:
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state in states:
            return new_state
    return state
rewards = {
    (0, 0): -1,
    (0, 1): -1,
    (0, 2): -1,
    (1, 0): -1,
    (1, 1): -1,
    (1, 2): -1,
    (2, 0): -1,
    (2, 1): -1,
    (2, 2): 1,
}
gamma = 0.9
policy = {
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'U',
    (1, 0): 'R',
    (1, 1): 'R',
    (1, 2): 'U',
    (2, 0): 'R',
    (2, 1): 'R',
    (2, 2): 'U',
}
V={state: 0 for state in states}
while True:
    delta = 0
    for state in states:
        if state not in policy:
            continue
        v = V[state]
        action = policy[state]
        next_state = transition(state, actions[action])
        reward = rewards[next_state]
        V[state] = reward + gamma * V[next_state]
        delta = max(delta, abs(v - V[state]))
    if delta < 1e-6:
        break
for i in range(3):
    for j in range(3):
        state = (i, j)
        print(f"State {state}: Value = {V[state]:.2f}")
