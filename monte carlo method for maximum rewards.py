import random 
states = [0, 1, 2, 3, 4] 
actions = ['left', 'right'] 
rewards = { 
(0, 'left', 0): -1, 
(0, 'right', 1): 5, 
(1, 'left', 0): -1, 
(1, 'right', 0): 2, 
(2, 'left', 0): -1, 
(2, 'right', 0): 0, 
(3, 'left', 0): -1, 
(3, 'right', 1): 10, 
(4, 'left', 0): -1, 
(4, 'right', 0): -1, 
} 
Q = {(state, action): 0 for state in states for action in actions} 
epsilon = 0.1 
gamma = 0.9 
num_episodes = 1000 
for _ in range(num_episodes): 
episode = [] 
state = random.choice(states) 
while True: 
if random.uniform(0, 1) < epsilon: 
action = random.choice(actions)  # Exploration 
else: 
action = max(actions, key=lambda a: Q[(state, a)])  # Exploitation 
next_state = state + (1 if action == 'right' else -1) 
reward = rewards.get((state, action, 0), 0) 
episode.append((state, action, reward)) 
state = next_state 
if state not in states: 
break 
G = 0 
for i, (state, action, reward) in enumerate(reversed(episode)): 
G = gamma * G + reward 
Q[(state, action)] = Q[(state, action)] + 0.1 * (G - Q[(state, action)]) 
optimal_policy = {} 
for state in states: 
optimal_action = max(actions, key=lambda a: Q[(state, a)]) 
optimal_policy[state] = optimal_action 
for state, action in optimal_policy.items(): 
print(f"State {state}: Take action '{action}'")
