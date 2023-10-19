import numpy as np 
Q = np.zeros((num_states, num_actions)) 
epsilon = 0.1 
alpha = 0.1 
gamma = 0.9 
num_episodes = 1000 
for _ in range(num_episodes): 
state = initial_state 
while not reached_destination: 
if random.uniform(0, 1) < epsilon: 
action = random.choice(possible_actions) 
else: 
action = np.argmax(Q[state, :]) 
next_state, reward = take_action(action)  # Simulate the traveler's action 
Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * 
np.max(Q[next_state, :])) 
state = next_state 
state = initial_state 
optimal_path = [state] 
while not reached_destination: 
action = np.argmax(Q[state, :]) 
next_state, _ = take_action(action) 
state = next_state 
optimal_path.append(state) 
print("Optimal Path:", optimal_path)
