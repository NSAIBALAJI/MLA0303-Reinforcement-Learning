import numpy as np 
n_articles = 10 
n_states = 5  
Q = np.zeros((n_states, n_articles)) 
rewards = np.random.rand(n_states, n_articles) 
alpha = 0.1  # Learning rate 
gamma = 0.9  # Discount factor 
num_episodes = 1000 
for _ in range(num_episodes): 
state = np.random.randint(n_states)  # Random initial state 
while True: 
action = np.argmax(Q[state, :])  # Exploitation: Select the action with the highest 
Q-value 
next_state = np.random.randint(n_states)  # Simulate user moving to a new state 
reward = rewards[state, action] 
Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, 
action]) 
state = next_state 
if np.random.rand() < 0.1:  # Simulate the end of an episode with 10% 
probability 
break 
optimal_policy = np.argmax(Q, axis=1) 
print("Optimal Policy:") 
print(optimal_policy) 
