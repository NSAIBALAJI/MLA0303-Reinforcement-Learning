import numpy as np 
q_values_a = np.zeros((num_states, num_actions)) 
q_values_b = np.zeros((num_states, num_actions)) 
q_values_c = np.zeros((num_states, num_actions)) 
epsilon = 0.1 
alpha = 0.1 
gamma = 0.9 
num_episodes = 1000 
for episode in range(num_episodes): 
state = initial_state 
total_reward_a = 0 
total_reward_b = 0 
total_reward_c = 0 
while not reached_goal:  
action_a = select_action_td0(q_values_a, state, epsilon) 
action_b = select_action_sarsa(q_values_b, state, epsilon) 
action_c = select_action_qlearning(q_values_c, state, epsilon) 
state = next_state 
total_reward_a += reward 
total_reward_b += reward 
total_reward_c += reward 
