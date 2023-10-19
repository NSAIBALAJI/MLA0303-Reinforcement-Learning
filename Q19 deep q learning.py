import numpy as np 
grid_world = np.array([ 
['S', 'E', 'E', 'E'], 
['E', 'O', 'E', 'O'], 
['E', 'E', 'E', 'E'], 
['O', 'E', 'E', 'G'] 
]) 
class QLearningAgent: 
def __init__(self, num_states, num_actions, epsilon=0.1, alpha=0.1, gamma=0.9): 
self.epsilon = epsilon 
self.alpha = alpha 
self.gamma = gamma 
self.q_table = np.zeros((num_states, num_actions)) 
def choose_action(self, state): 
if np.random.rand() < self.epsilon: 
return np.random.randint(self.q_table.shape[1]) 
else: 
return np.argmax(self.q_table[state, :]) 
def update_q_table(self, state, action, reward, next_state): 
predict = self.q_table[state, action] 
target = reward + self.gamma * np.max(self.q_table[next_state, :]) 
self.q_table[state, action] += self.alpha * (target - predict) 
def grid_to_states(grid): 
return [s for s in grid.reshape(-1) if s != 'O'] 
def find_state_index(grid, state): 
return np.where(grid.reshape(-1) == state)[0][0] 
states = grid_to_states(grid_world) 
num_states = len(states) 
num_actions = 4  # Up, Down, Left, Right 
q_agent = QLearningAgent(num_states, num_actions) 
def train_q_learning_agent(agent, grid, goal_state, episodes): 
for episode in range(episodes): 
current_state = 'S' 
while current_state != goal_state: 
state_index = find_state_index(grid, current_state) 
action = agent.choose_action(state_index) 
if action == 0: 
next_state = grid[state_index - 4] 
elif action == 1:
next_state = grid[state_index + 4] 
elif action == 2: 
next_state = grid[state_index - 1] 
else: 
next_state = grid[state_index + 1] 
reward = -1 if next_state != 'O' else -100  # Negative reward for obstacles 
next_state_index = find_state_index(grid, next_state) 
agent.update_q_table(state_index, action, reward, next_state_index) 
current_state = next_state 
train_q_learning_agent(q_agent, states, 'G', episodes=500) 
print("Learned Q-Values:") 
print(q_agent.q_table)
