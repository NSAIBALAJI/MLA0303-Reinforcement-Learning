import random 
class MarioGame: 
def __init__(self): 
self.state = 0 
self.actions = ['move_left', 'move_right', 'jump'] 
self.current_level = 1 
self.is_game_over = False 
self.max_state = 10  # Define the number of states (levels) 
def reset(self): 
self.state = 0 
self.current_level = 1 
self.is_game_over = False 
def step(self, action): 
if self.is_game_over: 
return 0, True 
if action == 'move_left': 
self.state -= 1 
elif action == 'move_right': 
self.state += 1 
elif action == 'jump': 
if self.state == 2: 
self.state = 3  # Advance to the next level 
self.current_level += 1 
if self.current_level > self.max_state: 
self.is_game_over = True 
if self.state < 0: 
self.state = 0 
elif self.state >= self.max_state: 
self.state = self.max_state - 1 
return -1, self.is_game_over  # Always return a negative reward 
class QLearningAgent: 
def __init__(self, n_actions): 
self.q_table = {} 
self.epsilon = 0.1 
self.alpha = 0.1 
self.gamma = 0.9 
self.n_actions = n_actions 
def choose_action(self, state): 
if random.uniform(0, 1) < self.epsilon: 
return random.choice(range(self.n_actions)) 
else: 
if state in self.q_table: 
return max(range(self.n_actions), key=lambda action: 
self.q_table[state].get(action, 0)) 
else: 
return random.choice(range(self.n_actions)) 
def learn(self, state, action, reward, next_state): 
if state not in self.q_table: 
self.q_table[state] = {} 
if next_state not in self.q_table: 
self.q_table[next_state] = {} 
if state not in self.q_table[next_state]: 
self.q_table[next_state][state] = 0 
if state in self.q_table and action in self.q_table[state]: 
max_next_q = max(self.q_table[next_state].values()) 
self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q - 
self.q_table[state][action]) 
if __name__ == '__main__': 
game = MarioGame() 
agent = QLearningAgent(len(game.actions)) 
num_episodes = 1000 
for _ in range(num_episodes): 
game.reset() 
state = game.state 
total_reward = 0 
while not game.is_game_over: 
action = agent.choose_action(state) 
reward, done = game.step(game.actions[action]) 
next_state = game.state 
agent.learn(state, action, reward, next_state) 
state = next_state 
total_reward += reward 
print(f"Episode {num_episodes}, Total Reward: {total_reward}") 
