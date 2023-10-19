import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from collections import deque 
import random 
import gym 
class ReplayBuffer: 
def __init__(self, max_size): 
self.buffer = deque(maxlen=max_size) 
def add(self, experience): 
self.buffer.append(experience) 
def sample(self, batch_size): 
batch = random.sample(self.buffer, batch_size) 
states, actions, rewards, next_states, dones = zip(*batch) 
return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), 
np.array(dones) 
def build_dqn_model(input_shape, num_actions): 
model = keras.Sequential([ 
keras.layers.Dense(24, activation='relu', input_shape=input_shape), 
keras.layers.Dense(24, activation='relu'), 
keras.layers.Dense(num_actions, activation='linear') 
]) 
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
return model  
class DDQNAgent: 
def __init__(self, state_size, action_size): 
self.state_size = state_size 
        self.action_size = action_size 
        self.target_update_frequency = 1000  
        self.dqn = build_dqn_model(state_size, action_size) 
        self.target_dqn = build_dqn_model(state_size, action_size) 
        self.target_dqn.set_weights(self.dqn.get_weights()) 
        self.replay_buffer = ReplayBuffer(max_size=2000) 
        self.batch_size = 32 
        self.gamma = 0.99  
        self.epsilon = 1.0 
        self.min_epsilon = 0.01 
        self.epsilon_decay = 0.995  
    def select_action(self, state): 
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size) 
        q_values = self.dqn.predict(state) 
        return np.argmax(q_values[0]) 
    def train(self): 
        if len(self.replay_buffer.buffer) < self.batch_size: 
            return 
        states, actions, rewards, next_states, dones = 
self.replay_buffer.sample(self.batch_size) 
        targets = self.dqn.predict(states) 
        target_values = self.target_dqn.predict(next_states) 
        for i in range(self.batch_size): 
            if dones[i]: 
                targets[i][actions[i]] = rewards[i] 
            else: 
                best_action = np.argmax(self.dqn.predict(next_states[i:i+1])[0]) 
                targets[i][actions[i]] = rewards[i] + self.gamma * 
target_values[i][best_action] 
        self.dqn.fit(states, targets, epochs=1, verbose=0) 
        if self.epsilon > self.min_epsilon: 
            self.epsilon *= self.epsilon_decay 
        if self.total_steps % self.target_update_frequency == 0: 
            self.target_dqn.set_weights(self.dqn.get_weights()) 
    def remember(self, state, action, reward, next_state, done): 
        self.replay_buffer.add((state, action, reward, next_state, done)) 
    def load(self, name): 
        self.dqn.load_weights(name) 
    def save(self, name): 
        self.dqn.save_weights(name) 
def train_ddqn_agent(): 
    env = gym.make("CartPole-v1") 
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n 
    agent = DDQNAgent(state_size, action_size) 
    episodes = 1000 
    for episode in range(episodes): 
        state = env.reset() 
        state = np.reshape(state, [1, state_size]) 
        done = False 
        for time in range(500):
            action = agent.select_action(state) 
            next_state, reward, done, _ = env.step(action) 
            next_state = np.reshape(next_state, [1, state_size]) 
            agent.remember(state, action, reward, next_state, done) 
            state = next_state 
            if done: 
                break 
            agent.train() 
        if episode % 10 == 0: 
            print("Episode: {}/{}, Total Steps: {}, Epsilon: {:.2}".format( 
                episode, episodes, agent.total_steps, agent.epsilon)) 
    agent.save("ddqn_model.h5") 
if __name__ == "__main__": 
    train_ddqn_agent() 
