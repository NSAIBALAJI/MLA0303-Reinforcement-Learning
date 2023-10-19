import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
env = gym.make('CartPole-v1')
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)
class Agent:
    def __init__(self, input_size, output_size):
        self.policy_network = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probabilities = self.policy_network(state)
        action = torch.multinomial(probabilities, 1)
        return action.item()
agent = Agent(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
num_episodes = 1000
for episode in range(num_episodes):
    state, episode_reward = env.reset(), 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.optimizer.zero_grad()
        state = torch.from_numpy(state).float()
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        log_prob = torch.log(agent.policy_network(state)[action])
        loss = -log_prob * reward
        loss.backward()
        agent.optimizer.step()
        episode_reward += reward
        state = next_state
        if done:
            break
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")
env.close()
