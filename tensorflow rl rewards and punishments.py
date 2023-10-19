import gym
from stable_baselines import PPO2
class CustomEnv(gym.Env):
  def __init__(self):
   super(CustomEnv, self).__init__()
   self.observation_space = gym.spaces.Discrete(3)
   self.action_space = gym.spaces.Discrete(2)
   self.state = 0
   self.max_steps = 5
   self.current_step = 0
  def step(self, action):
   if self.current_step >= self.max_steps:
     done = True
   else:
     done = False
   if action == 0:
     reward = 1
   else:
     reward = -1
   self.current_step += 1
   self.state += 1
   return self.state, reward, done, {}
  def reset(self):
   self.current_step = 0
   self.state = 0
   return self.state
env = CustomEnv()
model = PPO2("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
obs = env.reset()
total_reward = 0
for _ in range(5):
   action, _ = model.predict(obs)
   obs, reward, done, _ = env.step(action)
   total_reward += reward
   if done:
     break
print(f"Total Reward: {total_reward}")
