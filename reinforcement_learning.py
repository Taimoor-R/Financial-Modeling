import numpy as np
import gym
from stable_baselines3 import PPO

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.current_step = 0
        else:
            done = False
        
        reward = 0
        if action == 0:  # Buy
            reward = self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step-1]['Close']
        elif action == 2:  # Sell
            reward = self.data.iloc[self.current_step-1]['Close'] - self.data.iloc[self.current_step]['Close']
        
        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

def train_reinforcement_learning_model(data):
    env = StockTradingEnv(data)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("stock_trading_model")

def load_reinforcement_learning_model(filename):
    model = PPO.load(filename)
    return model
