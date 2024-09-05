import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)  # Updated shape

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values.astype(np.float32)  # Ensure dtype matches observation_space

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.current_step = 0
        else:
            done = False

        if self.current_step == 0:
            reward = 0  # No previous close price to compare
        else:
            prev_close = self.data.iloc[self.current_step - 1]['Close']
            current_close = self.data.iloc[self.current_step]['Close']

            if action == 0:  # Buy
                reward = current_close - prev_close
            elif action == 2:  # Sell
                reward = prev_close - current_close
            else:  # Hold
                reward = 0

        obs = self.data.iloc[self.current_step].values.astype(np.float32)  # Ensure dtype matches observation_space
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

def train_reinforcement_learning_model(data):
    env = DummyVecEnv([lambda: StockTradingEnv(data)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("stock_trading_model")  # Save model without extra extension

def load_reinforcement_learning_model(filename):
    model = PPO.load(filename)
    return model
