import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    def __init__(self, data, sentiment_scores):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.sentiment_scores = sentiment_scores
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 1,), dtype=np.float32)  # Including sentiment

    def reset(self):
        self.current_step = 0
        obs = self.get_observation()
        return obs

    def get_observation(self):
        # Ensure indices are valid
        if self.current_step < len(self.data):
            stock_obs = self.data.iloc[self.current_step].values.astype(np.float32)
            sentiment_obs = np.array([self.sentiment_scores[self.current_step]], dtype=np.float32)
            return np.concatenate([stock_obs, sentiment_obs])
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.current_step = 0
        else:
            done = False

        prev_close = self.data.iloc[self.current_step - 1]['Close'] if self.current_step > 0 else self.data.iloc[self.current_step]['Close']
        current_close = self.data.iloc[self.current_step]['Close']
        sentiment = self.sentiment_scores[self.current_step] if self.current_step < len(self.sentiment_scores) else 0
        
        reward = 0
        if self.current_step > 0:
            if action == 0:  # Buy
                reward = current_close - prev_close
            elif action == 2:  # Sell
                reward = prev_close - current_close
            else:  # Hold
                reward = 0

        obs = self.get_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

def train_reinforcement_learning_model(data, sentiment_scores):
    env = DummyVecEnv([lambda: StockTradingEnv(data, sentiment_scores)])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001)  # Adjust learning rate as needed
    model.learn(total_timesteps=10000)
    model.save("stock_trading_model")  # Save model without extra extension

def load_reinforcement_learning_model(filename):
    model = PPO.load(filename)
    return model
