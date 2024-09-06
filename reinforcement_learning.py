import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lstm_model import LSTMStockPredictor
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data, lstm_predictor, scaler):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.lstm_predictor = lstm_predictor
        self.scaler = scaler
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        obs = self.get_observation()
        return obs

    def get_observation(self):
        if self.current_step < len(self.data):
            stock_obs = self.data.iloc[self.current_step].values.astype(np.float32)
            return stock_obs
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)

        prev_close = self.data.iloc[self.current_step - 1]['Close'] if self.current_step > 0 else self.data.iloc[self.current_step]['Close']
        current_close = self.data.iloc[self.current_step]['Close']

        lstm_prediction = self.lstm_predictor.predict(self.data.iloc[self.current_step:self.current_step + self.lstm_predictor.look_back], self.scaler)

        reward = 0
        if action == 0:
            reward = current_close - prev_close
        elif action == 2:
            reward = prev_close - current_close

        prediction_error = abs(lstm_prediction - current_close)
        reward -= prediction_error

        obs = self.get_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

def train_reinforcement_learning_model(data, lstm_predictor, scaler):
    env = DummyVecEnv([lambda: StockTradingEnv(data, lstm_predictor, scaler)])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001)
    model.learn(total_timesteps=10000)
    model.save("stock_trading_model")

def load_reinforcement_learning_model(filename):
    model = PPO.load(filename)
    return model
