import numpy as np

class ReinforcementLearning:
    """
    Reinforcement Learning model for fine-tuning stock predictions.
    """
    def __init__(self):
        print("Initializing reinforcement learning model")
        self.q_table = {}

    def get_state(self, price):
        return round(float(price), 2)  # Simplified state representation, ensuring price is a float

    def choose_action(self, state):
        """
        Choose an action based on the current state.
        """
        if state not in self.q_table:
            return np.random.choice(['buy', 'sell', 'hold'])
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward):
        """
        Update the Q-table with the given state, action, and reward.
        """
        if state not in self.q_table:
            self.q_table[state] = {'buy': 0, 'sell': 0, 'hold': 0}
        self.q_table[state][action] = reward

    def train(self, stock_data):
        """
        Train the reinforcement learning model on stock data.
        """
        print("Starting reinforcement learning model training")
        for ticker, data in stock_data.items():
            print(f"Training on data for {ticker}")
            prices = data['Close'].values
            for i in range(1, len(prices)):
                state = self.get_state(prices[i-1])
                action = self.choose_action(state)
                reward = prices[i] - prices[i-1]  # Simple reward based on price change
                self.update_q_table(state, action, reward)
            print(f"Finished training on {ticker}")

    def make_predictions(self, lstm_predictions):
        """
        Make predictions based on LSTM predictions, fine-tuned with reinforcement learning.
        """
        print("Fine-tuning LSTM predictions with reinforcement learning")
        rl_predictions = []
        
        # Iterate through LSTM predictions (numpy array)
        for price in lstm_predictions:
            state = self.get_state(price)  # Ensure price is a float
            action = self.choose_action(state)
            rl_predictions.append((price, action))
        
        print("Reinforcement learning predictions complete")
        return rl_predictions
