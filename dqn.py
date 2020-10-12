# Use weights and biases for hyperparameter tuning
import wandb
from wandb.keras import WandbCallback

wandb.init(project="DQN")
config = wandb.config 
config.epsilon_decay = .996

import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import tensorflow as tf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines.common.env_checker import check_env

import numpy as np
from ads_utils import load_data, plot, Environment

from tqdm import tqdm

import numpy as np
env = Environment(data.close.to_list(), INITIAL_BALANCE)
env.seed(0)
np.random.seed(0)

class DQN:

    """ Implementation of deep q learning algorithm """

    ef __init__(self, action_space, state_space, γ=1.0, ε_decay=1.0):

        self.action_space = action_space
        self.state_space = state_space
        self.ε = 0.01
        self.γ = γ
        self.batch_size = 32
        self.ε_min = 0.01
        self.lr = 0.001
        self.ε_decay = ε_decay
        self.memory = deque(maxlen=10_000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.state_space, activation=relu))
        model.add(Dense(100, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, exploit=False):

        if np.random.rand() + int(exploit) <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states      = np.array([i[0] for i in batch])
        actions     = np.array([i[1] for i in batch])
        rewards     = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones       = np.array([i[4] for i in batch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.γ*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0, callbacks=[WandbCallback()])
        if self.ε > self.ε_min:
            self.ε *= self.ε_decay


def train_dqn(agent, env, episodes):

    rewards = []
    for e in tqdm(range(episodes)):
        state = env.reset(rand_start=True)
        state = np.reshape(state, (1, 6))
        score = 0
        max_steps = 1_000
        for i in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 6))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay()
                
            if done:
                break
        rewards.append(score)
        
        # You can log whatever attributes you want and see a plot in W&B!
        wandb.log({'score': score})

    return rewards

from tqdm import tqdm
fake = np.linspace(100, 200, len(data))
data['close'] = fake

env = Environment(data.close.to_list(), balance=INITIAL_BALANCE, past_ticks=PAST_TICKS)
episodes = 50
agent = DQN(env.action_space.n, env.observation_space.shape[0])
loss = train_dqn(agent, env, episodes=30)
plt.plot(np.arange(0, len(loss)), loss)
plt.show()


state = env.reset()
state = np.reshape(state, (1, 6))

portfolio_values = []
prices = []
actions = []
n_ticks = 10000
for i in tqdm(range(n_ticks)):
    action = agent.get_action(state, exploit=True)
    
    price, portfolio_value = env.get_data()
    actions.append(action)
    prices.append(price)
    portfolio_values.append(portfolio_value)
    
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, (1, 6))
    state = next_state
    
    if done:
        break

plot(prices, actions, portfolio_values, right_y_adjust=1.1)