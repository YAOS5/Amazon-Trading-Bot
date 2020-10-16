import gym
from gym import spaces
from gym.utils import seeding

import pandas as pd
import numpy as np

MAX_SHARES = 1000
INITIAL_BALANCE = 10_000
PAST_TICKS = 5

class Environment_ddpg(gym.Env):  
    # required for stable baselines 
    metadata = {'render.modes': ['human']}
    HOLD = 0
    #SELL, HOLD, BUY = 0, 1, 2
    
    def __init__(self, data, balance=INITIAL_BALANCE, transaction_cost=0.001, i=0, position=0, past_ticks=PAST_TICKS,
                train=False):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            raise ValueError('Only lists or arrays allowed')
        
        self.train = train
        if self.train:
            self.history = []
        
        self.past_ticks = past_ticks
        self.curr_step = self.past_ticks+1
        self.initial_balance = self.balance = balance
        
        self.done = False
        self.data = data
        self.position = position
        self.transaction_cost = transaction_cost
        
        self._seed()
        
        # Amount of Sell, Hold = 0, Buy 
        self.action_space = spaces.Box(low=-MAX_SHARES, high=MAX_SHARES, shape = (1, ))

        # Observation space has past_ticks prices up to and included current price, then position
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.past_ticks+1, ))
        
    def _next_observation(self):        
        '''Getting the next observation'''
        # Convert frame into returns
        # These two lines don't work for the DQN
        frame = np.array(self.data[self.curr_step - self.past_ticks: self.curr_step + 1])###################################
        
        # Testing stuff out
        #frame = np.diff(frame) / frame[:-1] * 100

        if isinstance(self.position, np.ndarray):
            obs = np.append(frame, self.position)
            
        else: 
            obs = np.append(frame, [self.position], axis=0)
        return obs

    def _take_action(self, action):
        curr_price = self.data[self.curr_step]
        
        # Perform position transition (transaction cost is a proportion of price)
        self.balance -= curr_price * self.transaction_cost * abs(action - self.position)
        
        # A Buy 
        if (self.position >= 0) and (action - self.position > 0):
            self.balance -= curr_price * action # amount to buy
            
        # A sell
        elif (self.position <= 0) and (action - self.position < 0):
            self.balance += curr_price * action # amount to sell
        
        # Flip Position
        elif ((self.position >= 0) and (action - self.position < 0)) or ((self.position <= 0) and (action - self.position > 0)):
            self.balance -= 2 * (action) * curr_price
            

        # Update position and time
        self.position = action
        self.curr_step += 1
        
    def step(self, action):
        ''' Updates environment with action taken, returns new state and reward from state transition '''
        
        prior_portfolio_value = self.get_portfolio_value()

        # Take action
        self._take_action(action)
        
        # current portfolio value
        self.portfolio_value = self.get_portfolio_value()
        
        reward = self.portfolio_value - prior_portfolio_value   

        # Percentange change from initial portfolio value
        #reward = 100 * ((self.portfolio_value/self.initial_balance) - 1)  
        
        # Are we done?
        if self.balance <= 0:
            self.done = True
            reward = -1e6
        if self.curr_step >= len(self.data) - 2:
            self.done = True
            
        if self.train:
            self.history.append([self.portfolio_value, self.data[self.curr_step], action])
        
        obs = self._next_observation()
        
        # required to return: observation, reward, done, info
        return obs, reward, self.done, {"history": self.history}
    
    def get_portfolio_value(self):
        ''' Returns current portfolio value
        - please check the logic!'''
        
        curr_price = self.data[self.curr_step]
        
        if self.position > 0:
            # balance + asset
            return self.balance + curr_price * self.position
        
        elif self.position < 0:
            # balance - asset
            return self.balance - curr_price * (-self.position)
        
        return self.balance
    
    def reset(self, rand_start=False):
        '''Reset everything as if we just started (for a new episode)'''
        self.position = self.HOLD
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.done = False
        self.curr_step = np.random.randint(self.past_ticks, 2*len(self.data)//3) if rand_start else self.past_ticks+1
        
        # Must return first observation
        return self._next_observation()   

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def __repr__(self):
        return f'Balance: ${round(self.balance, 2)}, Price: ${round(self.data[self.curr_step], 2)}, ' +\
               f'Position: {self.position}'
    
    def get_data(self):
        ''' Returns curr_price, balance, portfolio_value '''
        return self.data[self.curr_step], self.get_portfolio_value()