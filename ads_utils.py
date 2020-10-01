import pandas as pd
import os

def load_data(months, instrument='AMZN'):
    ''' Returns pandas dataframe for one instrument
    Inputs:
        months     - list of month ints to be loaded over the two years e.g. [11, 12, 13, 14]
        instrument - str instrument to be loaded in for those months e.g. 'GOOG'
    Outputs:
        data       - pd.DataFrame
    '''
    data = pd.DataFrame()
    
    for month in months:
        try:
            # Create path regardless of os system
            path_name = os.path.join(os.getcwd(), 'data', instrument, f'{month}.csv')
            monthly_data = pd.read_csv(path_name)
        except:
            raise FileNotFoundError("Refer to generate_data script to obtain data.")
        
        # Assuming we only want to keep time, close and volume - removing open, high and low
        data = pd.concat([data, monthly_data[['time', 'close', 'volume']]], axis=0)
        
    # Sort by time and reset index
    data['time'] = pd.to_datetime(data.time)
    return data.sort_values('time').reset_index().drop('index', axis=1)

############################################################################
import matplotlib.pyplot as plt
import numpy as np

SMALL, MED, LARGE = 22, 28, 34
LW = 1.5
plt.rc('xtick',labelsize=SMALL)
plt.rc('ytick',labelsize=SMALL)

def plot(prices, target_positions=[], portfolio_values=[], title='', filename='', right_y_adjust=1.07, legend_loc='upper left'):
    ''' Output a graph of prices and optionally positions and portfolio values. Will save if filename provided 
    Inputs:
        NECESSARY ARGUMENTS:
        prices           - array of stock prices e.g. [1.02, 1.03, 1.01, 1.03, 1.05]
        
        OPTIONAL ARGUMENTS:
        target_positions - array of target positions in {-1, 0, 1}, equal in length to prices e.g. [0, 0, 1, 1, -1]
        portfolio_values - array of portfolio values, equal in length to prices e.g. [100.00, 99.89, 99.93, 100.02, 100.10]
        title            - string title e.g. 'Test Title'
        filename         - string filename, plot will be saved if a non-empty value is given e.g. 'test graph'
        right_y_adjust   - float for adjusting rightmost y axis if there is clipping
        legend_loc       - string describing legend position according to matplotlib.pyplot legend locs
    '''
    portfolio_values = list(portfolio_values)
    target_positions = list(target_positions)
    prices_colour, portfolio_colour, buy_colour, sell_colour = 'C0', 'C1', '#49E20E', '#FF0000'
    
    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.spines['top'].set_color('none')
    ax2.spines['top'].set_color('none')
    
    # Plot positions
    if target_positions:
        buy_indexes   = np.where(np.diff(target_positions) ==  1)[0]
        buy2_indexes  = np.where(np.diff(target_positions) ==  2)[0]
        sell_indexes  = np.where(np.diff(target_positions) == -1)[0]
        sell2_indexes = np.where(np.diff(target_positions) == -2)[0]
        buys   = np.take(prices, buy_indexes, 0)
        buys2  = np.take(prices, buy2_indexes, 0)
        sells  = np.take(prices, sell_indexes, 0)
        sells2 = np.take(prices, sell2_indexes, 0)
        
        
        ax1.scatter(buy2_indexes,  buys2,  zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='^',
                    label='Buy x2',   c=buy_colour)
        ax1.scatter(buy_indexes,   buys,   zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o',
                    label='Buy',   c=buy_colour)
        ax1.scatter(sell_indexes,  sells,  zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o',
                    label='Sell',  c=sell_colour)
        ax1.scatter(sell2_indexes, sells2, zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='v',
                    label='Sell x2',  c=sell_colour)
        ax1.legend(frameon=False, fontsize=SMALL, loc=legend_loc)
    
    # Plot prices and portfolio values
    ax1.plot(prices, lw=LW, c=prices_colour)
    if portfolio_values:
        ax2.plot(portfolio_values, lw=LW, c=portfolio_colour)
    
    # Label
    ax1.set_title(title, fontsize=LARGE)
    ax1.set_xlabel('Index', fontsize=LARGE)
    ax1.set_ylabel('Stock Price ($)', fontsize=MED, c=prices_colour)
    ax1.tick_params(axis='y', labelcolor=prices_colour)
   
    if portfolio_values:
        ax2.set_ylabel('Portfolio Value ($)', fontsize=MED, zorder=100, c=portfolio_colour, rotation=270)
        ax2.yaxis.set_label_coords(right_y_adjust, 0.5)
        ax2.tick_params(axis='y', labelcolor=portfolio_colour)
    else:
        ax2.get_yaxis().set_visible(False)
        ax1.spines['right'].set_color('none')
        ax2.spines['right'].set_color('none')
        
    # Plot and potentially save
    plt.tight_layout()
    if filename:
        plt.savefig(f'{filename}.png', dpi=fig.dpi)

    plt.show()
    
    
############################################################################
import gym
from gym import spaces
from gym.utils import seeding

INITIAL_BALANCE = 1_000_000
PAST_DAYS = 5

class Environment(gym.Env):  
    # required for stable baselines 
    metadata = {'render.modes': ['human']}
    
    SELL, HOLD, BUY = 0, 1, 2
    PRICES, POSITION, BALANCE = 0, 1, 2
    
    def __init__(self, data, balance=INITIAL_BALANCE, transaction_cost=0.001, i=0, position=1, past_days=PAST_DAYS):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            raise ValueError('Only lists or arrays allowed')
            
        self.past_days = past_days
        self.curr_step = self.past_days
        self.initial_balance = self.balance = balance
        
        self.done = False
        self.data = data
        self.position = position
        self.transaction_cost = transaction_cost
        
        self._seed()
        
        # Sell, Hold, Buy == 0, 1, 2 
        self.action_space = spaces.Discrete(3)

        # Observation space has past_days prices up to and included current price, then balance and position
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.past_days+2, ))
        
    def _next_observation(self):        
        '''Getting the next observation'''
        # Hi Grace - delete this when you read it, I added +1 here such that the frame included the current step - Cameron
        frame = np.array(self.data[self.curr_step - self.past_days + 1: self.curr_step + 1])
        obs = np.append(frame, [self.balance, self.position], axis=0)
        return obs

    def _take_action(self, action):
        curr_price = self.data[self.curr_step]
        
        # Perform position transition (transaction cost is a proportion of price)
        self.balance -= curr_price * self.transaction_cost * abs(action - self.position)
        
        # A Buy
        if (action == self.BUY and self.position == self.HOLD) or (action == self.HOLD and self.position == self.SELL):
            self.balance -= curr_price
        
        # A Sell
        elif (action == self.SELL and self.position == self.HOLD) or (action == self.HOLD and self.position == self.BUY):
            self.balance += curr_price
            
        # Flip Position
        elif abs(action - self.position) == 2:
            self.balance -= 2 * (action-1) * curr_price
        
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
        
        # the change in portfolio value
        reward = self.portfolio_value - prior_portfolio_value     
        
        # Are we done?
        if self.balance <= 0:
            self.done = True
            reward = -1e6
        if self.curr_step >= len(self.data) - 1:
            self.done = True
        
        obs = self._next_observation()
        
        # required to return: observation, reward, done, info
        return obs, reward, self.done, {}
    
    def get_portfolio_value(self):
        ''' Returns current portfolio value '''
        curr_price = self.data[self.curr_step]
        
        if self.position == self.BUY:
            return self.balance + curr_price
        
        elif self.position == self.SELL:
            return self.balance - curr_price
        
        return self.balance
    
    def reset(self):
        '''Reset everything as if we just started (for a new episode)'''
        self.position = self.HOLD
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.done = False
        self.curr_step = self.past_days - 1
        
        # Must return first observation
        return self._next_observation()   

    def save_portfolio(self, mode='human'):
        with open('output.csv', 'a') as file:
            file.write(f'{self.curr_step},{self.portfolio_value},{self.balance}\n')
        return
    
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def __repr__(self):
        return f'Balance: ${round(self.balance, 2)}, Price: ${round(self.data[self.curr_step], 2)}, ' +\
               f'Position: {self.position}'