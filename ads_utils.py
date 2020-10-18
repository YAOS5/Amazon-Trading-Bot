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

SELL, HOLD, BUY = 0, 1, 2

def continuous_actions(target):
    ''' Input: Target actions (continuous space)
    
        Output: Returns the indexes where a BUY, HOLD or SELL occurs'''
    buy_indexes = []
    buy2_indexes = []
    sell_indexes = []
    sell2_indexes = []

    for interval in range(1, len(target)):
        diff = target[interval] - target[interval - 1]
        sign = target[interval] * target[interval - 1]

        if diff == 0:
            continue

        if diff > 0 and sign >= 0:
            buy_indexes.append(interval)
        elif diff > 0 and sign < 0:
            buy2_indexes.append(interval)
        elif diff < 0 and sign >= 0:
            sell_indexes.append(interval)
        elif diff < 0 and sign < 0:
            sell2_indexes.append(interval)
    
    return buy_indexes, buy2_indexes, sell_indexes, sell2_indexes
    
def plot(prices, target_positions=[], portfolio_values=[], title='', filename='', right_y_adjust=1.07, legend_loc='upper left',
        cont=False):
    ''' Output a graph of prices and optionally positions and portfolio values. Will save if filename provided 
    Inputs:
        NECESSARY ARGUMENTS:
        prices           - array of stock prices e.g. [1.02, 1.03, 1.01, 1.03, 1.05]
        
        OPTIONAL ARGUMENTS:
        target_positions - array of target positions in {0, 1, 2}, equal in length to prices e.g. [0, 0, 1, 1, -1]
        portfolio_values - array of portfolio values, equal in length to prices e.g. [100.00, 99.89, 99.93, 100.02, 100.10]
        title            - string title e.g. 'Test Title'
        filename         - string filename, plot will be saved if a non-empty value is given e.g. 'test graph'
        right_y_adjust   - float for adjusting rightmost y axis if there is clipping
        legend_loc       - string describing legend position according to matplotlib.pyplot legend locs
        cont             - flag for continuous action space
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
        
        # if the agent has continuous action space
        if cont:
            buy_indexes, buy2_indexes, sell_indexes, sell2_indexes = continuous_actions(target_positions)
        else:
            buy_indexes   = np.where(np.diff(target_positions) ==  1)[0] + 1
            buy2_indexes  = np.where(np.diff(target_positions) ==  2)[0] + 1
            sell_indexes  = np.where(np.diff(target_positions) == -1)[0] + 1
            sell2_indexes = np.where(np.diff(target_positions) == -2)[0] + 1
        buys   = np.take(prices, buy_indexes, 0)
        buys2  = np.take(prices, buy2_indexes, 0)
        sells  = np.take(prices, sell_indexes, 0)
        sells2 = np.take(prices, sell2_indexes, 0)
        
        first = target_positions[0]
        if first == BUY:
            ax1.scatter([0], prices[0], zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o', alpha=0.7, c=buy_colour)
        elif first == SELL:
            ax1.scatter([0], prices[0], zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o', alpha=0.7, c=sell_colour)
        
        ax1.scatter(buy2_indexes,  buys2,  zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='^', alpha=0.7,
                    label='Buy x2',   c=buy_colour)
        ax1.scatter(buy_indexes,   buys,   zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o', alpha=0.7,
                    label='Buy',   c=buy_colour)
        ax1.scatter(sell_indexes,  sells,  zorder=10, s=150, edgecolors='black', linewidths=0.5, marker='o', alpha=0.7,
                    label='Sell',  c=sell_colour)
        ax1.scatter(sell2_indexes, sells2, zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='v', alpha=0.7,
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

INITIAL_BALANCE = 10_000
PAST_TICKS = 5

class Environment(gym.Env):  
    # required for stable baselines 
    metadata = {'render.modes': ['human']}
    
    SELL, HOLD, BUY = 0, 1, 2
    PRICES, POSITION, BALANCE = 0, 1, 2
    
    def __init__(self, data, balance=INITIAL_BALANCE, transaction_cost=0.001, i=0, position=1, past_ticks=PAST_TICKS):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            raise ValueError('Only lists or arrays allowed')
        
        self.logger = []
        self.epoch_count = 0

        self.past_ticks = past_ticks
        self.curr_step = self.past_ticks+1
        self.initial_balance = self.balance = balance
        
        self.done = False
        self.data = data
        self.position = position
        self.transaction_cost = transaction_cost
        self.cumulative_tc = 0
        self._seed()
        
        # Sell, Hold, Buy == 0, 1, 2 
        self.action_space = spaces.Discrete(3)

        # Observation space has past_ticks prices up to and included current price, then position
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.past_ticks+1, ))
        
    def _next_observation(self):        
        '''Getting the next observation'''
        # Hi Grace - delete this when you read it, I added +1 here such that the frame included the current step - Cameron
        
        # Convert frame into returns
        # These two lines don't work for the DQN
        frame = np.array(self.data[self.curr_step - self.past_ticks: self.curr_step + 1])###################################
        frame = np.diff(frame) / frame[:-1] * 100
        
        # This line does for some reason
        #frame = np.array(self.data[self.curr_step - self.past_ticks + 1: self.curr_step + 1])
        
        obs = np.append(frame, [self.position], axis=0)
        return obs

    def _take_action(self, action):
        curr_price = self.data[self.curr_step]
        
        # Perform position transition (transaction cost is a proportion of price)
        self.balance -= curr_price * self.transaction_cost * abs(action - self.position)
        
        # A Buy
        if (action == self.BUY and self.position == self.HOLD) or (action == self.HOLD and self.position == self.SELL):
            self.balance -= curr_price
            self.cumulative_tc += 1
        
        # A Sell
        elif (action == self.SELL and self.position == self.HOLD) or (action == self.HOLD and self.position == self.BUY):
            self.balance += curr_price
            self.cumulative_tc += 1
            
        # Flip Position
        elif abs(action - self.position) == 2:
            self.balance -= 2 * (action-1) * curr_price
            self.cumulative_tc += 1
        
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
        # Percentange change from initial portfolio value
        #reward = 100 * ((self.portfolio_value/self.initial_balance) - 1)  
        self.logger.append([self.epoch_count, reward, self.portfolio_value, self.cumulative_tc, self.curr_step])

        # Are we done?
        if self.balance <= 0:
            self.done = True
            reward = -1e6
        if self.curr_step >= len(self.data) - 2:
            self.done = True
        
        obs = self._next_observation()
        
        # required to return: observation, reward, done, info
        return obs, reward, self.done, {"logs": self.logger}
    
    def get_portfolio_value(self):
        ''' Returns current portfolio value '''
        curr_price = self.data[self.curr_step]
        
        if self.position == self.BUY:
            return self.balance + curr_price
        
        elif self.position == self.SELL:
            return self.balance - curr_price
        
        return self.balance
    
    def reset(self, rand_start=False):
        '''Reset everything as if we just started (for a new episode)'''
        self.position = self.HOLD
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.done = False
        self.curr_step = np.random.randint(self.past_ticks, 2*len(self.data)//3) if rand_start else self.past_ticks+1
        self.epoch_count += 1
        # Must return first observation
        return self._next_observation()   

    def save_portfolio(self, mode='human'):
        with open('output.csv', 'a') as file:
            file.write(f'{self.curr_step},{self.portfolio_value},{self.balance}\n')
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def __repr__(self):
        return f'Balance: ${round(self.balance, 2)}, Price: ${round(self.data[self.curr_step], 2)}, ' +\
               f'Position: {self.position}'
    
    def get_data(self):
        ''' Returns curr_price, balance, portfolio_value '''
        return self.data[self.curr_step], self.get_portfolio_value()

############################################################################
def write_to_logs(logs, filename="logs"):
    # [reward, portfolio, curr_step]
    path = f"logs/{filename}.csv"
    logs = pd.DataFrame(logs) 
    logs.to_csv(filename, header=["epoch", "reward", "portfolio", "cumulative_tc", "curr_step"], index=False)

def moving_average(values, window=10):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')    
    
def plot_k_timesteps(logs="logs.csv", k=100, y_col="reward"):
    ''' logs - the file where logs are stored
        k    - log at each k timesteps 
        y    - reward or portfolio'''
    df = pd.read_csv(logs)
    df.new = df.iloc[::k, :]
    x = np.arange(0, len(df.new))
    y = df.new[y_col]
    y = moving_average(y, window=10)
    
    # Truncate x
    x = x[len(x) - len(y):]

    #plots reward at each k timestep
    fig, ax = plt.subplots(figsize=(14,8))
    plt.plot(x, y)
    
    ax.set_title(f"{y} at each timestep", fontsize=22)
    ax.set_xlabel('timestep', fontsize=20)
    ax.set_ylabel(y_col, fontsize=20)