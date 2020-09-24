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
        # Sorry, only works for Windows atm, please update so it works for both and remove this comment
        monthly_data = pd.read_csv(os.getcwd() + f'\data\\{instrument}\\{month}.csv')
        # Assuming we only want to keep time, close and volume - removing open, high and low
        data = pd.concat([data, monthly_data[['time', 'close', 'volume']]], axis=0)
        
    # Sort by time and reset index
    data['time'] = pd.to_datetime(data.time)
    return data.sort_values('time').reset_index().drop('index', axis=1)

############################################################################
import matplotlib.pyplot as plt

SMALL, MED, LARGE = 22, 28, 34
LW = 1.5
plt.rc('xtick',labelsize=SMALL)
plt.rc('ytick',labelsize=SMALL)

def plot(prices, positions=[], portfolio_values=[], title='', filename=''):
    ''' Output a graph of prices and optionally positions and portfolio values. Will save if filename provided 
    Inputs:
        prices           - array of stock prices
        positions        - array of positions in {-1, 0, 1}, equal in length to prices
        portfolio_values - array of portfolio values, equal in length to prices
        title            - string title
        filename         - string filename, plot will be saved if a non-empty value is given
    '''
    portfolio_values = list(portfolio_values)
    positions = list(positions)
    prices_colour, portfolio_colour, buy_colour, sell_colour = 'C0', 'C1', '#49E20E', '#FF0000'
    
    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.spines['top'].set_color('none')
    ax2.spines['top'].set_color('none')
    
    # Plot positions
    if positions:
        buy_indexes  = np.where(np.diff(positions) ==  1)[0]
        sell_indexes = np.where(np.diff(positions) == -1)[0]
        buys  = np.take(prices, buy_indexes, 0)
        sells = np.take(prices, sell_indexes, 0)
        
        ax1.scatter(buy_indexes,  buys,  zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='^',
                    label='Buy',  c=buy_colour)
        ax1.scatter(sell_indexes, sells, zorder=10, s=200, edgecolors='black', linewidths=0.5, marker='v',
                    label='Sell', c=sell_colour)
        ax1.legend(frameon=False, fontsize=SMALL, loc='upper left')
    
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
        ax2.yaxis.set_label_coords(1.07, 0.5)
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
