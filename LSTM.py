'''
    Script trains LSTM on training data.
    Generates LSTM_portfolio_value.csv which records the portfolio values of LSTM trader on testing data.
'''
# with transaction cost included
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

INITIAL_BALANCE = 10_000
SELL, HOLD, BUY = 0, 1, 2

def create_model():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(x_train.shape[1],1), return_sequences=True))
    model.add(LSTM(units=100)) 
    model.add(Dropout(0.4))
    model.add(Dense(1))
    ADAM = Adam(0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=ADAM)
    
    return model


transaction_cost = 0.001
def take_action(action, curr_position, curr_price, balance):
    
    # Perform position transition (transaction cost is a proportion of price)
    balance -= curr_price * transaction_cost * abs(action - curr_position)

    # A Buy
    if (action == BUY and curr_position == HOLD) or (action == HOLD and curr_position == SELL):
        balance -= curr_price

    # A Sell
    elif (action == SELL and curr_position == HOLD) or (action == HOLD and curr_position == BUY):
        balance += curr_price

    # Flip Position
    elif abs(action - curr_position) == 2:
        balance -= 2 * (action-1) * curr_price
    
    return balance

actions = []
portfolio_values = [INITIAL_BALANCE]
balance = INITIAL_BALANCE
prices = []
prev_action = HOLD

for i in range(6):
    
    data = load_data([i for i in range(6-i,6-i+4)])
    
    dataset = np.reshape(data['close'].values,(-1,1))
    
    scaled_dataset = dataset
    train= scaled_dataset[:int(scaled_dataset.shape[0]*0.75)]
    valid = scaled_dataset[int(scaled_dataset.shape[0]*0.75)-60:]
    
    sc = MinMaxScaler(feature_range = (0, 1))
    train2 = sc.fit_transform(train)
    valid2 = sc.transform(valid)
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for j in range(60,train2.shape[0]):
        x_train.append(train2[j-60:j,0])
        y_train.append(train2[j,0])
        
    for z in range(60,valid2.shape[0]):
        x_test.append(valid2[z-60:z,0])
        y_test.append(valid2[z,0])
    
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

    model = tf.keras.models.load_model(f'./extras/LSTM_checkpoints/{i}')
    
    predicted_stock_price = sc.inverse_transform(model.predict(x_test))
    actual_stock_price = sc.inverse_transform(y_test.reshape((-1, 1)))
    
    for j in range(len(actual_stock_price)):
        prices.append(actual_stock_price[j][0])
    
    
    for j in range(1, len(predicted_stock_price)):
        buy_cost = actual_stock_price[j-1] * transaction_cost * abs(BUY - prev_action)
        sell_cost = actual_stock_price[j-1] * transaction_cost * abs(SELL - prev_action)
        
        # target action : BUY
        if predicted_stock_price[j] - actual_stock_price[j-1] > buy_cost:
            action = BUY
            actions.append(BUY)
             
        # target action : SELL
        elif actual_stock_price[j-1] - predicted_stock_price[j] > sell_cost:
            action = SELL
            actions.append(SELL)
            
        # target action: HOLD
        else:
            action = HOLD
            actions.append(HOLD)
        
        # take action
        balance = take_action(action, prev_action, actual_stock_price[j-1], balance)
        
        # reward
        if action == BUY:
            portfolio_value = balance + actual_stock_price[j]
        elif action == SELL:
            portfolio_value = balance - actual_stock_price[j]
        else:
            portfolio_value = balance
            
        portfolio_values.append(float(portfolio_value))
        prev_action = action
        
# plot(prices=prices,target_positions = actions,portfolio_values = portfolio_values)
    
    
df = pd.DataFrame(portfolio_values)
df.to_csv("LSTM_portfolio_values", header=None)