'''
    Script trains LSTM on training data.
    Generates LSTM_portfolio_value.csv which records the portfolio values of LSTM trader on testing data.
'''
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler

#requires tensorflow2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.keras.optimizers import Adam

from ads_utils import load_data, plot, Environment

INITIAL_BALANCE = 10_000
N_TICKS = 60
EPOCHS = 20
BATCH_SIZE = 72

train_range = [i for i in range(24, 13-1, -1)]
train_data = load_data(train_range)
scaled_dataset = np.reshape(train_data['close'].values, (-1, 1))

scaled_dataset = dataset
train = scaled_dataset[:int(scaled_dataset.shape[0]*0.75)]
valid = scaled_dataset[int(scaled_dataset.shape[0]*0.75)-60:]

############################################################################
# Scale the dataset
sc = MinMaxScaler(feature_range = (0, 1))
train2 = sc.fit_transform(train)
valid2 = sc.transform(valid)
x_train, y_train, x_test, y_test = [], [], [], []

for j in range(N_TICKS, train2.shape[0]):
    x_train.append(train2[j-N_TICKS:j, 0])
    y_train.append(train2[j, 0])
    
for z in range(60, valid2.shape[0]):
    x_test.append(valid2[z-N_TICKS:z, 0])
    y_test.append(valid2[z, 0])

x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

############################################################################
# Define the model
model = Sequential()
model.add(LSTM(units=100, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(LSTM(units=100))
model.add(Dropout(0.4))
model.add(Dense(1))
ADAM = Adam(0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=ADAM)

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), verbose=1,
                    shuffle=False, workers=-1)

############################################################################
# Testing Data
test_range = [i for i in range(6, 1-1, -1)]
test_data = load_data(test_range)
scaled_test_dataset = np.reshape(test_data['close'].values, (-1,1))
test_set = sc.transform(scaled_test_dataset)

x_test, y_test = [], []
for z in range(60, test_set.shape[0]):
    x_test.append(test_set[z-N_TICKS:z, 0])
    y_test.append(test_set[z, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

############################################################################
# Transform the Testing Data 
predicted_stock_price = sc.inverse_transform(model.predict(x_test))
actual_stock_price = sc.inverse_transform(y_test.reshape((-1, 1)))

# Obtain portfolio values
actions = []
portfolio_values = [INITIAL_BALANCE]
balance = INITIAL_BALANCE
prices = []

for j in range(len(actual_stock_price)):
    prices.append(actual_stock_price[j][0])

for j in range(1,len(predicted_stock_price)):
    if predicted_stock_price[j] > actual_stock_price[j-1]:
        #go long
        balance += actual_stock_price[j] - actual_stock_price[j-1]
        actions.append(2)
    else:
        #go short
        balance += actual_stock_price[j-1] - actual_stock_price[j]
        actions.append(0)
    portfolio_values.append(float(balance))

#plot(prices=prices, target_positions=actions, portfolio_values=portfolio_values, right_y_adjust=1.1)
############################################################################

df = pd.DataFrame(portfolio_values)
df.to_csv("LSTM_portfolio_values", header=None)