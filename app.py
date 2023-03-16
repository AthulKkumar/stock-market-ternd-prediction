import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import load_model
import streamlit as st

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()


start = '2010-01-01'
end = '2022-12-31'
# start = dt.datetime(2013, 1, 1)
# end = dt.datetime(2016, 1, 27)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL') 

df = pdr.get_data_yahoo(user_input, start, end)
# print(df)
# df = web.DataReader('AAPL', 'yahoo', start='2019-09-10', end='2019-10-09')
# df = web.DataReader('AAPL', 'yahoo' ,start ,end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visulization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data_traning = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_traning_array = scaler.fit_transform(data_traning)


# Splitting data into xtrain and ytrain
# x_train = []
# y_train = []

# for i in range(100, data_traning_array.shape[0]):
#     x_train.append(data_traning_array[i-100: i])
#     y_train.append(data_traning_array[i,0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# Load my model

model = load_model('kera_model.h5')

# Testing part

past_100_days = data_traning.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor



# Final Graph

st.subheader('Prediction vs Orginal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)