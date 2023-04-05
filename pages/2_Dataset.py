# PageOne contents

import streamlit as st

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

start = '2010-01-01'
end = '2022-12-31'
# start = dt.datetime(2013, 1, 1)
# end = dt.datetime(2016, 1, 27)


user_input = st.text_input('Enter Stock Ticker', 'AAPL') 

df = pdr.get_data_yahoo(user_input, start, end)

import streamlit as stm

# stm.title("This is PageOne Geeks.")


st.subheader('Data from 2010 - 2019')
# st.write(df.describe())

st.write(df)