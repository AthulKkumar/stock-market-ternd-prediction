import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import load_model
import streamlit as st

import requests
from streamlit_lottie import st_lottie


# from st_aggrid import AgGrid, GridOptionsBuilder
# from st_aggrid.shared import GridUpdateMode
from streamlit_option_menu import option_menu

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

# st. set_page_config(layout="wide")

# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_anime_json = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_yr4PZhPT5T.json")

# with st.sidebar:
#     selected = option_menu("Menu", ['Home', 'Dataset', 'Prediction'],
#         icons=['house', '', 'gear'], default_index=0)
# if selected == "Home":
#     st.title("Predicting House Price in Bengaluru")
#     st_lottie(lottie_anime_json, key="hello", width='1100px', height='540px')
#     abstract = '<p style="text-align: justify; font-size: 20px;">In recent years, Karnataka, one of the hotspots for real estate\
#     development, has seen an increase in demand from potential home\
#     buyers and investors, and is expected to witness a further boom\
#     in the sector by 2024. What are the things that a potential home\
#     buyer considers before purchasing a house? The location, the size\
#     of the property, vicinity to offices, schools, parks, restaurants,\
#     hospitals or the stereotypical white picket fence? What about the\
#     most important factor — the price? Now with the lingering impact of\
#     COVID-19, the enforcement of the Real Estate Act, and the lack of\
#     trust in property developers in the city, housing units sold across\
#     India in 2021 dropped by 3 per cent. In fact, the property prices in\
#     Bengaluru fell by almost 5 per cent in the second half of 2021. Buying\
#     a home, especially in a city like Bengaluru, is a tricky choice.\
#     While the major factors are usually the same for all metros, there are\
#     others to be considered for the Silicon Valley of India. With its help\
#     millennial crowd, vibrant culture, great climate and a slew of job\
#     opportunities, it is difficult to ascertain the price of a house in\
#     Bengaluru. People are looking to buy a new home with their budgets and\
#     by analysing market strategies. But main disadvantage of current system\
#     is to calculate a price of house without necessary prediction about future\
#     market trends and result is price increase. So, the main aim of this project\
#     is to predict accurate price of house without any loss. There are many\
#     factors that have to be taken into consideration for predicting house price\
#     and try to predict efficient house pricing for customers with respect to their\
#     budget as well as also according to their priorities. So, this project aims to\
#     create a housing cost prediction model. By using Machine learning algorithms like\
#     Linear Regression, Decision Tree Regression, K-Means Regression and Random Forest\
#     Regression. This model will help people to put their requirements into a web\
#     application without moving towards a broker.</p>'
#     st.markdown(abstract, unsafe_allow_html=True)
# elif selected == "Dataset":
#     # st.title("Dataset")
#     # st_lottie(lottie_anime_json2, key="data",width='1100px', height='700px')
#     # st.subheader("The Cleaned Dataset Used For Preparing the Model")
#     # data = pd.read_csv('Cleaned_data.csv')
#     # gd = GridOptionsBuilder.from_dataframe(data)
#     # # gd.configure_pagination(enabled=True)
#     # # gd.configure_default_column(editable=True, groupable=True)
#     # AgGrid(data)
#     # st.title("  ")
#     # st.title("  ")
#     # st.subheader("The Uncleaned Dataset")
#     # udata = pd.read_csv('Bengaluru_House_Data.csv')
#     # gd = GridOptionsBuilder.from_dataframe(data)
#     # AgGrid(udata)
#     start = '2010-01-01'
#     end = '2022-12-31'
#     # start = dt.datetime(2013, 1, 1)
#     # end = dt.datetime(2016, 1, 27)


#     user_input = st.text_input('Enter Stock Ticker', 'AAPL') 

#     df = pdr.get_data_yahoo(user_input, start, end)

#     import streamlit as stm

#     # stm.title("This is PageOne Geeks.")


#     st.subheader('Data from 2010 - 2019')
#     # st.write(df.describe())

#     st.write(df)

# if selected == "Prediction":

    # data = pd.read_csv('Cleaned_data.csv')
    # pipe = pickle.load(open("RidgeModelnew.pk1", "rb"))
    # locations = sorted(data['location'].unique())
    # st.title("House Price Prediction")
    # location = st.selectbox('Select the loctaion', locations)
    # bhk = st.number_input('No of bedrooms', min_value=1, max_value=4, step=1)
    # bath = st.number_input('No of bathrooms', min_value=1, max_value=4, step=1)
    # sqft = st.number_input('Square feet area', min_value=400, max_value=4000, step=100)
    # if st.button('Predict'):
    #     user_input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    #     prediction = pipe.predict(user_input)[0] * 1e5
    #     prediction = int(prediction)
    #     statename = "ksjfldj"
    #     st.markdown("""
    #         #### <span>Predicted Price:</span> <span style="color:red">Rs.{temp1}</span>   
    #             """.format(temp1=prediction)    , unsafe_allow_html=True)

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