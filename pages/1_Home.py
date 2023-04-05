# # Code in pagetwo.py
# import json
# import requests

# import streamlit as st
# from streamlit_lottie import st_lottie

# import streamlit as stm

# stm.title("STOCK MARKET TREND PREDICTION")

# # def load_lottiefile(filepath: str):
# #     with open(filepath, "r") as f:
# #         return json.load(f)
    
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

    
# # lottie_file = load_lottiefile("lotie/stockmarket.json")  
# lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_kuhijlvx.json")  

# # st_lottie(lottie_file, speed=1, height=300)

# st_lottie(
#     lottie_hello,
#     height=300
# )

# st.text( ' Stock market is one of the major fields that investors are dedicated to, thus stock market price trend prediction is always a hot topic for researchers from both financial and technical domains. In this research, our objective is to build a state-of-art prediction model for price trend prediction, which focuses on short-term price trend prediction. 
# ')

# stm.sidebar.success("You are currently viewing Page Two Geek")
# Code in pagetwo.py
import json
import requests

import streamlit as st
from streamlit_lottie import st_lottie

import streamlit as stm

stm.title("STOCK MARKET TREND PREDICTION")

# def load_lottiefile(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)
    
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

    
# lottie_file = load_lottiefile("lotie/stockmarket.json")  
lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_kuhijlvx.json")  

# st_lottie(lottie_file, speed=1, height=300)

st_lottie(
    lottie_hello,
    height=300
)

st.text(
    '''Stock market is one of the major fields that investors are dedicated to, thus stock 
market price trend prediction is always a hot topic for researchers from both financial 
and technical domains. In this research, our objective is to build a state-of-art prediction
model for price trend prediction, which focuses on short-term price trend prediction.''')
# Debugging code

