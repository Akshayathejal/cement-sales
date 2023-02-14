import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.header('Cement Sales Forecast Prediction')
uploaded_file = st.file_uploader(" ", type=['xlsx'])

if uploaded_file is not None:     
    df = pd.read_excel(uploaded_file)
    
    
    hwe_model_mul_add = ExponentialSmoothing(df["sales"][:71], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
    
   pred = hwe_model_mul_add.predict(start = df.index[0], end = df.index[-1])
    
    
    st.subheader("Cement sales Prediction App")
   
    st.write("sales Forecast: ", pred)
   
    
    st.subheader("Thanks for visit.")
