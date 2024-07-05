import streamlit as st
import pickle
import numpy as np

#Import model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# ScreenSize
screen_size = st.number_input('Screen Size')

# Resolution
screen_res = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1440', '2560x1600'])

# Cpu
processor = st.selectbox('Processor', df['Cpu Brand'].unique())

# Hdd
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# Ssd
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024,2048])

# Gpu
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Os
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    X_res = int(screen_res.split('x')[0])
    Y_res = int(screen_res.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,processor,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
