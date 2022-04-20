import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


start  = '2010-01-01'
end = '2022-01-01'

st.title("Will's Stock Trend Prediction")

user_input = st.text_input('Endter Stock Ticker', 'AAPL')
st.subheader('Examples: TSLA, AAPL, RAND(All these work, some dont)')
df = data_reader.DataReader(user_input,'yahoo',start,end)

#Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA")
ma100 =  df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100 =  df.Close.rolling(100).mean()
ma200 =  df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

#splitting data into training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*.7):])


scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


#load my model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted =  model.predict(x_test)

scaler = scaler.scale_[0]
scale_factor = 1/scaler
y_predicted = y_predicted * scale_factor
y_test = y_test*scale_factor


#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)





#st.subheader('PS: Hi dad')
