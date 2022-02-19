import pandas as pd
import numpy as np
import math
import pandas_datareader as web
from matplotlib import pyplot as plt
import tensorflow 
from tensorflow.keras.models import load_model
import streamlit as st

import yfinance as yf

st.title('Stock Prediction Portal')

#####################################################################

# SUMMARY SECTION

st.header('Know the Stock')

stocks1 = ("AAPL","TSLA","MSFT","GOOG","FB","AMZN","WMT","NFLX")
user_input1 = st.selectbox('Select the Stock : ', stocks1)

stdate1 = st.date_input("Starting Date : ")
endate1 = st.date_input("Ending Date : ")


company = yf.Ticker(user_input1)

comp = yf.download(user_input1, start = stdate1, end= endate1)

validate = ("1d","5d","1mo","3mo","6mo")
user_input2 = st.selectbox('Select a Valid duration : ', validate)

datacom = company.history(period=user_input2)

st.subheader(user_input1)
st.write(company.info['longBusinessSummary'])
st.write(comp)
st.line_chart(datacom.values)


#####################################################################

# COMPARE SECTION

st.header('Compare the Stocks')

tickers = ("AAPL","TSLA","MSFT","GOOG","FB","AMZN","WMT","NFLX")

dropdown = st.multiselect('Select the Stock(s)', tickers)

stdate = st.date_input('Enter Starting Date : ')
endate = st.date_input('Enter Ending Date: ')

def relativeret(df1):
    rel = df1.pct_change()
    cumret = (1+rel).cumprod() - 1
    cumret = cumret.fillna(0)
    return cumret

if len(dropdown) > 0:
    df1 = relativeret(yf.download(dropdown, stdate, endate)['Adj Close'])
    st.line_chart(df1)

#####################################################################

# PREDICTION SECTION

st.header('Get Information about the Stock')


stocks = ("AAPL","TSLA","MSFT","GOOG","FB","AMZN","WMT","NFLX")
user_input = st.selectbox('Input Name : ', stocks)
enddate = st.date_input("Enter Today's Date : ")

df = web.DataReader(user_input, data_source = 'yahoo', start = '2015-01-01', end = enddate)

st.subheader('Stock data')
st.text('Full Data: ')
st.write(df.reset_index())
st.text('First Five : ')
st.write(df.reset_index().head())
st.text('Last Five : ')
st.write(df.reset_index().tail())


st.subheader('Visualizing the Graph')
fig = plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price $USD', fontsize=18)
st.pyplot(fig)



# Train and test

# DF with only close column

data = df.filter(['Close'])

# Into numpy
dataset = data.values

# Rows to train the model 80%
training_data_len = math.ceil(len(dataset)* .8)


# Scaling 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Load ,ode

model = load_model('keras_model.h5')



# Test

# Create the testing dataset
# Create a new array containing scaled values from index 1543 to 2003

test_data = scaled_data[training_data_len-60:, :]

# Create the dataset of x_test and y_test

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

# Convert data into numpy array

x_test = np.array(x_test)

# Reshape the data

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Get the model price prediction values

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Final 

st.subheader('Prediction Graph')

# Plotting the data

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data

fig2 = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Closed price USD $', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Val' , 'Predictions'], loc = 'lower right')
st.pyplot(fig2)


st.subheader('Stock Prediction area')
enddate2 = st.date_input('Enter the last date: ')
# Getting the quote and predicting on 2019-12-18(next day)

C_quote = web.DataReader(user_input, data_source='yahoo', start='2015-01-01', end= enddate2)

# Create a new dataframe

new_df = C_quote.filter(['Close'])

# Taking last 60 days data and converting into array

last_60_days = new_df[-60:].values

# Scale the data b/w 0 and 1

last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list

X_test = []

# Append last 60 days data

X_test.append(last_60_days_scaled)
# Converting into array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test)

#Undo the scaling

pred_price = scaler.inverse_transform(pred_price)
st.write(pred_price)


st.subheader('Matching the stock')
# Getting the quote and predicting on 2019-12-18(next day) - ACTUAL
startdate3 = st.date_input('Enter the date to match')
enddate3 = st.date_input('Enter the date again')

C_quote2 = web.DataReader(user_input, data_source='yahoo', start= startdate3, end=enddate3)
st.write(C_quote2['Close'])
