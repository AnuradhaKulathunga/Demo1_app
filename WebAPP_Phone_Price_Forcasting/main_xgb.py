import streamlit as st
from datetime import date
import plotly.graph_objects as go
import pandas as pd

phone_options_df = pd.read_excel("/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Phone_Options_Collecting.xlsx")

st.title('Used Apple phones Price Forecast App')

phones = phone_options_df.Phone
selected_phone = st.selectbox('Select an Apple phone for Price prediction', phones,index=0)
import json
# Load Phone_Dict.json into phone_dict
with open("/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Dictionaries_TextBlob/Sort_Encodes_Phones_dict.json", 'r') as f:
    phone_dict = json.load(f)

selected_phone=phone_dict[selected_phone]

st.write(f"Selected phone codde: {selected_phone}")

Full_df = pd.read_excel("/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Full_Data_FbProphet_Sentiments.xlsx")
End_date = Full_df["Date"].max()

st.write("### Select The DATE To Forecast")
# st.write(f"Selected Date and Time: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")

# Extracting year, month, day, hour, and minute from the End_date
initial_year = End_date.year
initial_month = End_date.month
initial_day = End_date.day
initial_hour = End_date.hour
initial_minute = End_date.minute

# Setting initial values for the sliders
year = st.slider("Select a year", min_value=2024, max_value=2030, value=initial_year)
month = st.slider("Select a month", min_value=1, max_value=12, value=initial_month)
day = st.slider("Select a day", min_value=1, max_value=30, value=initial_day)
hour = st.slider("Select an hour", min_value=0, max_value=23, value=initial_hour)
minute = st.slider("Select a minute", min_value=0, max_value=59, value=initial_minute)

# Checking if the selected date is earlier than the initial date
selected_date = pd.Timestamp(year, month, day, hour, minute)
initial_date = pd.Timestamp(initial_year, initial_month, initial_day, initial_hour, initial_minute)

if selected_date < initial_date:
    st.warning("Selected date cannot be earlier than the initial date. Resetting to the initial date.")
    year = initial_year
    month = initial_month
    day = initial_day
    hour = initial_hour
    minute = initial_minute



# Function to calculate  future date
from datetime import datetime
def calculate_dates(year, month, day, hour, minute):
    future_date = datetime(year, month, day, hour, minute)
    return future_date



future_date = calculate_dates(year, month, day, hour, minute)




def auto_generate_points(start, end):
    # Calculate the total time difference in seconds
    delta = (end - start).total_seconds()

    # Define thresholds for different frequencies

    if delta <= 3600:  # less than or equal to 1 hour
        freq = 'T'  # minute frequency
    elif delta <= 86400:  # less than or equal to 1 day
        freq = 'H'  # Hourly frequency
    elif delta <= 2678400:  # less than or equal to 1 month
        freq = 'D'  # Daily frequency
    elif delta <= 31536000:  # less than or equal to 1 year
        freq = 'D'  # month-end frequency
    else:  # more than 1 year
        freq = 'MS'  # month start frequency

    return pd.date_range(start=start, end=end, freq=freq)


date_range = auto_generate_points(End_date, future_date)
# date_range = create_date_range(Full_df["Date"].max(), future_date)
df = pd.DataFrame(date_range,columns=['Date'])
# Extract features from the "Date" column
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day
df['Minute'] = df['Date'].dt.minute
df['Hour'] = df['Date'].dt.hour

Cleaned_phone_options_df = pd.read_excel("/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Cleaned_phone_options_df.xlsx")

# Function to duplicate rows
def duplicate_rows(df, n):
    return pd.concat([df]*n, ignore_index=True)


# Filter rows from Cleaned_phone_options_df
filtered_rows = Cleaned_phone_options_df[Cleaned_phone_options_df.Phone_Encode == selected_phone ]

# Duplicate rows
duplicated_df = duplicate_rows(filtered_rows, len(df))

# # Concatenate df and duplicated_df
# df = pd.concat([df, duplicated_df.reset_index(drop=True)], axis=1)
# Concatenate df and filtered_rows
df = pd.concat([df, duplicated_df], axis=1)

# Calculate the date difference
df["Date_Difference"] = (df["Date"] - df["Released_Date"]).dt.days
# Calculate the minute difference
df["Minute_Difference"] = (df["Date"] - df["Released_Date"]).dt.total_seconds() / 60



import os
import pandas as pd
import joblib


def predict_sentiment(df, phone):
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Prepare the DataFrame for prediction
    future_dates = df[['Date']].rename(columns={'Date': 'ds'})

    # Load the pre-trained models
    models_path = f'/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/FbProphet_models_Sentiment_forcasting/{phone}'
    positive_model_filename = os.path.join(models_path, f'{phone}_Positive.joblib')
    negative_model_filename = os.path.join(models_path, f'{phone}_Negative.joblib')

    if not os.path.exists(positive_model_filename) or not os.path.exists(negative_model_filename):
        raise FileNotFoundError(f"Model files not found for phone {phone}")

    positive_model = joblib.load(positive_model_filename)
    negative_model = joblib.load(negative_model_filename)

    # Make predictions
    positive_forecast = positive_model.predict(future_dates)
    negative_forecast = negative_model.predict(future_dates)

    # Add predictions to the original DataFrame
    df['Positive_Predicted'] = positive_forecast['yhat']
    df['Negative_Predicted'] = negative_forecast['yhat']

    return df


# Predict sentiment for the example phone
df = predict_sentiment(df, selected_phone)
print(df)

import numpy as np
import pickle

# Load the scaler object from the file
with open('/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Price_Predict_XGBBOOST/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

# Load the best parameters model
with open('/mount/src/demo1_app/WebAPP_Phone_Price_Forcasting/Price_Predict_XGBBOOST/best_xgb_params.pkl', 'rb') as f:
    loaded_model = joblib.load(f)
import xgboost as xgb
def predict_phone_prices(df, loaded_model=loaded_model,scaler_X=scaler_X):
    # Select numerical columns from the DataFrame
    X = df.select_dtypes(include=[np.number]).reindex(columns=['Display_Size', 'Camera', 'Ram', 'Battery_Capacity', 'Date_Difference',
       'Minute_Difference', 'Phone_Encode', 'Positive_Predicted',
       'Negative_Predicted', 'Year', 'Month', 'Day', 'Minute', 'Hour'])



    # # Scale the features

    X_scaled=scaler_X.transform(X)

    # Predict prices
    predictions = loaded_model.predict(X_scaled)

    # Save predicted prices in DataFrame
    df["Price"] = predictions

    return df

df = predict_phone_prices(df)

st.write(df)

st.write(f'Forecast Price plot till {selected_date}')

original = Full_df[["Date", "Price"]].sort_values(by="Date")[Full_df.Phone_Encode==selected_phone]
forecast = df[["Date", "Price"]]

# Extract the last point of the original data
last_original_point = original.iloc[-1]
# Extract the first point of the forecast data
first_forecast_point = forecast.iloc[0]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=original["Date"], y=original["Price"], mode='lines', name='Original'))
fig1.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Price"], mode='lines', name='Forecast', line=dict(color='red')))

# Add connecting line trace
fig1.add_trace(go.Scatter(
    x=[last_original_point["Date"], first_forecast_point["Date"]],
    y=[last_original_point["Price"], first_forecast_point["Price"]],
    mode='lines',
    name='Connecting Line',
    line=dict(color='red')
))

st.plotly_chart(fig1)

st.subheader(f'Predicted Price: RS. {forecast.loc[forecast["Date"].idxmax(), "Price"]}')

