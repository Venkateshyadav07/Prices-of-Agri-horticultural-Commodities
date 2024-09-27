import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
file_path = "dataset.csv"  
commodity_data = pd.read_csv(file_path)

# Set the index to 'Commodities' and transpose the dataframe
commodity_data.set_index('Commodities', inplace=True)
transposed_data = commodity_data.T

# Set the index to a monthly date range starting from 2014
transposed_data.index = pd.date_range(start='2014-01', periods=len(transposed_data), freq='M')

# Forward fill missing values if any
transposed_data = transposed_data.ffill()

# List of available commodities
commodity_options = transposed_data.columns.tolist()

# Streamlit app title
st.title("Agricultural Commodity Price Forecasting")

# Dropdown to select commodity
selected_item = st.selectbox("Select a Commodity", commodity_options)

# When the user submits the selected commodity
if st.button("Predict"):
    # Extract the time series data for the selected commodity
    selected_data = transposed_data[selected_item]

    # Define and fit the SARIMAX model
    sarimax_model = SARIMAX(selected_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    model_fitted = sarimax_model.fit(disp=False)

    # Generate a forecast for the next 5 years (60 months)
    future_forecast = model_fitted.get_forecast(steps=12 * 5)
    predicted_prices = future_forecast.predicted_mean

    # Create a date range for the forecasted period (2025-2029)
    forecast_dates = pd.date_range(start='2025-01', periods=12 * 5, freq='M')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates, 
        f'{selected_item}_Price_Prediction': predicted_prices
    })

    # Display forecasted prices
    st.write(f"### Forecasted Prices for {selected_item} (2025-2029)")
    st.write(forecast_df)

    # Plot actual and forecasted prices
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data, label=f'Actual {selected_item} Prices')
    plt.plot(forecast_dates, predicted_prices, label=f'Predicted {selected_item} Prices', color='orange')
    plt.title(f'{selected_item} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate the training RMSE (Root Mean Squared Error)
    training_rmse = np.sqrt(((selected_data - model_fitted.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {training_rmse:.4f}")
