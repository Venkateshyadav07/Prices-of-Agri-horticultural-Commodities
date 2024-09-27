import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
file_path = "Dataset2.csv"  
commodity_prices = pd.read_csv(file_path)

# Set 'Commodities' as the index and transpose the dataframe to align the data by years
commodity_prices.set_index('Commodities', inplace=True)
transposed_prices = commodity_prices.T

# Set the index to reflect the year-end dates starting from 2014
transposed_prices.index = pd.date_range(start='2014', periods=len(transposed_prices), freq='YE')

# Forward-fill any missing data
transposed_prices = transposed_prices.ffill()

# List of commodities available for selection
commodity_list = transposed_prices.columns.tolist()

# Streamlit application title
st.title("Agricultural Commodity Price Forecasting")

# Dropdown menu to select a commodity
selected_commodity = st.selectbox("Choose a Commodity", commodity_list)

# Action when "Submit" button is clicked
if st.button("Submit"):
    # Get the data for the selected commodity
    price_data = transposed_prices[selected_commodity]

    # Build the SARIMAX model
    model = SARIMAX(price_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    fitted_model = model.fit(disp=False)

    # Forecast for the next 5 years
    price_forecast = fitted_model.get_forecast(steps=5)
    forecast_values = price_forecast.predicted_mean

    # Create a date range for the forecasted period (2025-2029)
    forecast_dates = pd.date_range(start='2025', periods=5, freq='YE')
    forecast_df = pd.DataFrame({'Year': forecast_dates, f'{selected_commodity}_Price_Forecast': forecast_values})

    # Display the forecasted prices
    st.write(f"### {selected_commodity} Price Forecast (2025-2029)")
    st.write(forecast_df)

    # Plot the actual and forecasted prices
    plt.figure(figsize=(10, 6))
    plt.plot(price_data, label=f'Actual {selected_commodity} Prices')
    plt.plot(forecast_dates, forecast_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
    plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate and display the training Root Mean Squared Error (RMSE)
    training_rmse = np.sqrt(((price_data - fitted_model.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {training_rmse:.4f}")
