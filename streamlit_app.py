import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Preprocess data
def preprocess_data(df, company):
    company_data = df[df['Company'] == company].copy()
    company_data.sort_values('Date', inplace=True)
    company_data.reset_index(drop=True, inplace=True)  # Reset index to ensure alignment
    
    # Feature engineering
    company_data['Year'] = company_data['Date'].dt.year
    company_data['Quarter'] = company_data['Date'].dt.quarter
    company_data['Days'] = (company_data['Date'] - company_data['Date'].min()).dt.days
    
    # Create lag features
    for lag in [1, 2, 4]:
        company_data[f'Close_lag_{lag}'] = company_data['Close'].shift(lag)
        company_data[f'Volume_lag_{lag}'] = company_data['Volume'].shift(lag)
    
    # Create rolling features
    company_data['Close_rolling_mean_4'] = company_data['Close'].rolling(4).mean()
    company_data['Volume_rolling_mean_4'] = company_data['Volume'].rolling(4).mean()
    
    company_data.dropna(inplace=True)
    company_data.reset_index(drop=True, inplace=True)  # Reset index again after dropping NA
    
    return company_data

# Train model
def train_model(company_data, test_size=0.2):
    features = ['total_profit', 'total_spending', 'total_revenue', 
                'High', 'Low', 'Open', 'Volume', 'Year', 'Quarter', 'Days',
                'Close_lag_1', 'Close_lag_2', 'Close_lag_4',
                'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_4',
                'Close_rolling_mean_4', 'Volume_rolling_mean_4']
    
    X = company_data[features]
    y = company_data['Close']
    
    # Ensure we don't split in the middle of time series
    split_idx = int(len(company_data) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate R-squared
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return model, X_train, X_test, y_train, y_test, train_pred, test_pred, train_r2, test_r2

# Make future predictions
def predict_future(model, company_data, start_date, periods=4):
    last_data = company_data.iloc[-1].copy()
    predictions = []
    dates = []
    
    current_date = pd.to_datetime(start_date)
    
    for _ in range(periods):
        # Prepare features for prediction
        features = {
            'total_profit': last_data['total_profit'],
            'total_spending': last_data['total_spending'],
            'total_revenue': last_data['total_revenue'],
            'High': last_data['High'],
            'Low': last_data['Low'],
            'Open': last_data['Open'],
            'Volume': last_data['Volume'],
            'Year': current_date.year,
            'Quarter': (current_date.month - 1) // 3 + 1,
            'Days': (current_date - company_data['Date'].min()).days,
            'Close_lag_1': last_data['Close'],
            'Close_lag_2': company_data.iloc[-2]['Close'] if len(predictions) == 0 else predictions[-1],
            'Close_lag_4': company_data.iloc[-4]['Close'] if len(predictions) < 3 else predictions[-3],
            'Volume_lag_1': last_data['Volume'],
            'Volume_lag_2': company_data.iloc[-2]['Volume'],
            'Volume_lag_4': company_data.iloc[-4]['Volume'],
            'Close_rolling_mean_4': np.mean([company_data.iloc[-4]['Close'], 
                                           company_data.iloc[-3]['Close'],
                                           company_data.iloc[-2]['Close'],
                                           last_data['Close']]),
            'Volume_rolling_mean_4': np.mean([company_data.iloc[-4]['Volume'], 
                                            company_data.iloc[-3]['Volume'],
                                            company_data.iloc[-2]['Volume'],
                                            last_data['Volume']])
        }
        
        # Create DataFrame for prediction
        X_pred = pd.DataFrame([features])
        
        # Make prediction
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        dates.append(current_date)
        
        # Update for next iteration
        last_data['Close'] = pred
        current_date += relativedelta(months=3)
    
    return dates, predictions

# Streamlit app
st.title('WallStreetBot')
st.write("""
This app predicts stock prices using historical financial data.
Select a company and a date to see predictions.
""")

# Sidebar - Company selection
st.sidebar.header('User Input')
selected_company = st.sidebar.selectbox('Select Company', sorted(data['Company'].unique()))

# Get company data
company_data = preprocess_data(data, selected_company)

# Sidebar - Date selection
min_date = company_data['Date'].min().to_pydatetime()
max_date = company_data['Date'].max().to_pydatetime()
default_date = max_date - relativedelta(years=1)

selected_date = st.sidebar.date_input(
    'Select prediction start date',
    value=default_date,
    min_value=min_date,
    max_value=max_date - relativedelta(months=3)
)

localeRows = {
    "Reel GDP per capitas": "GDP_per_capitas",
    "Federals Funds": "Federal_Funds",
    "Unemployment Rates": "Unemployment_Rates",
    "Inflation Rates": "Inflation_rate",
    # "Total Profit": "total_profit",
    # "Total Spending": "total_spending",
    # "Total Revenue": "total_revenue",
    ## you can uncoment the following line but the data is weird
}
extern_data = st.sidebar.multiselect('External Data to show', localeRows.keys())


# Convert to datetime
selected_date = pd.to_datetime(selected_date)

# Train model
model, X_train, X_test, y_train, y_test, train_pred, test_pred, train_r2, test_r2 = train_model(company_data)
# Make predictions
if st.sidebar.button('Predict'):
    # Get the index of the selected date
    try:
        idx = company_data[company_data['Date'] == selected_date].index[0]
    except IndexError:
        st.error("Selected date not found in the dataset. Please choose a different date.")
        st.stop()
    
    actual_data = company_data.iloc[:idx+1].copy()
    
    # Predict future values
    prediction_dates, predictions = predict_future(model, actual_data, selected_date, periods=4)
    
    # Get actual values for comparison
    actual_future = company_data[company_data['Date'].isin(prediction_dates)]
    
    # Create DataFrame for results
    results = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Close': predictions
    })
    
    if not actual_future.empty:
        results = results.merge(actual_future[['Date', 'Close']], on='Date', how='left')
        results.rename(columns={'Close': 'Actual Close'}, inplace=True)
        
        # Calculate R-squared for predictions
        if len(results) > 1:
            pred_r2 = r2_score(results['Actual Close'], results['Predicted Close'])
            st.subheader('Prediction Results')
            st.write(f"R-squared for predictions: {pred_r2:.4f}")
        
        # Display results table
        st.dataframe(results.style.format({
            'Date': lambda x: x.strftime('%Y-%m-%d'),
            'Predicted Close': '{:.2f}',
            'Actual Close': '{:.2f}'
        }))
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(company_data['Date'], company_data['Close'], label='Historical Close', color='blue')
        ax.plot(results['Date'], results['Predicted Close'], 'ro-', label='Predicted Close')
        
        if 'Actual Close' in results.columns:
            ax.plot(results['Date'], results['Actual Close'], 'go-', label='Actual Close')
        
        ax.axvline(x=selected_date, color='gray', linestyle='--', label='Prediction Start')
        ax.set_title(f'{selected_company} Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
    else:
        st.warning("No actual data available for the predicted period.")
        st.dataframe(results.style.format({
            'Date': lambda x: x.strftime('%Y-%m-%d'),
            'Predicted Close': '{:.2f}'
        }))

# Display model performance
st.subheader('Model Performance')
st.write(f"Training R-squared: {train_r2:.4f}")
st.write(f"Test R-squared: {test_r2:.4f}")

# Plot training vs test predictions
fig2, ax2 = plt.subplots(figsize=(10, 6))
train_dates = company_data.loc[X_train.index, 'Date']
test_dates = company_data.loc[X_test.index, 'Date']
all_dates = pd.concat([train_dates, test_dates], axis=0)

ax2.plot(train_dates, y_train, label='Actual (Train)', color='blue')
ax2.plot(train_dates, train_pred, label='Predicted (Train)', color='orange', linestyle='--')
ax2.plot(test_dates, y_test, label='Actual (Test)', color='green')
ax2.plot(test_dates, test_pred, label='Predicted (Test)', color='red', linestyle='--')
for row in extern_data:
    data = company_data[localeRows[row]]
    ax2.plot(all_dates, data, label=row)

ax2.set_title(f'{selected_company} Model Performance')
ax2.set_xlabel('Date')
ax2.set_ylabel('Close Price')
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# Display raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(company_data)
