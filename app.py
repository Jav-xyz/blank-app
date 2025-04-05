import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📈 Stock Price Predictor")
st.markdown("""
Predict stock prices 2 years into the future and validate against historical data.
Select a company and date to see what the model would have predicted.
""")

# Constants
DATA_DIR = './stocks2/'  # Update this path as needed
COMPANIES = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']  # Add more as needed

class StockPredictor:
    def __init__(self, company='AAPL'):
        self.company = company
        self.model = None
        self.df = None
        self.feature_cols = None
        
    @st.cache_data
    def load_and_preprocess_data(_self, company):
        """Load and preprocess the quarterly stock data"""
        file_path = os.path.join(DATA_DIR, f"{company}.csv")
        df = pd.read_csv(file_path)
        
        # Convert date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    
    def create_features(self, forecast_horizon=8):
        """
        Create features for the model
        forecast_horizon: Number of quarters to predict ahead (8 = 2 years)
        """
        # Create target - price forecast_horizon quarters in the future
        self.df['target'] = self.df['Close'].shift(-forecast_horizon)
        
        # Basic price features
        for window in [1, 2, 4, 8]:  # Different lookback periods
            self.df[f'pct_change_{window}q'] = self.df['Close'].pct_change(window)
            self.df[f'rolling_avg_{window}q'] = self.df['Close'].rolling(window=window).mean()
            self.df[f'rolling_std_{window}q'] = self.df['Close'].rolling(window=window).std()
        
        # Volume features
        self.df['volume_pct_change'] = self.df['Volume'].pct_change()
        for window in [2, 4]:
            self.df[f'volume_rolling_avg_{window}q'] = self.df['Volume'].rolling(window=window).mean()
        
        # Technical indicators
        self.df['high_low_spread'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['close_open_spread'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        self.df['volatility'] = self.df['High'] / self.df['Low'] - 1
        
        # Drop rows with NaN values (from target shift and rolling calculations)
        self.df = self.df.dropna()
        return self.df
    
    def train_model(self, test_size=0.2):
        """Train and evaluate the Random Forest model"""
        # Define feature columns
        self.feature_cols = [col for col in self.df.columns 
                            if col not in ['Date', 'target', 'Unnamed: 0']]
        X = self.df[self.feature_cols]
        y = self.df['target']
        
        # Split data chronologically
        split_idx = int(len(self.df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8,
            min_samples_split=5,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        return self.model, X_test, y_test
    
    def predict_from_date(self, input_date):
        """
        Make a 2-year prediction from a specific historical date
        and compare with actual values if available
        """
        # Convert input date to datetime
        if isinstance(input_date, str):
            input_date = pd.to_datetime(input_date)
        
        # Find the closest date in the data
        idx = self.df['Date'].sub(input_date).abs().idxmin()
        prediction_date = self.df.loc[idx, 'Date']
        
        # Get features for prediction
        X = self.df.loc[idx, self.feature_cols].values.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        actual = self.df.loc[idx, 'target'] if idx + 8 < len(self.df) else np.nan
        
        return prediction_date, prediction, actual

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    company = st.selectbox("Select Company", COMPANIES)
    
    # Load data to get date range
    predictor = StockPredictor()
    df = predictor.load_and_preprocess_data(company)
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime() - timedelta(days=730)  # 2 years before end
    
    prediction_date = st.date_input(
        "Prediction Date",
        value=min_date + timedelta(days=365*5),  # Default to 5 years after start
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Select a company and historical date
    2. The model predicts what the price would be 2 years later
    3. Compares with actual price (if available)
    """)

# Main app
tab1, tab2 = st.tabs(["Prediction", "Model Details"])

with tab1:
    if st.button("Run Prediction"):
        with st.spinner("Training model and making prediction..."):
            # Initialize and train model
            predictor = StockPredictor()
            df = predictor.load_and_preprocess_data(company)
            predictor.df = df
            predictor.create_features()
            model, X_test, y_test = predictor.train_model()
            
            # Make prediction
            prediction_date, prediction, actual = predictor.predict_from_date(prediction_date)
            prediction_date = prediction_date.to_pydatetime()
            future_date = prediction_date + timedelta(days=730)  # 2 years
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction Date", prediction_date.strftime("%Y-%m-%d"))
                st.metric("Price on Date", f"${df[df['Date'] == prediction_date]['Close'].values[0]:.2f}")
            
            with col2:
                st.metric("Predicted Future Price", f"${prediction:.2f}", 
                         delta=f"{((prediction/df[df['Date'] == prediction_date]['Close'].values[0])-1)*100:.1f}%")
            
            with col3:
                if not np.isnan(actual):
                    st.metric("Actual Future Price", f"${actual:.2f}", 
                             delta=f"{((actual/df[df['Date'] == prediction_date]['Close'].values[0])-1)*100:.1f}%")
                    error_pct = (prediction - actual) / actual * 100
                    st.metric("Prediction Error", f"{error_pct:.1f}%")
                else:
                    st.metric("Actual Future Price", "Not available")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical data
            hist_df = df[df['Date'] <= prediction_date]
            ax.plot(hist_df['Date'], hist_df['Close'], 'b-', label='Historical Prices')
            
            # Prediction point
            ax.plot(prediction_date, hist_df['Close'].iloc[-1], 'bo')
            
            # Actual future if available
            if not np.isnan(actual):
                future_df = df[(df['Date'] > prediction_date) & 
                              (df['Date'] <= future_date)]
                ax.plot(future_df['Date'], future_df['Close'], 'g-', label='Actual Future')
                ax.plot(future_date, actual, 'go')
            
            # Predicted future
            ax.plot(future_date, prediction, 'ro', label='Predicted Future')
            
            ax.set_title(f'{company} Stock Price Prediction from {prediction_date.strftime("%Y-%m-%d")}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid()
            
            st.pyplot(fig)
            
            # Show recent data
            st.subheader("Recent Data Points")
            st.dataframe(df[['Date', 'Close', 'Volume']].tail(10).sort_values('Date', ascending=False))

with tab2:
    st.header("Model Information")
    st.markdown("""
    **Model Type:** Random Forest Regressor  
    **Prediction Horizon:** 2 years (8 quarters)  
    **Features Used:**
    - Price changes over different lookback periods
    - Rolling averages and standard deviations
    - Volume changes and averages
    - Technical indicators (price spreads, volatility)
    """)
    
    if 'predictor' in locals():
        st.subheader("Feature Importance")
        feature_importance = pd.Series(
            predictor.model.feature_importances_,
            index=predictor.feature_cols
        ).sort_values(ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        feature_importance.head(10).plot(kind='barh', ax=ax2)
        ax2.set_title('Top 10 Important Features')
        st.pyplot(fig2)
        
        st.subheader("Model Performance")
        test_preds = predictor.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        st.metric("Test RMSE", f"${rmse:.2f}")
    else:
        st.info("Run a prediction first to see model details")

# Footer
st.markdown("---")
st.markdown("""
*Note: This is for educational purposes only. Past performance is not indicative of future results.*
""")