import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    
    # Convert Date to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year_Quarter'] = df['Year'].astype(str) + 'Q' + df['Quarter'].astype(str)
    
    # Calculate additional financial metrics
    df['Profit_Margin'] = df['total_profit'] / df['total_revenue']
    df['Spending_Ratio'] = df['total_spending'] / df['total_revenue']
    
    # Sort by Company and Date
    df = df.sort_values(['Company', 'Date'])
    
    return df

def prepare_features(df, company, selected_features):
    # Filter for selected company
    company_df = df[df['Company'] == company].copy()
    
    # Create lag features
    for lag in [1, 2, 3, 4]:
        company_df[f'Close_lag_{lag}'] = company_df['Close'].shift(lag)
        company_df[f'Volume_lag_{lag}'] = company_df['Volume'].shift(lag)
        company_df[f'Profit_Margin_lag_{lag}'] = company_df['Profit_Margin'].shift(lag)
    
    # Drop rows with NaN values from lag features
    company_df = company_df.dropna()
    
    # Base features available for selection
    all_features = [
        'GDP_per_capitas', 'Federal_Funds', 'Unemployment_Rates', 'Inflation_rate',
        'total_profit', 'total_spending', 'total_revenue', 'Profit_Margin', 'Spending_Ratio',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4',
        'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_4',
        'Profit_Margin_lag_1', 'Profit_Margin_lag_2',
        'Year', 'Month', 'Quarter'
    ]
    # Use selected features if provided, otherwise use all
    features_to_use = selected_features or all_features
    
    X = company_df[features_to_use]
    y = company_df['Close']
    
    return X, y, company_df, features_to_use

def train_model(X, y, test_size=0.2, n_estimators=200):
    # Split data (maintaining chronological order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, min_samples_split=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    return model, rmse, train_r2, test_r2, mape, X_test.index, y_test, y_pred, X_train.index, y_train, train_pred

def predict_future(model, df, company, prediction_date, features_used):
    # Convert prediction_date to datetime
    prediction_date = pd.to_datetime(prediction_date)
    year = prediction_date.year
    month = prediction_date.month
    quarter = (prediction_date.month - 1) // 3 + 1
    
    # Get the most recent data before prediction date
    company_data = df[df['Company'] == company]
    recent_data = company_data[company_data['Date'] < prediction_date]
    
    if len(recent_data) == 0:
        return None, "No historical data available before the prediction date"
    
    latest_record = recent_data.sort_values('Date', ascending=False).iloc[0]
    lag_records = recent_data.sort_values('Date', ascending=False)
    
    # Prepare feature dictionary
    feature_map = {
        'GDP_per_capitas': latest_record['GDP_per_capitas'],
        'Federal_Funds': latest_record['Federal_Funds'],
        'Unemployment_Rates': latest_record['Unemployment_Rates'],
        'Inflation_rate': latest_record['Inflation_rate'],
        'total_profit': latest_record['total_profit'],
        'total_spending': latest_record['total_spending'],
        'total_revenue': latest_record['total_revenue'],
        'Profit_Margin': latest_record['Profit_Margin'],
        'Spending_Ratio': latest_record['Spending_Ratio'],
        'Close_lag_1': latest_record['Close'],
        'Close_lag_2': lag_records.iloc[1]['Close'] if len(lag_records) > 1 else latest_record['Close'],
        'Close_lag_3': lag_records.iloc[2]['Close'] if len(lag_records) > 2 else latest_record['Close'],
        'Close_lag_4': lag_records.iloc[3]['Close'] if len(lag_records) > 3 else latest_record['Close'],
        'Volume_lag_1': latest_record['Volume'],
        'Volume_lag_2': lag_records.iloc[1]['Volume'] if len(lag_records) > 1 else latest_record['Volume'],
        'Volume_lag_3': lag_records.iloc[2]['Volume'] if len(lag_records) > 2 else latest_record['Volume'],
        'Volume_lag_4': lag_records.iloc[3]['Volume'] if len(lag_records) > 3 else latest_record['Volume'],
        'Profit_Margin_lag_1': latest_record['Profit_Margin'],
        'Profit_Margin_lag_2': lag_records.iloc[1]['Profit_Margin'] if len(lag_records) > 1 else latest_record['Profit_Margin'],
        'Year': year,
        'Month': month,
        'Quarter': quarter
    }
    
    # Select only the features used in training
    pred_features = {k: feature_map[k] for k in features_used}
    
    # Create DataFrame for prediction
    pred_df = pd.DataFrame([pred_features])
    
    # Make prediction
    predicted_price = model.predict(pred_df)
    
    predicted_price = predicted_price[0]
    
    # Get actual price if available
    actual_data = df[(df['Company'] == company) & (df['Date'] == prediction_date)]
    actual_price = actual_data['Close'].values[0] if not actual_data.empty else None
    
    # Prepare company status
    status = {
        'Company': company,
        'Prediction Date': prediction_date.strftime('%Y-%m-%d'),
        'Predicted Price': round(predicted_price, 2),
        'Latest Financial Data Date': latest_record['Date'].strftime('%Y-%m-%d'),
        'Total Profit': latest_record['total_profit'],
        'Total Spending': latest_record['total_spending'],
        'Total Revenue': latest_record['total_revenue'],
        'Profit Margin': f"{latest_record['Profit_Margin']:.2%}",
        'Spending Ratio': f"{latest_record['Spending_Ratio']:.2%}",
        'GDP per Capita': latest_record['GDP_per_capitas'],
        'Federal Funds Rate': latest_record['Federal_Funds'],
        'Unemployment Rate': latest_record['Unemployment_Rates'],
        'Inflation Rate': latest_record['Inflation_rate'],
        'Actual Price': actual_price,
        'Prediction Error': round(abs(predicted_price - actual_price), 2) if actual_price else None,
        'Prediction Error %': round(abs(predicted_price - actual_price)/actual_price*100, 2) if actual_price else None,
        'Features Used': ', '.join(features_used)
    }
    
    return status, None

def generate_random_hex_color():
    """Generates a random color in hex format."""
    hex_color = "#"
    for _ in range(6):
        hex_color += random.choice("0123456789abcdef")
    return hex_color
def add_external_data_plot(localeRows, rows, df):
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    for row in rows:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[localeRows[row]],
            mode='lines',
            name=row,
            marker=dict(color=generate_random_hex_color(), size=8),
            opacity=0.7
        ))
    # Update layout
    fig.update_layout(
        title="External Data",
        xaxis_title='Data',
        yaxis_title='Price ($)',
        legend_title='Legend',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    return fig

def create_prediction_chart(company_df, train_idx, test_idx, y_train, y_test, 
                          train_pred, test_pred, future_dates=None, future_pred=None):
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=company_df['Date'],
        y=company_df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#3498db', width=2),
        opacity=0.8
    ))
    
    # Plot training predictions
    fig.add_trace(go.Scatter(
        x=company_df.loc[train_idx, 'Date'],
        y=train_pred,
        mode='markers',
        name='Train Predictions',
        marker=dict(color='#2ecc71', size=8),
        opacity=0.7
    ))
    
    # Plot test predictions
    fig.add_trace(go.Scatter(
        x=company_df.loc[test_idx, 'Date'],
        y=test_pred,
        mode='markers',
        name='Test Predictions',
        marker=dict(color='#e74c3c', size=8),
        opacity=0.7
    ))
    
    # Plot future predictions if available
    if future_dates is not None and future_pred is not None:
        pass
        # fig.add_trace(go.Scatter(
        #     x=future_dates,
        #     y=future_pred,
        #     mode='lines+markers',
        #     name='Future Predictions',
        #     line=dict(color='#f39c12', width=2, dash='dot'),
        #     marker=dict(size=10, symbol='diamond'),
        #     opacity=0.9
        # ))
    
    # Update layout
    fig.update_layout(
        title=f"{company_df['Company'].iloc[0]} Stock Price Prediction",
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend_title='Legend',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    return fig

def main():
    st.set_page_config(layout="wide", page_title="WallStreetBot")
    
    localeRows = {
        "Reel GDP per capitas": "GDP_per_capitas",
        "Federals Funds": "Federal_Funds",
        "Unemployment Rates": "Unemployment_Rates",
        "Inflation Rates": "Inflation_rate",
        "Total Profit": "total_profit",
        "Total Spending": "total_spending",
        "Total Revenue": "total_revenue",
        ## you can uncoment the following line but the data is weird
    }
    # Custom CSS
    st.markdown("""
    <style>
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 14px;
            color: #555;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .header {
            color: #2c3e50;
        }
        .feature-selector {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 WallStreetBot")
    st.write("Predict stock prices with customizable features and extended timeline")
    
    # Load data
    df = load_data()
    companies = df['Company'].unique()
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Prediction Parameters")
        selected_company = st.selectbox("Select Company", companies)
        
        # Feature selection
        st.subheader("Feature Selection")
        st.write("Choose features for the prediction model:")
        
        feature_options = [
            'GDP_per_capitas', 'Federal_Funds', 'Unemployment_Rates', 'Inflation_rate',
            'total_profit', 'total_spending', 'total_revenue', 'Profit_Margin', 'Spending_Ratio',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4',
            'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_4',
            'Profit_Margin_lag_1', 'Profit_Margin_lag_2',
            'Year', 'Month', 'Quarter'
        ]
        
        selected_features = st.multiselect(
            "Select features to include in model",
            feature_options,
            default=['Close_lag_1', 'Close_lag_2', 'Profit_Margin', 'total_revenue', 'Inflation_rate']
        )
        
        # Date range for prediction
        min_date = df['Date'].min().to_pydatetime()
        max_date = df['Date'].max().to_pydatetime()
        default_date = min(max_date, datetime.now() + timedelta(days=90))

        external_data = st.multiselect("External Data to show", localeRows.keys())
        
        prediction_date = st.date_input(
            "Select Prediction Date",
            min_value=min_date,
            max_value=datetime(2025, 12, 31),
            value=default_date
        )
        
        st.markdown("---")
        st.markdown("**Model Settings**")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        n_estimators = st.slider("Number of Trees", 50, 500, 200)
    
    # Prepare features and train model
    X, y, company_df, features_used = prepare_features(df, selected_company, selected_features)
    
    if len(company_df) < 10:
        st.error(f"Not enough data to train model for {selected_company}. Only {len(company_df)} records available.")
        return
    
    model, rmse, train_r2, test_r2, mape, test_idx, y_test, test_pred, train_idx, y_train, train_pred = train_model(
        X, y, test_size=test_size/100, n_estimators=n_estimators
    )
    
    # Make prediction
    status, error = predict_future(model, df, selected_company, prediction_date, features_used)
    
    if error:
        st.error(error)
        return
    
    # Generate future predictions (up to 2025)
    future_dates = pd.date_range(
        start=company_df['Date'].max() + timedelta(days=1),
        end=datetime(2025, 12, 31),
        freq='ME'
    )
    
    future_predictions = []
    for date in future_dates:
        future_status, _ = predict_future(model, df, selected_company, date, features_used)
        future_predictions.append(future_status['Predicted Price'])
    
    # Display results in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Prediction for {status['Company']}")
        
        # Prediction cards
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Closing Price</div>
                <div class="metric-value">${status['Predicted Price']:,.2f}</div>
                <div class="metric-title">for {status['Prediction Date']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if status['Actual Price']:
            error_class = "positive" if status['Prediction Error %'] < 5 else "negative"
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Actual Closing Price</div>
                    <div class="metric-value">${status['Actual Price']:,.2f}</div>
                    <div class="metric-title">Prediction Error</div>
                    <div class="metric-value {error_class}">${status['Prediction Error']:,.2f} ({status['Prediction Error %']}%)</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive chart
        st.subheader("Price Prediction Chart")
        fig = create_prediction_chart(
            company_df, train_idx, test_idx, y_train, y_test,
            train_pred, test_pred, future_dates, future_predictions
        )

        #External Data

        data_fig = add_external_data_plot(localeRows,external_data, company_df)


        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(data_fig, use_container_width=True)

        
    
    with col2:
        st.subheader("Model Performance")
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr; gap: 10px;">
            <div class="metric-card">
                <div class="metric-title">Train R² Score</div>
                <div class="metric-value">{train_r2:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Test R² Score</div>
                <div class="metric-value">{test_r2:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Root Mean Squared Error</div>
                <div class="metric-value">{rmse:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Mean Absolute % Error</div>
                <div class="metric-value">{mape:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            feat_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = go.Figure(go.Bar(
                x=feat_importance['Importance'],
                y=feat_importance['Feature'],
                orientation='h',
                marker_color='#3498db'
            ))
            fig_importance.update_layout(
                title='Feature Importance',
                height=400,
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                margin=dict(l=150)
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")
    
    # Show model details and data
    with st.expander("Model Details and Data"):
        st.subheader("Selected Features")
        st.write(f"Features used in this model: {status['Features Used']}")
        
        st.subheader("Recent Financial Data")
        recent_data = company_df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
                                'total_profit', 'total_spending', 'total_revenue',
                                'Profit_Margin', 'Spending_Ratio']].tail(10)
        st.dataframe(recent_data.style.format({
            'Date': lambda x: x.strftime('%Y-%m-%d'),
            'Close': '${:,.2f}',
            'High': '${:,.2f}',
            'Low': '${:,.2f}',
            'Open': '${:,.2f}',
            'Volume': '{:,.0f}',
            'total_profit': '${:,.0f}',
            'total_spending': '${:,.0f}',
            'total_revenue': '${:,.0f}',
            'Profit_Margin': '{:.2%}',
            'Spending_Ratio': '{:.2%}'
        }))

if __name__ == "__main__":
    main()
