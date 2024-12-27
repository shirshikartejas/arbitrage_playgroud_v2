import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb
#import tensorflow as tf
import random
import time
import lightgbm as lgb

import seaborn as sns
sns.set_style('darkgrid')

st.set_page_config(page_title="Arbitrage Playground", page_icon=None, layout="centered", initial_sidebar_state="collapsed")

FORECAST_WINDOW_MIN=1

st.title("Arbitrage Playground")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools. Enter an Etherscan API key, your budget,  and click run to simulate performance."
)


threshold = st.slider('Select Budget', min_value=1000, max_value=40000, value=10000, step=500)
st.write(f'Selected Budget: {threshold}')

# fetch data from Etherscan API
@st.cache_data
def etherscan_request(action, api_key, address, startblock=0, endblock=99999999, sort='desc'):
    """
    fetch transactions from address
    
    Return: 

    """
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': startblock,
        'endblock': endblock,
        'sort': sort,
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}")
        return None, None
    
    data = response.json()
    if data['status'] != '1':
        st.error(f"API returned an error: {data['result']}")
        return None, None
    
    df = pd.DataFrame(data['result'])
    
    expected_columns = ['hash', 'blockNumber', 'timeStamp', 'from', 'to', 'gas', 'gasPrice', 'gasUsed', 'cumulativeGasUsed', 'confirmations', 'tokenSymbol', 'value', 'tokenName']
    
    for col in expected_columns:
        if col not in df.columns:
            raise Exception(f"Expected column '{col}' is missing from the response")
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Set Transaction Value in Appropriate Format
    df['og_value'] = df['value'].copy()
    df['value'] = np.where(df['tokenDecimal']=='6', df['value']/100000, df['value']/1000000000000000000)

    # Sort by timestamp in descending order and select the most recent 10,000 trades
    df['timeStamp'] = pd.to_numeric(df['timeStamp'])
    df = df.sort_values(by='timeStamp', ascending=False).head(10000)
    
    consolidated_data = {}

    for index, row in df.iterrows():
        tx_hash = row['hash']
        
        if tx_hash not in consolidated_data:
            consolidated_data[tx_hash] = {
                'blockNumber': row['blockNumber'],
                'timeStamp': row['timeStamp'],
                'hash': tx_hash,
                'from': row['from'],
                'to': row['to'],
                'WETH_value': 0,
                'USDC_value': 0,
                'tokenName_WETH': '',
                'tokenName_USDC': '',
                'gas': row['gas'],
                'gasPrice': row['gasPrice'],
                'gasUsed': row['gasUsed'],
                'cumulativeGasUsed': row['cumulativeGasUsed'],
                'confirmations': row['confirmations']
            }
        
        if row['tokenSymbol'] == 'WETH':
            consolidated_data[tx_hash]['WETH_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_WETH'] = row['tokenName']
        elif row['tokenSymbol'] == 'USDC':
            consolidated_data[tx_hash]['USDC_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_USDC'] = row['tokenName']

    return pd.DataFrame.from_dict(consolidated_data, orient='index')

def shift_column_by_time(df, time_col, value_col, shift_minutes):
    """
    The purpose of this method is to create a shifted set of columns
    that will act as labels downstream model.

    Returns: the same df with an additional column f"{value_col}_label"
    """
    # Ensure 'time_col' is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort the DataFrame by time
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Create an empty column for the shifted values
    df[f'{value_col}_label'] = None

    # Iterate over each row and find the appropriate value at least shift_minutes minutes later
    for i in range(len(df)):
        current_time = df.loc[i, time_col]
        future_time = current_time + pd.Timedelta(minutes=shift_minutes)
        
        # Find the first row where the time is greater than or equal to the future_time
        future_row = df[df[time_col] >= future_time]
        if not future_row.empty:
            df.at[i, f'{value_col}_label'] = future_row.iloc[0][value_col]
    
    return df

def str_to_datetime(s):
    split = s.split(' ')
    date_part, time_part = split[0], split[1]
    year, month, day = map(int, date_part.split('-'))
    hour, minute, second = map(int, time_part.split(':'))
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

@st.cache_data
def merge_pool_data(p0,p1):
    #Format P0 and P0 variables of interest
    p0['time'] = p0['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p0['p0.weth_to_usd_ratio'] = p0['WETH_value']/p0['USDC_value']
    p0['gasPrice'] = p0['gasPrice'].astype(float)
    p0['gasUsed']= p0['gasUsed'].astype(float)
    p0['p0.gas_fees_usd'] = (p0['gasPrice']/1e9)*(p0['gasUsed']/1e9)*p0['p0.weth_to_usd_ratio']
    p1['time'] = p1['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p1['p1.weth_to_usd_ratio'] = p1['WETH_value']/p1['USDC_value']
    p1['gasPrice'] = p1['gasPrice'].astype(float)
    p1['gasUsed']= p1['gasUsed'].astype(float)
    p1['p1.gas_fees_usd'] = (p1['gasPrice']/1e9)*(p1['gasUsed']/1e9)*p1['p1.weth_to_usd_ratio']

    #Merge Pool data
    both_pools = pd.merge(p0[['time','timeStamp','p0.weth_to_usd_ratio','p0.gas_fees_usd']],
                          p1[['time','timeStamp','p1.weth_to_usd_ratio','p1.gas_fees_usd']],
                          on=['time','timeStamp'], how='outer'
                         ).sort_values(by='timeStamp')
    both_pools = both_pools.ffill().reset_index(drop=True)
    both_pools = both_pools.dropna()
    both_pools['percent_change'] = (both_pools['p0.weth_to_usd_ratio'] - both_pools['p1.weth_to_usd_ratio'])/both_pools[['p0.weth_to_usd_ratio','p1.weth_to_usd_ratio']].min(axis=1)
    both_pools['total_gas_fees_usd'] = both_pools['p0.gas_fees_usd'] + both_pools['p1.gas_fees_usd']
    
    # Replace inf with NaN
    both_pools.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values (which were originally inf)
    both_pools.dropna(inplace=True)
    return both_pools
    
def XGB_preprocessing(both_pools):
    int_df = both_pools.select_dtypes(include=['datetime64[ns]','int64', 'float64'])
    int_df = int_df[['time','total_gas_fees_usd']]
    df_3M = shift_column_by_time(int_df, 'time', 'total_gas_fees_usd', FORECAST_WINDOW_MIN)
    df_3M.index = df_3M.pop('time')
    df_3M.index = pd.to_datetime(df_3M.index)

    num_lags = 9  # Number of lags to create
    for i in range(1, num_lags + 1):
        df_3M[f'lag_{i}'] = df_3M['total_gas_fees_usd'].shift(i)
    
    df_3M['rolling_mean_3'] = df_3M['total_gas_fees_usd'].rolling(window=3).mean()
    df_3M['rolling_mean_6'] = df_3M['total_gas_fees_usd'].rolling(window=6).mean()
    df_nan = df_3M[df_3M['total_gas_fees_usd_label'].isna()]
    
    df_3M.dropna(inplace=True)
    lag_features = [f'lag_{i}' for i in range(1, num_lags + 1)]
    X_gas_test = df_3M[['total_gas_fees_usd']+lag_features + ['rolling_mean_3', 'rolling_mean_6']]
    y_gas_test = df_3M['total_gas_fees_usd_label']
    
    df_nan = df_nan[['total_gas_fees_usd', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
       'lag_9', 'rolling_mean_3', 'rolling_mean_6']]

    return df_nan, X_gas_test, y_gas_test
    
def Final_results_processing(dates_test,y_test,test_predictions,y_gas_test,y_gas_pred):
    df_percent_change = pd.DataFrame({
        'time': dates_test,
        'percent_change_actual': y_test,
        'percent_change_prediction': test_predictions
    })
    
    df_gas = y_gas_test.to_frame()
    df_gas = df_gas.reset_index()
    df_gas['gas_fees_prediction'] = y_gas_pred
    
    df_final = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df_final = df_final.dropna()
    df_final['min_amount_to_invest_prediction'] = df_final.apply(
        lambda row: row['gas_fees_prediction'] /
                    (
                        (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003)
                    ),
        axis=1
    )
    
    df_final['Profit'] = df_final.apply(
        lambda row: (row['min_amount_to_invest_prediction'] * 
                     (1 + row['percent_change_actual']) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                     row['min_amount_to_invest_prediction'] * (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003) -
                     row['total_gas_fees_usd_label']),
        axis=1
    )
    
    df_final['Double_Check'] = df_final.apply(
        lambda row: (row['min_amount_to_invest_prediction'] * 
                     (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                     row['min_amount_to_invest_prediction'] * (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003) -
                     row['gas_fees_prediction']),
        axis=1
    )
    return df_final

def LGBM_Preprocessing(both_pools):
    """
    Creating evaluation data from the pool.  The model for LGBM is predicting percent_change
    across the two pools.  

    Return: 
    - int_df (nans dropped): all columns 'time','percent_change_label', 'percent_change','rolling_mean_8','lag_1','lag_2'
    - df_nan: data frame with all columns as above, but only keeping the rows where the percent_change_label is nan.  
                i.e. the first shift_minutes of data.
    - X_pct_test - only the input features for LGBM.
    - y_pct_test - only the labels column
    """
    int_df = both_pools.copy()
    int_df = int_df[['time','percent_change']]
    int_df = shift_column_by_time(int_df, 'time', 'percent_change', FORECAST_WINDOW_MIN)
    num_lags = 2  # Number of lags to create
    for i in range(1, num_lags + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)
    int_df['rolling_mean_8'] = int_df['percent_change'].rolling(window=8).mean()
    int_df = int_df[['time','percent_change_label', 'percent_change','rolling_mean_8','lag_1','lag_2']]
    df_nan = int_df[int_df['percent_change_label'].isna()]
    
    int_df.dropna(inplace=True)
    
    X_pct_test = int_df[['percent_change','rolling_mean_8','lag_1','lag_2']]
    y_pct_test = int_df['percent_change_label']
    return int_df, df_nan, X_pct_test, y_pct_test

def final_min_amt_invest(dates_test, test_predictions, index, gas_predictions):
    df_percent_change = pd.DataFrame({
            'time': dates_test,
            'percent_change_prediction': test_predictions
        })
    df_gas = pd.DataFrame({
            'time': index,
            'gas_fees_prediction': gas_predictions
        })
    df_final = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df_final = df_final.dropna()
    df_final['min_amount_to_invest_prediction'] = df_final.apply(
        lambda row: row['gas_fees_prediction'] /
                    (
                        (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003)
                    ),
        axis=1
    )
    return df_final


def display_current_arbitrage(df_min):
    now = datetime.now()
        
    # Difference between current time and given timestamp
    time_difference = now - df_min['time'].iloc[-1]
    
    # Check if the difference is less than 5 minutes
    is_less_than_five_minutes = time_difference < timedelta(minutes=FORECAST_WINDOW_MIN)
    
    if is_less_than_five_minutes:
        if df_min['min_amount_to_invest_prediction'].iloc[-1] < 0:
            print(f'Arbitrage Opportunity does not exist five minutes after {df_min["time"].iloc[-1]}')
        else:
            if df_min['percent_change_prediction'].iloc[-1] < 0:
                print(f'Buy Pool 1 ({pool1_address}) and Sell in Pool 2 ({pool2_address})\n Minimum amount to invest {df_min["min_amount_to_invest_prediction"].iloc[-1]} ten minutes after {df_min["time"].iloc[-1]}')
            else:
                print(f'Pool 1 ({pool1_address} \n Pool 2 ({pool2_address}) \n Buy Pool 2 and Sell in Pool 1 \n Minimum amount to invest {df_min["min_amount_to_invest_prediction"].iloc[-1]} {FORECAST_WINDOW_MIN} minutes after {df_min["time"].iloc[-1]}')
                
    else:
        print(f"Last Data point received from query was at {df_min['time'].iloc[-1]}\nData queried is greater than {FORECAST_WINDOW_MIN} minutes old, unable to provide minimum amount to invest")


@st.cache_resource
def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    base_model_path = os.path.join(models_dir, model_name)
    
    # Check for different possible file extensions
    possible_extensions = ['', '.h5', '.pkl', '.joblib']
    model_path = next((base_model_path + ext for ext in possible_extensions if os.path.exists(base_model_path + ext)), None)
    
    if model_path is None:
        st.error(f"Model file not found for: {model_name}")
        return None
    
    try:
        #if model_name.startswith("LSTM"):
        #    model = tf.keras.models.load_model(model_path)
        #else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Sidebar
st.sidebar.header("API Configuration")



# API key input
api_key = st.sidebar.text_input("Etherscan API Key", "16FCD3FTVWC3KDK17WS5PTWRQX1E2WEYV2")
pool0_address = st.sidebar.text_input("Pool 0 Address", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")
pool1_address = st.sidebar.text_input("Pool 1 Address", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")


st.sidebar.markdown(
    '[Back to Main Page (mydiamondhands)](https://mydiamondhands.io/)',
    unsafe_allow_html=True
)

if st.button("Run Analysis"):
    with st.spinner("Fetching and processing data..."):
        # Fetch and process data for both pools
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)
        
        if p0 is None or p1 is None:
            st.error("Failed to fetch data from Etherscan. Please check your API key and try again.")
        else:
            both_pools = merge_pool_data(p0, p1)

       
        
            # LGBM Preprocessing
            LGBM_org_data, df_final_LGBM_X_test, X_pct_test, y_pct_test = LGBM_Preprocessing(both_pools)
        
            # XGB Preprocessing
            df_final_XGB_X_test, X_gas_test, y_gas_test = XGB_preprocessing(both_pools)
        
            if df_final_LGBM_X_test is None or X_pct_test is None or y_pct_test is None or df_final_XGB_X_test is None or X_gas_test is None or y_gas_test is None:
                st.error("Preprocessing failed. Cannot proceed with analysis.")
            else:
                test_predictions = None
                y_gas_pred = None

                # Run LGBM model
                with st.spinner("Running LGBM model..."):
                    LGBM = load_model(f"percent_change_{FORECAST_WINDOW_MIN}min_forecast_LGBM")
                    if LGBM is not None:
                        try:
                            y_pct_pred = LGBM.predict(X_pct_test, num_iteration=LGBM.best_iteration)
                            y_pct_rmse = root_mean_squared_error(y_pct_test, y_pct_pred)
                            y_pct_r2 = r2_score(y_pct_test, y_pct_pred)
                            
                        except Exception as e:
                            st.error(f"Error running LGBM model: {str(e)}")
                    else:
                        st.error("Failed to load LGBM model. Skipping LGBM analysis.")

                # Run XGB model
                with st.spinner("Running XGB model..."):
                    XGB = load_model(f"gas_fees_{FORECAST_WINDOW_MIN}min_forecast_XGBoost")
                    if XGB is not None:
                        try:
                            y_gas_pred = XGB.predict(X_gas_test)
                            y_gas_rmse = root_mean_squared_error(y_gas_test, y_gas_pred)
                            y_gas_r2 = r2_score(y_gas_test, y_gas_pred)


                        except Exception as e:
                            st.error(f"Error running XGB model: {str(e)}")
                    else:
                        st.error("Failed to load XGB model. Skipping XGB analysis.")

                # Process final results
                if y_pct_pred is not None and y_gas_pred is not None:
                    df_final = Final_results_processing(LGBM_org_data['time'],y_pct_test,y_pct_pred,y_gas_test,y_gas_pred)

                    #to compensate for the for loop
                    threshold = threshold + 100
                    thresholds = list(range(1000, threshold, 100))
                    net_gains = []
                    total_losses = []
                    total_gains = []
                    
                   

                    for threshold in thresholds:
                        # Filter the DataFrame for gains
                        df_gain = df_final[(df_final['Profit'] > 0) 
                                           & (df_final['min_amount_to_invest_prediction'] > 0) 
                                           & (df_final['min_amount_to_invest_prediction'] < threshold)]
                        total_gain = df_gain['Profit'].sum()
                        
                        # Filter the DataFrame for losses
                        df_loss = df_final[((df_final['Profit'] < 0) 
                                        & (df_final['min_amount_to_invest_prediction'] > 0) 
                                        & (df_final['min_amount_to_invest_prediction'] < threshold))]
                        df_loss = df_final[((df_final['Profit'] < 0) 
                                        & (df_final['min_amount_to_invest_prediction'] > 0) 
                                        & (df_final['min_amount_to_invest_prediction'] < threshold))]
                        total_loss = df_loss['Profit'].sum()
                        
                        # Calculate net gain
                        total_losses.append(total_loss)
                        total_gains.append(total_gain)
                        
                        net_gain = total_gain + total_loss
                        net_gains.append(net_gain)
                    
                    #to compensate for the for loop
                    #threshold = threshold - 100
                    
                    # Display results
                    #st.subheader("Raw Results")
                    #st.subheader("Potential Returns based on your Budget")
                    total_gain = total_gains[-1]
                    #st.write(f"Total Potential Gain: ${total_gain:.2f}")
                    #st.write(df_final)

                    
                    total_loss = total_losses[-1]
                    #st.write(f"Total Potential Loss: ${total_loss:.2f}")
                    #st.write(df_loss)

                    experiment_duration = df_final['time'].iloc[-1] - df_final['time'].iloc[0]
                    avg_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].mean()
                    median_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].median()

                    avg_profit = df_final['Profit'].mean()
                    med_profit = df_final['Profit'].median()

                    number_of_simulated_swaps = df_final.shape[0]


                    st.subheader(f'Selected Budget Simulated Results')
                    st.write(f"Simulation Duration: {experiment_duration.total_seconds() / 3600:.1f} hour(s)")
                    st.write(f"Number of Transactions used in Simulation: {number_of_simulated_swaps}")
                    #st.write(f"Budget threshold: ${threshold:.2f}")
                    #st.write(f"Total Net Gain: ${net_gain:.2f}")
                    st.write(f"Average Recommended Minimum Investment: ${avg_positive_min_investment:.2f}")
                    st.write(f"Median Recommended Minimum Investment: ${median_positive_min_investment:.2f}")
                    st.write(f"Percent of Transactions with Profitable Outcomes: {df_gain.shape[0]/df_final.shape[0]*100:.1f}%")
                    st.write(f"Median Profit Per Transaction: ${med_profit:.2f}")
                    #st.write(f"Average Profit Per Transaction: ${avg_profit:.2f}")
                    
                    # Plot Net Gain vs. Minimum Amount to Invest
                    print(f"Time Duration: {experiment_duration}")

                    st.subheader(f"Recommended Minimum Investment in the last {experiment_duration.total_seconds() / 3600:.1f} hour(s).")

                    # Create a single figure with two subplots (1 row, 2 columns)
                    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

                    # First plot (scatter plot)
                  
                    axs[0].scatter(df_final['time'], df_final['min_amount_to_invest_prediction'], marker='o')
                    axs[0].set_xlabel('time')
                    axs[0].set_ylabel('Recommended Minimum Amount to Invest')
                    max_ylim = df_final['min_amount_to_invest_prediction'].max()

                    # Add horizontal line at the threshold
                    axs[0].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
                    # Add annotation to the line
                    axs[0].annotate(
                        "Selected Budget",
                        xy=(df_final['time'].min(), threshold),  # Far right of the plot
                        xytext=(10, 5),  # Offset to position the text just above the line
                        textcoords="offset points",
                        fontsize=8,  # Small font size
                        color='black',  # Same color as the line
                        ha='left',  # Align text to the left
                        va='bottom'  # Align text above the line
                    )

                    #axs[0].set_ylim(0, max_ylim * 1.2)
                    axs[0].set_yscale('log')
                    axs[0].set_title(f'Scatter Plot of Recommended Minimum Investment in last {experiment_duration.total_seconds() / 3600:.1f} hour(s)')

                    # Format the x-axis to show HH:MM:SS
                    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency
                    #fig.autofmt_xdate()  # Rotates the labels to prevent overlap


                    # Second plot (histogram)
                    df_final_fig1b = df_final[
                        (df_final['min_amount_to_invest_prediction'] > 0) & 
                        (df_final['min_amount_to_invest_prediction'] < threshold)
                    ][['min_amount_to_invest_prediction']]

                    # Calculate histogram data
                    data = df_final_fig1b['min_amount_to_invest_prediction']
                    hist_values, bin_edges = np.histogram(data, bins=50)

                    # Find the largest bin
                    max_bin_index = np.argmax(hist_values)
                    max_bin_count = hist_values[max_bin_index]

                    # Calculate the center and width of the largest bin
                    bin_width = bin_edges[1] - bin_edges[0]
                    bin_center = bin_edges[max_bin_index] + bin_width / 2

                    print(f"Bin Width: {bin_width}, Bin Center: {bin_center}")
                    #
                    #  Annotate the largest bin
                    axs[1].annotate(
                        f"Bin Center: {bin_center:.2f} +/- {bin_width/2:.2f}\n",
                        xy=(bin_center, max_bin_count),
                        xytext=(bin_center + bin_width, max_bin_count*0.8),  # Middle of the plot, slightly to the right
                        textcoords='data',
                        #arrowprops=dict(facecolor='black', arrowstyle='--'),
                        fontsize=8,
                        ha='left'
                    )

                    # Plot the histogram
                    axs[1].hist(data, bins=50, alpha=0.7)
                    axs[1].set_xlabel('Recommended Minimum Amount to Invest')
                    axs[1].set_title(f'Distribution of Recommended Minimum Investment in last {experiment_duration.total_seconds() / 3600:.1f} hour(s)')
                    axs[1].set_xlim(0, threshold)

                    # Display the figure in Streamlit
                    st.pyplot(fig)

                    net_gain = total_gain + total_loss

                    
                    
                    #st.subheader("Net Gain vs. Minimum Amount to Invest")
                    #st.write(f"This graph is created by simulating {number_of_simulated_swaps} transactions using actual market conditions in the last {experiment_duration.total_seconds() / 3600:.1f} hour(s).  The simulation assumes that if the transactions during the day were within your budget you invested the minimum amount predicted by the model each time.")

                    ## Create a DataFrame for plotting
                    #results_df = pd.DataFrame({
                    #    'Threshold': thresholds,
                    #    'Net Gain': net_gains
                    #})
                    
                    #experimental_df = results_df[results_df['Net Gain']>0]

                    ## Plot the results
                    #fig2 = plt.figure(figsize=(10, 6))
                    #plt.plot(results_df['Threshold'], results_df['Net Gain'], marker='o')
                    #plt.title('Net Gain vs. Minimum Amount to Invest')
                    #plt.xlabel('Minimum Amount to Invest')
                    #plt.ylabel('Net Gain')
                    #plt.yscale('log')
                    ##plt.ylim(-1000000, 1000000)
                    #plt.grid(True)
                    ##plt.show()
                    #st.pyplot(fig2)
                    
                    
                    st.subheader(f'Recommended Minimum Investment Prediction in next {FORECAST_WINDOW_MIN} minute(s)')

                    # Predictions
                    df_final_LGBM_X_test_dates = df_final_LGBM_X_test['time']
                    df_final_LGBM_X_test = df_final_LGBM_X_test[['percent_change', 'rolling_mean_8', 'lag_1', 'lag_2']]
                    y_final_pct_pred = LGBM.predict(df_final_LGBM_X_test, num_iteration=LGBM.best_iteration)
                    y_final_gas_pred = XGB.predict(df_final_XGB_X_test)
                    
                    df_min = final_min_amt_invest(df_final_LGBM_X_test_dates, y_final_pct_pred, df_final_XGB_X_test.index, y_final_gas_pred)

                    now = datetime.now()

                    # Difference between current time and last timestamp
                    time_difference = now - df_min['time'].iloc[-1]

                    # Check if the difference is less than 10 minutes
                    is_less_than_ten_minutes = time_difference < timedelta(minutes=FORECAST_WINDOW_MIN)
                    ten_minutes = timedelta(minutes=FORECAST_WINDOW_MIN)
                    if is_less_than_ten_minutes:
                        if df_min['min_amount_to_invest_prediction'].iloc[-1] < 0:
                            st.write(f'Arbitrage Opportunity is not expected {FORECAST_WINDOW_MIN} minute(s) from now.')
                        else:
                            if df_min['percent_change_prediction'].iloc[-1] < 0:
                                st.write(f'Buy Pool 0 ({pool0_address}) and Sell in Pool 1 ({pool1_address})\n Minimum amount to invest {df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+ten_minutes}')
                            else:
                                st.write(f'Buy Pool 1 ({pool1_address}) and Sell in Pool 0 ({pool0_address})\n Minimum amount to invest {df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+ten_minutes}')
                                
                    else:
                        st.write(f"Last Data point received from query was at {df_min['time'].iloc[-1]}\nData queried is greater than {FORECAST_WINDOW_MIN} minute(s) old, unable to provide minimum amount to invest")
                    


                    
                    
                else:
                    st.error("Cannot proceed with final processing. Some model predictions are missing.")

        st.write(
            "Disclaimer: The creators of this app are not licensed to provide, offer or recommend financial instruments in any way shape or form. This information and the information within the site should not be considered unique financial advice and if you consider utilizing the model, or investing in crypto you should first seek financial advice from a trained professional, to ensure that you fully understand the risk. Further, while modelling efforts have been undertaken in an effort to avoid risk through the application of the principles of arbitrage, the model has not been empirically tested, and should be approached with extreme caution, care and be utilized at one’s own risk (do not make trades to which you would be unable to fulfill, or would be in a detrimental financial position if it was to not complete as expected)."
        )

        # Display Raw Table results


        st.subheader("Raw Results")
        st.write(df_final)

        st.subheader("XGB Gas Fee Model Results")
        st.write(f"Root Mean Squared Error: {y_gas_rmse:.4f}")
        st.write(f"R² Score: {y_gas_r2:.4f}")

        st.subheader("LGBM Price Model Results")
        st.write(f"Root Mean Squared Error: {y_pct_rmse:.4f}")
        st.write(f"R² Score: {y_pct_r2:.4f}")


