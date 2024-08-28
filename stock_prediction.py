import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import matplotlib.pyplot as plt
import mplfinance as mpf  #importing mpfinance library for candlestick chart

# Configuration parameters
N_STEPS = 50                # Number of historical steps to use for prediction
LOOKUP_STEP = 15            # Number of days to predict into the future
SCALE = True                # Whether to scale feature columns
SHUFFLE = True              # Whether to shuffle the dataset
SPLIT_BY_DATE = False       # Whether to split by date or randomly
TEST_SIZE = 0.2             # Ratio of the data to be used for testing
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]  # Features to use
date_now = "2021-05-31"     # Date for saving the model and data

# Model configuration
N_LAYERS = 2                # Number of LSTM layers
CELL = LSTM                 # LSTM cell type
UNITS = 256                 # Number of units in each LSTM layer
DROPOUT = 0.4               # Dropout rate
BIDIRECTIONAL = False       # Whether to use Bidirectional LSTM

# Training configuration
LOSS = "huber_loss"         # Loss function
OPTIMIZER = "adam"          # Optimizer
BATCH_SIZE = 64             # Batch size for training
EPOCHS = 500                # Number of epochs for training

ticker = "AMZN"             # Stock ticker symbol

# Get user input for start and end dates
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")


ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")  # Path for saving data
model_name = f"{date_now}_{ticker}-sh-{int(SHUFFLE)}-sc-{int(SCALE)}-sbd-{int(SPLIT_BY_DATE)}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"  # Append '-b' if using Bidirectional LSTM

def shuffle_in_unison(a, b):
    """
    Shuffle two arrays in the same way.
    
    Args:
        a (np.array): First array to shuffle.
        b (np.array): Second array to shuffle.
    """
    state = np.random.get_state()  # Save the random state
    np.random.shuffle(a)  # Shuffle the first array
    np.random.set_state(state)  # Restore the random state
    np.random.shuffle(b)  # Shuffle the second array with the same state

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Load and preprocess stock data from Yahoo Finance.
    
    Args:
        ticker (str/pd.DataFrame): Stock ticker or pre-loaded dataframe.
        n_steps (int): Number of historical steps to use for prediction.
        scale (bool): Whether to scale the features.
        shuffle (bool): Whether to shuffle the data.
        lookup_step (int): Number of steps to predict into the future.
        split_by_date (bool): Whether to split data by date or randomly.
        test_size (float): Ratio of test data size.
        feature_columns (list): List of feature columns to use.
        
    Returns:
        dict: Processed data including training and testing sets, scalers, and more.
    """
    # Load data from Yahoo Finance if ticker is a string, or use pre-loaded dataframe
    if isinstance(ticker, str):
        df = si.get_data(ticker)  
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker must be a string or a pandas DataFrame")

    result = {'df': df.copy()}  # Initialize result dictionary to store data

    # Ensure all feature columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    if "date" not in df.columns:
        df["date"] = df.index  # Add date column if not present

    # Scale feature columns if specified
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler  # Store scalers for later use

    df['future'] = df['adjclose'].shift(-lookup_step)  # Create future price column for prediction

    last_sequence = np.array(df[feature_columns].tail(lookup_step))  # Extract the last sequence of features
    df.dropna(inplace=True)  # Remove rows with NaN values

    sequence_data = []
    sequences = deque(maxlen=n_steps)  # Initialize deque to store sequences

    # Create sequences of historical data and corresponding targets
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Append the last sequence to the results
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # Extract dates for the test set and remove duplicates
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """
    Create and compile an LSTM model.
    
    Args:
        sequence_length (int): Length of the input sequences.
        n_features (int): Number of features in the input.
        units (int): Number of units in each LSTM layer.
        cell (tf.keras.layers.Layer): LSTM cell type.
        n_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        loss (str): Loss function.
        optimizer (str): Optimizer.
        bidirectional (bool): Whether to use Bidirectional LSTM.
        
    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential()  # Initialize the Sequential model
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))  # Add dropout layer to reduce overfitting
    model.add(Dense(1, activation="linear"))  # Add output layer with linear activation
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)  # Compile the model
    return model


#function created to plot candlestick chart while also specifying the type of the chart
# with the title over the x and y-axis
def plot_candlestick_chart(df, n_days=1):   
    # plotting a candlestick chart with specified number of trading days wehere defaukt numbmer is set to 1
    df_resampled = df.resample(f'{n_days}D').agg({
        'open':"first",         #first price of the period
        "high":"max",           #highest price
        "low":"min",             #lowest price
        "adjclose": "last",     #last price
        "volume": "sum"         #total volume 

    })

     # Rename columns to match mplfinance requirements
    df_resampled.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'adjclose': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    #this will drop the rows with missing values after resampling
    df_resampled.dropna(inplace=True)

    #plot the resampled data
    mpf.plot(df_resampled, type="candle", style="charles", title="Candlestick cahrt", ylabel="Price") 

def plot_boxplot_chart(df, n_days=1):
    # Ensure the index is a DateTime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Resample data by n_days and calculate min, 25%, 50%, 75%, and max values
    df_resampled = df.resample(f"{n_days}D").agg({
        'adjclose': ['min', '25%', '50%', '75%', 'max']  # Calculate summary statistics
    })
    
    # Drop any rows with NaN values to clean the data
    df_resampled.dropna(inplace=True)

    print("Resampled Data for Boxplot:")
    print(df_resampled.head())

    # Prepare data for the boxplot
    boxplot_data = [
        df_resampled[('adjclose', 'min')].values,
        df_resampled[('adjclose', '25%')].values,
        df_resampled[('adjclose', '50%')].values,
        df_resampled[('adjclose', '75%')].values,
        df_resampled[('adjclose', 'max')].values
    ] 

    print("Boxplot Data:")
    print(boxplot_data)
    
    # Plot the boxplot for the resampled data
    plt.figure(figsize=(12, 6))  # Set the figure size
    plt.boxplot(boxplot_data, labels=[f'{n_days} Days'])  # Create the boxplot
    plt.title(f'Boxplot Chart ({n_days} Days per Boxplot)')  # Set plot title
    plt.ylabel('Adjusted Close Price')  # Set y-axis label
    plt.xlabel('Time')  # Set x-axis label
    plt.savefig('boxplot_chart.png')
    plt.close() 

def plot_graph(test_df):
    """
    Plot the true vs. predicted prices.
    
    Args:
        test_df (pd.DataFrame): DataFrame containing true and predicted prices.
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b', label='True Price')  # Plot true prices
    plt.plot(test_df[f'pred_adjclose_{LOOKUP_STEP}'], c='r', label='Predicted Price')  # Plot predicted prices
    plt.title(f'{ticker} Price Prediction')  # Set plot title
    plt.xlabel('Time')  # Set x-axis label
    plt.ylabel('Adjusted Close Price')  # Set y-axis label
    plt.legend()  # Show legend
    plt.show()  # Display the plot

def get_final_df(model, data):
    """
    Get the final DataFrame with predictions and profits.
    
    Args:
        model (tf.keras.Model): Trained model.
        data (dict): Data dictionary containing test data.
        
    Returns:
        final_df (pd.DataFrame): DataFrame with true prices, predicted prices, and profit calculations.
    """
    def buy_profit(current, pred_future, true_future):
        """
        Calculate profit for buying at the current price.
        
        Args:
            current (float): Current price.
            pred_future (float): Predicted future price.
            true_future (float): True future price.
            
        Returns:
            float: Profit from buying.
        """
        return pred_future - current if pred_future > current else 0

    def sell_profit(current, pred_future, true_future):
        """
        Calculate profit for selling at the current price.
        
        Args:
            current (float): Current price.
            pred_future (float): Predicted future price.
            true_future (float): True future price.
            
        Returns:
            float: Profit from selling.
        """
        return current - true_future if pred_future < current else 0

    # Prepare results
    result = {"true_adjclose": data['df']["adjclose"].tail(data["X_test"].shape[0])}
    result["true_adjclose"] = np.concatenate([result["true_adjclose"].values, [np.nan] * LOOKUP_STEP])
    result["pred_adjclose"] = np.concatenate([data["column_scaler"]["adjclose"].inverse_transform(model.predict(data["X_test"])).flatten(), [np.nan] * LOOKUP_STEP])
    
    # Inverse transform if scaling is used
    if SCALE:
        result["true_adjclose"] = np.concatenate([data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(data["y_test"], axis=1)).flatten(), [np.nan] * LOOKUP_STEP])
        result["pred_adjclose"] = np.concatenate([data["column_scaler"]["adjclose"].inverse_transform(model.predict(data["X_test"])).flatten(), [np.nan] * LOOKUP_STEP])

    final_df = pd.DataFrame(result)  # Create DataFrame with results
    final_df['profit_buy'] = final_df.apply(lambda row: buy_profit(row['true_adjclose'], row['pred_adjclose'], row['true_adjclose']) if not pd.isna(row['true_adjclose']) else np.nan, axis=1)
    final_df['profit_sell'] = final_df.apply(lambda row: sell_profit(row['true_adjclose'], row['pred_adjclose'], row['true_adjclose']) if not pd.isna(row['true_adjclose']) else np.nan, axis=1)
    
    return final_df

def new_version():
    """
    Test the new model version by training and evaluating it.
    """
    n_days = 1

    # Load and prepare the data
    data = load_data(ticker, n_steps=N_STEPS, scale=SCALE, shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, split_by_date=SPLIT_BY_DATE, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

    #plotting candlestick chart for the loaded chart
    plot_candlestick_chart(data['df'], n_days= n_days)

    #plotting boxplot chart for the loaded chart
    plot_boxplot_chart(data['df'], n_days = n_days)
    
    # Create and compile the model
    model = create_model(sequence_length=N_STEPS, n_features=len(FEATURE_COLUMNS), units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    
    # Train the model
    model.fit(data["X_train"], data["y_train"], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    
    # Evaluate the model and plot results
    final_df = get_final_df(model, data)
    plot_graph(final_df)

if __name__ == "__main__":
    new_version()
