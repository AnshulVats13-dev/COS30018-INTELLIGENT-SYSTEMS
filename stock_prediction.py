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
import mplfinance as mpf
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_absolute_error


# Configuration parameters
N_STEPS = 50  # Number of time steps in each sequence
LOOKUP_STEP = 15  # How many steps ahead to predict
SCALE = True  # Whether to scale the feature values
SHUFFLE = True  # Whether to shuffle the training data
SPLIT_BY_DATE = False  # Whether to split the data by date or randomly
TEST_SIZE = 0.2  # Proportion of data to use for testing
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]  # Features used in the model
date_now = "2021-05-31"  # Current date used for file naming

# Model configurations to test
model_configurations = [
    {"units": 256, "n_layers": 2, "dropout": 0.3, "bidirectional": False},
    {"units": 128, "n_layers": 2, "dropout": 0.3, "bidirectional": False},
    {"units": 256, "n_layers": 3, "dropout": 0.5, "bidirectional": True},
    {"units": 128, "n_layers": 3, "dropout": 0.5, "bidirectional": True}
]

# Training configuration
LOSS = "huber_loss"  # Loss function used for training
OPTIMIZER = "adam"  # Optimizer used for training
BATCH_SIZE = 64  # Batch size for training
EPOCHS = 500  # Number of epochs for training

ARIMA_ORDER = (1,1,1) #P,D,Q ORDER FOR ARIMA

ENSEMBLE_WEIGHTS = {'ARIMA': 0.3, 'LSTM': 0.7}


ticker = "AMZN"  # Stock ticker symbol

# Get user inputs for start and end dates
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# File and model naming
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{date_now}_{ticker}-sh-{int(SHUFFLE)}-sc-{int(SCALE)}-sbd-{int(SPLIT_BY_DATE)}-{LOSS}-{OPTIMIZER}-LSTM-seq-{N_STEPS}-step-{LOOKUP_STEP}"


# Function to shuffle two arrays in unison
def shuffle_in_unison(a, b):
    state = np.random.get_state()  # Save the current random state
    np.random.shuffle(a)  # Shuffle the first array
    np.random.set_state(state)  # Restore the random state
    np.random.shuffle(b)  # Shuffle the second array

# Function to load and prepare data for training
def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    if isinstance(ticker, str):
        df = si.get_data(ticker)  # Fetch data from Yahoo Finance
    elif isinstance(ticker, pd.DataFrame):
        df = ticker  # Use provided DataFrame
    else:
        raise TypeError("ticker must be a string or a pandas DataFrame")

    result = {'df': df.copy()}  # Save a copy of the DataFrame

    # Ensure all feature columns are present in the DataFrame
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # Add date column if not present
    if "date" not in df.columns:
        df["date"] = df.index

    # Scale feature columns if required
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # Create the target variable 'future' shifted by lookup_step
    df['future'] = df['adjclose'].shift(-lookup_step)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))  # Last sequence for prediction
    df.dropna(inplace=True)  # Drop rows with NaN values

    # Prepare sequences for training
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

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

    # Prepare the test DataFrame for plotting
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

# Function to create the LSTM model
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
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
        model.add(Dropout(dropout))  # Add dropout to prevent overfitting
    model.add(Dense(1, activation="linear"))  # Output layer
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)  # Compile the model
    return model

# Multivariate Prediction Function
def multivariate_prediction(data, k_days, sequence_length=N_STEPS, n_features=len(FEATURE_COLUMNS)):
    X, y = [], []
    for i in range(len(data) - sequence_length - k_days):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + k_days, 0])  # Assuming adjclose is first feature
    return np.array(X), np.array(y)

#Funtion to predict multiple future prices (multistep prediction)

def get_multistep_predictions(model, last_sequence, steps, scaler):
    """
    Predicts multiple future prices based on the last known sequence.
    
    Args:
        model: Trained LSTM model.
        last_sequence: The last sequence of input features.
        steps: Number of future steps to predict.
        scaler: Scaler to inverse transform predictions.
        
    Returns:
        predictions: A list of predicted future prices.
    """
    predictions = []   #list to store predicted future prices
    current_sequence = last_sequence   #this means it starts with the last known sequence

    #loop through the number of steps to predict

    for _ in range(steps):
        #predict the nexr price
        prediction = model.predict(current_sequence[np.newaxis, ...])[0,0]
        predictions.append(prediction)

        #update the current sequence by appending the new prediction
        #it also removes the oldest entry and then append the new prediction
        new_data = np.array(current_sequence[0][-N_STEPS + 1:] + [prediction])
        current_sequence = np.vstack([current_sequence, new_data])[-N_STEPS:]   #it keeps the last N_STEPS

        #Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
        print(f"Multistep predictions: {predictions}")  
        return predictions   #returns the list of predictions

# Function to plot a candlestick chart
def plot_candlestick_chart(df, n_days=1):
    df_resampled = df.resample(f'{n_days}D').agg({
        'open': "first",
        "high": "max",
        "low": "min",
        "adjclose": "last",
        "volume": "sum"
    })

    df_resampled.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'adjclose': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    df_resampled.dropna(inplace=True)
    mpf.plot(df_resampled, type="candle", style="charles", title="Candlestick Chart", ylabel="Price")

# Function to plot a boxplot of stock prices
def plot_boxplot_chart(df, n_days=1):
    df = df.copy()
    df.columns = df.columns.str.strip()  # Strip any extra spaces from column names
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
    plt.boxplot([df['Open'], df['High'], df['Low'], df['Close']], labels=['Open', 'High', 'Low', 'Close'])
    plt.title('Boxplot of Stock Prices')
    plt.show()

# Function to plot the true vs predicted prices
def plot_graph(test_df):
    plt.plot(test_df['true_adjclose'], label='True Prices')
    plt.plot(test_df['pred_adjclose'], label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('True vs Predicted Prices')
    plt.legend()
    plt.show()

# Function to prepare the final DataFrame for plotting predictions
def get_final_df(test_df, scaler):
    # Predict the future prices using the test data
    predictions = model.predict(data["X_test"])

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Prepare the final DataFrame for plotting
    pred_df = pd.DataFrame({
        'true_adjclose': data['y_test'],
        'pred_adjclose': predictions.flatten()
    })
    return pred_df


#function for arima and ensemble
def fit_arima_model(data):
    model = ARIMA(data, order=ARIMA_ORDER)
    results = model.fit()
    return results

def ensemble_predict(arima_pred, lstm_pred, weights):
    return weights['ARIMA'] * arima_pred + weights['LSTM'] * lstm_pred

class TrainingPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TrainingPlotCallback, self).__init__()
        self.history = {'loss': [], 'val_loss': []}
    
    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        if epoch % 50 == 0:  # Plot every 50 epochs
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.show()


# Main function to run the process
def main():
    # Load data
    data = load_data(ticker, n_steps=N_STEPS, scale=SCALE, shuffle=SHUFFLE, 
                     lookup_step=LOOKUP_STEP, split_by_date=SPLIT_BY_DATE, 
                     test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
    
    # Fit ARIMA model
    arima_model = fit_arima_model(data['df']['adjclose'])
    arima_forecast = arima_model.forecast(steps=len(data['y_test']))

    best_mae = float('inf')
    best_model = None
    best_config = None

    for config in model_configurations:
        model = create_model(sequence_length=N_STEPS, n_features=len(FEATURE_COLUMNS), 
                             units=config["units"], cell=LSTM, n_layers=config["n_layers"], 
                             dropout=config["dropout"], loss=LOSS, optimizer=OPTIMIZER, 
                             bidirectional=config["bidirectional"])
        
        history = model.fit(data["X_train"], data["y_train"], 
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(data["X_test"], data["y_test"]), 
                            verbose=1, callbacks=[TrainingPlotCallback()])

        # Evaluate the model
        _, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        print(f"MAE: {mae:.4f} for config: {config}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_config = config

    print(f"Best model configuration: {best_config}")

    # Use the best model for predictions
    lstm_predictions = best_model.predict(data["X_test"])

    # Inverse transform predictions if scaled
    if SCALE:
        lstm_predictions = data["column_scaler"]["adjclose"].inverse_transform(lstm_predictions)
        y_test = data["column_scaler"]["adjclose"].inverse_transform(data["y_test"].reshape(-1, 1))
    else:
        y_test = data["y_test"]

    # Ensemble predictions
    ensemble_predictions = ensemble_predict(arima_forecast, lstm_predictions.flatten(), ENSEMBLE_WEIGHTS)

    # Prepare final DataFrame for plotting
    pred_df = pd.DataFrame({
        'true_adjclose': y_test.flatten(),
        'lstm_pred': lstm_predictions.flatten(),
        'arima_pred': arima_forecast,
        'ensemble_pred': ensemble_predictions
    })

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(pred_df['true_adjclose'], label='True Prices')
    plt.plot(pred_df['lstm_pred'], label='LSTM Predictions')
    plt.plot(pred_df['arima_pred'], label='ARIMA Predictions')
    plt.plot(pred_df['ensemble_pred'], label='Ensemble Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Predictions')
    plt.legend()
    plt.show()

    # Calculate and print performance metrics
    for model in ['lstm_pred', 'arima_pred', 'ensemble_pred']:
        mae = mean_absolute_error(pred_df['true_adjclose'], pred_df[model])
        mse = mean_squared_error(pred_df['true_adjclose'], pred_df[model])
        rmse = np.sqrt(mse)
        print(f"{model} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # Multistep predictions
    last_sequence = data['last_sequence'][-N_STEPS:]
    lstm_multistep = get_multistep_predictions(best_model, last_sequence, LOOKUP_STEP, data["column_scaler"]["adjclose"])
    arima_multistep = arima_model.forecast(steps=LOOKUP_STEP)
    ensemble_multistep = ensemble_predict(arima_multistep, lstm_multistep, ENSEMBLE_WEIGHTS)

    print(f"Multistep predictions for the next {LOOKUP_STEP} days:")
    print(f"LSTM: {lstm_multistep}")
    print(f"ARIMA: {arima_multistep}")
    print(f"Ensemble: {ensemble_multistep}")

    plot_candlestick_chart(data['df'])
    plot_boxplot_chart(data['df'])

if __name__ == "__main__":
    main()
