import os
import time
from tensorflow.keras.layers import LSTM

# Configuration Parameters
# Window size or the sequence length
N_STEPS = 50  # Number of historical time steps to use for prediction

# Lookup step: Defines how many steps (days) into the future we want to predict
LOOKUP_STEP = 15  # For example, predicting 15 days into the future

# Configuration for data preprocessing and training
# Whether to scale feature columns and output price
SCALE = True  # If True, scale feature columns
scale_str = f"sc-{int(SCALE)}"  # String representation for model file naming

# Whether to shuffle the dataset
SHUFFLE = True  # If True, shuffle the dataset before splitting
shuffle_str = f"sh-{int(SHUFFLE)}"  # String representation for model file naming

# Whether to split the dataset by date
SPLIT_BY_DATE = False  # If True, split the data by date rather than randomly
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"  # String representation for model file naming

# Test set size ratio
TEST_SIZE = 0.2  # Ratio of the dataset to be used for testing (20%)

# Features to be used for model input
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]  # List of features

# Date for saving data and model files
date_now = "2021-05-31"  # Date string for naming files

### Model Parameters
# Number of LSTM layers in the model
N_LAYERS = 2  # Number of LSTM layers to be used in the model

# Type of LSTM cell
CELL = LSTM  # LSTM cell to be used in the model

# Number of units (neurons) in each LSTM layer
UNITS = 256  # Number of neurons in each LSTM layer

# Dropout rate to prevent overfitting
DROPOUT = 0.4  # Dropout rate (40%)

# Whether to use Bidirectional LSTMs
BIDIRECTIONAL = False  # If True, use Bidirectional LSTM layers

### Training Parameters
# Loss function to be used for training
# Options are Mean Absolute Error or Huber Loss
# LOSS = "mae"  # Uncomment if using Mean Absolute Error
LOSS = "huber_loss"  # Uncomment if using Huber Loss

# Optimizer to be used for training
OPTIMIZER = "adam"  # Optimizer (Adam)

# Training parameters
BATCH_SIZE = 64  # Batch size for training
EPOCHS = 500  # Number of epochs for training

# Stock ticker symbol for data fetching and model naming
ticker = "AMZN"  # Amazon stock symbol

# Filename for saving stock data
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")  # Path for data file

# Model filename to save, uniquely named based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"  # Append '-b' if using Bidirectional LSTM
