import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from stock_prediction import create_model, load_data
from parameters import *  # Import parameters and configurations from the 'parameters.py' file

# Create necessary directories if they do not exist
if not os.path.isdir("results"):
    os.mkdir("results")  # Create 'results' directory for saving model weights
if not os.path.isdir("logs"):
    os.mkdir("logs")  # Create 'logs' directory for TensorBoard logs
if not os.path.isdir("data"):
    os.mkdir("data")  # Create 'data' directory for storing data files

# Load data using the parameters from 'parameters.py'
data = load_data(
    ticker,                # Stock ticker symbol (e.g., "AMZN")
    N_STEPS,               # Number of time steps for the model input
    scale=SCALE,           # Whether to scale feature columns and output price
    split_by_date=SPLIT_BY_DATE,  # Whether to split the dataset by date
    shuffle=SHUFFLE,      # Whether to shuffle the dataset
    lookup_step=LOOKUP_STEP,      # Number of steps into the future to predict
    test_size=TEST_SIZE,  # Proportion of the dataset to be used for testing
    feature_columns=FEATURE_COLUMNS  # List of feature columns to be used
)

# Construct the model using specified parameters
model = create_model(
    N_STEPS,               # Number of time steps for the model input
    len(FEATURE_COLUMNS),  # Number of features in the dataset
    loss=LOSS,             # Loss function to be used (e.g., "mae" or "huber_loss")
    units=UNITS,           # Number of units (neurons) in each LSTM layer
    cell=CELL,             # Type of LSTM cell (e.g., LSTM)
    n_layers=N_LAYERS,     # Number of LSTM layers
    dropout=DROPOUT,       # Dropout rate to prevent overfitting
    optimizer=OPTIMIZER,   # Optimizer to be used (e.g., "adam")
    bidirectional=BIDIRECTIONAL  # Whether to use Bidirectional LSTM layers
)

# Define the path for saving model weights
model_path = os.path.join("results", f"{model_name}.h5")

# Load model weights if the file exists
if os.path.exists(model_path):
    model.load_weights(model_path)  # Load previously saved weights to resume training or continue from the last saved state

# Set up callbacks for model training
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# ModelCheckpoint callback to save model weights during training
checkpointer = ModelCheckpoint(
    model_path,           # Path where model weights will be saved
    save_weights_only=True,  # Save only the model weights
    save_best_only=True,    # Save only the best model weights based on validation loss
    verbose=1             # Print progress messages
)

# TensorBoard callback for visualizing training process
tensorboard = TensorBoard(
    log_dir=os.path.join("logs", model_name)  # Directory where TensorBoard logs will be saved
)

# Train the model with specified parameters
history = model.fit(
    data["X_train"],          # Training features
    data["y_train"],          # Training labels
    batch_size=BATCH_SIZE,    # Batch size for training
    epochs=EPOCHS,            # Number of epochs to train the model
    validation_data=(data["X_test"], data["y_test"]),  # Validation data for evaluation during training
    callbacks=[checkpointer, tensorboard],  # List of callbacks to use during training
    verbose=1                 # Print progress messages
)
