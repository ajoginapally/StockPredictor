import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.regularizers import l2

def analyzeStockData(companyName: str, df: pd.DataFrame):
    # Print the company name in uppercase with stars
    print("************************************* ")
    print(companyName.upper())
    print("************************************* ")

    # Print the DataFrame information
    print("\n===>>> INFO")
    print(df.columns)

    # Print the descriptive statistics of the DataFrame
    # print("\n===>>> DESCRIBE")
    # print(df.describe().to_string())

    # Print the first few rows of the DataFrame
    print("\n===>>> HEAD")
    print(df.head())

def modelSummary(model):
    print("\n\n===>>> MODEL SUMMARY\n")
    print(model.summary())
    print("\n\n")
    
    
    
def plot_high_low_close_data(df: pd.DataFrame):
    """
    Plot the high, low, and close prices along with volume data.

    Parameters:
        df (pd.DataFrame): DataFrame containing high, low, close, and volume data.
    """
    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    # Plot the high, low, and close prices on the first subplot
    ax1.plot(df['High'], label='High')
    ax1.plot(df['Low'], label='Low')
    ax1.plot(df['Close'], label='Close')

    # Set the title and labels for the first subplot
    ax1.set_title('High, Low, and Close Prices')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')

    # Plot the volume on the second subplot
    ax2.bar(df.index, df['Volume'])

    # Set the title and labels for the second subplot
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')

    # Rotate the x-axis labels on the second subplot
    plt.xticks(rotation=35)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_stock_averages(df: pd.DataFrame):
    """
    Plot stock close price along with moving averages.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.
    """
    # Compute the moving averages
    df['1W_MA'] = df['Close'].rolling(window=7).mean()  # 1-week moving average
    df['1M_MA'] = df['Close'].rolling(window=30).mean() # 1-month moving average (assuming 30 days as a month)
    df['3M_MA'] = df['Close'].rolling(window=90).mean() # 3-month moving average (assuming 60 days as a month)


    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot the close prices
    plt.plot(df['Close'], label='Close Price', linewidth=1)

    # Plot the moving averages
    plt.plot(df['1W_MA'], label='Weekly Moving Average', linewidth=1.5)
    plt.plot(df['1M_MA'], label='Monthly Moving Average', linewidth=1.5)
    plt.plot(df['3M_MA'], label='Quarterly Moving Average', linewidth=1.5)

    # Set the title and labels
    plt.title('Stock Close Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=35)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_stock_data(df: pd.DataFrame):
    """
    Plot high, low, close prices along with volume and stock close price with moving averages.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.
    """
    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot high, low, close data along with volume
    axes[0].plot(df['High'], label='High')
    axes[0].plot(df['Low'], label='Low')
    axes[0].plot(df['Close'], label='Close')
    axes[0].set_title('High, Low, and Close Prices')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')

    axes[1].bar(df.index, df['Volume'])
    axes[1].set_title('Volume')
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Date')
    axes[1].tick_params(axis='x', rotation=35)
    # Compute the moving averages
    df['1W_MA'] = df['Close'].rolling(window=7).mean()  # 1-week moving average
    df['1M_MA'] = df['Close'].rolling(window=30).mean() # 1-month moving average (assuming 30 days as a month)
    df['3M_MA'] = df['Close'].rolling(window=90).mean() # 3-month moving average (assuming 60 days as a month)

    # Plot stock close price with moving averages
    axes[2].plot(df['Close'], label='Close Price', linewidth=1)
    axes[2].plot(df['1W_MA'], label='Weekly Moving Average', linewidth=1.5)
    axes[2].plot(df['1M_MA'], label='Monthly Moving Average', linewidth=1.5)
    axes[2].plot(df['3M_MA'], label='Quarterly Moving Average', linewidth=1.5)
    axes[2].set_ylabel('Price')
    axes[2].legend(loc='upper left')

    # Set the title and labels for the combined plot
    axes[2].set_title('Stock Close Price with Moving Averages')
    axes[2].set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_predictions(df: pd.DataFrame, predictions: pd.Series, train_size: int):
    """
    Visualize stock price predictions.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.
        predictions (pd.Series): Predicted stock prices.
        train_size (int): Size of the training data.
    """
    # Split the DataFrame into training and testing sets
    train = df[:train_size]
    test = df[train_size:].copy()  # Avoid SettingWithCopyWarning

    # Add predictions to the test set
    test['Predictions'] = predictions

    # Create the plot
    plt.figure(figsize=(12, 5))
    plt.plot(train['Close'], label='Training Data')
    plt.plot(test['Close'], label='Actual Stock Price')
    plt.plot(test['Predictions'], label='Predicted Stock Price')

    # Set the title and labels
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def create_sequences(data, seq_length):
    """
    Create sequences of data for LSTM.

    Parameters:
        data (np.ndarray): The input data.
        seq_length (int): Length of each sequence.

    Returns:
        np.ndarray, np.ndarray: Input sequences and corresponding target values.
    """
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

def prepare_lstm_data(scaled_data, seq_length, train_split=0.8):
    """
    Prepare data for LSTM training.

    Parameters:
        scaled_data (np.ndarray): Scaled input data.
        seq_length (int): Length of each sequence.
        train_split (float): Percentage of data to use for training.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, int: Training and testing data along with the size of the training data.
    """
    # Split into training and testing datasets
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - seq_length:]

    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)

    # Reshape for LSTM layers (samples, time steps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, train_size


def build_lstm_model(input_shape):
    """
    Build LSTM model with increased units and additional dense layers.

    Parameters:
        input_shape (tuple): Shape of input data.

    Returns:
        keras.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

def build_lstm_model_2(input_shape):
    """
    Build LSTM model with increased units, adjusted dropout rate, and different activation functions.

    Parameters:
        input_shape (tuple): Shape of input data.

    Returns:
        keras.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))  # Increased units
    model.add(Dropout(0.3))  # Adjusted dropout rate
    model.add(LSTM(units=32, return_sequences=False))  # Increased units
    model.add(Dropout(0.3))  # Adjusted dropout rate
    model.add(Dense(16, activation='relu'))  # Experiment with different activation function
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model_3(input_shape):
    """
    Build LSTM model with increased units, adjusted dropout rate, and different activation functions.

    Parameters:
        input_shape (tuple): Shape of input data.

    Returns:
        keras.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))  # Increased units
    model.add(Dropout(0.3))  # Adjusted dropout rate
    model.add(Dense(32, activation='relu'))  # Experiment with different activation function
    model.add(Dropout(0.3))  # Adjusted dropout rate
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Mean Absolute Percentage Error (MAPE) in percentage.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_regression_accuracy(y_true, y_pred, threshold):
    """
    Compute regression accuracy based on a threshold.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.
        threshold (float): Threshold value for considering predictions as correct.

    Returns:
        float: Accuracy percentage.
    """
    correct_predictions = np.abs(y_true - y_pred) <= threshold
    accuracy = np.mean(correct_predictions) * 100
    return accuracy

def print_threshold_table(y_true, y_pred):
    """
    Print a table of accuracy for different threshold percentages.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.
    """
    thresholds = [0.01, 0.02, 0.03, 0.05]  # Threshold percentages
    print("\nThreshold (%) | Accuracy")
    print("--------------------------")
    for threshold in thresholds:
        accuracy = compute_regression_accuracy(y_true, y_pred, threshold)
        print(f"{threshold * 100: <14.0f}| {accuracy:.2f}%")




def run_lstm_stock_prediction(df, seq_length, scaled_data, scaler, epochs=5, batch_size=32, validation_split=0.2):
    """
    Run LSTM stock price prediction for multiple models, print evaluation metrics, and visualize predictions.

    Parameters:
        df (pd.DataFrame): DataFrame containing the stock data.
        seq_length (int): Sequence length for creating input sequences.
        scaled_data (np.ndarray): Scaled stock data.
        scaler: Scaler object used for scaling the data.
        epochs (int): Number of epochs for training the models.
        batch_size (int): Batch size for training the models.
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        None
    """
    # Generate Data
    x_train, y_train, x_test, y_test, train_size = prepare_lstm_data(scaled_data, seq_length)

    # Define models
    models = [build_lstm_model, build_lstm_model_2, build_lstm_model_3]

    for idx, build_model_func in enumerate(models, start=1):
        print("\n\n")
        print("*******************************************")
        print(f"Training LSTM Model {idx}")
        print("*******************************************\n")

        # Build and train the model
        model = build_model_func(x_train.shape[1:])
        model.summary()
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        # Predict
        predictions = model.predict(x_test)

        # Calculate evaluation metrics
        mape_model = calculate_mape(y_test, predictions)
        print("Mean Absolute Percentage Error :", mape_model)
        print_threshold_table(y_test, predictions)

        # Transform predictions back to the original scale
        predictions = scaler.inverse_transform(predictions)

        # Visualize predictions
        visualize_predictions(df, predictions, train_size)



def predict_stock_data(company_name, filename, seq_length, epochs, batch_size, validation_split):
    """
    Predict stock data using LSTM models.

    Parameters:
        company_name (str): Name of the company.
        filename (str): Path to the CSV file containing stock data.
        seq_length (int): Sequence length for creating input sequences.
        epochs (int): Number of epochs for training the models.
        batch_size (int): Batch size for training the models.
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        None
    """
    # Read data from file
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter data from 2010-01-01
    df = df.loc[df['Date'] > '2010-01-01']

    # Rename 'Close/Last' column to 'Close' if present
    if 'Close' not in df.columns and 'Close/Last' in df.columns:
        df = df.rename(columns={'Close/Last': 'Close'})

    # Set 'Date' column as index
    df.set_index('Date', inplace=True)

    # Analyze and plot stock data
    analyzeStockData(company_name, df)
    plot_stock_data(df)

    # Extract 'Close' prices
    data = df['Close'].values

    # Normalize the dataset using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Model prediction
    run_lstm_stock_prediction(df, seq_length, scaled_data, scaler, epochs, batch_size, validation_split)


predict_stock_data(company_name="GOOGLE",
                   filename="GOOGL.csv",
                   seq_length=60,
                   epochs=8,
                   batch_size=32,
                   validation_split=0.2)
