# Stock Market Prediction

This repository contains a project focused on predicting stock prices for Google and Apple using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

## Project Overview

This project aims to predict the weekly and monthly stock prices of Google and Apple. The dataset includes columns for Open, High, Low, Close, Adjusted Close, and Volume. The data is preprocessed and used to train an LSTM model to predict future stock prices.

## Repository Structure

- `Stock_Market_Prediction.ipynb`: The main Jupyter notebook containing the data analysis, model training, and predictions.
- `data/`: Directory containing the stock price data files.

## Setup Instructions

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Necessary Python libraries: `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Data Analysis

The notebook includes functions to analyze the stock data, including:

- **`analyzeStockData`**: Prints the company name, data frame information, and the first few rows of the data.
- **`plot_high_low_close_data`**: Plots high, low, close prices, and volume data.

## Model Training

The notebook demonstrates how to preprocess the data and train an LSTM model. Key steps include:

1. Data normalization using `MinMaxScaler`.
2. Splitting data into training and testing sets.
3. Building and compiling the LSTM model using `tensorflow.keras`.

## Prediction and Visualization

The notebook provides functions to visualize the predictions:

- **`visualize_predictions`**: Plots actual vs. predicted stock prices.
- **`plot_moving_averages`**: Plots stock prices with 1-week, 1-month, and 3-month moving averages.

## Results

The project demonstrates the LSTM model's capability to predict stock prices with reasonable accuracy. The visualizations show the comparison between actual and predicted prices, highlighting the model's performance.

## Conclusion

This project provides a comprehensive approach to stock price prediction using LSTM models. The steps outlined in the notebook can be adapted for other stock datasets or further refined for improved accuracy.
