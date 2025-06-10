# LSTM-StockPrice-DL

# 🧠 End-to-End Stock Price Prediction with Deep Learning

## 📈 Project Overview

This project is an **end-to-end deep learning pipeline** that predicts stock prices using an LSTM (Long Short-Term Memory) neural network. It specifically uses historical data from **ADANIPORTS** (Adani Ports & SEZ Ltd.) to forecast future closing prices. The project includes data loading, preprocessing, visualization, model training, evaluation, and future prediction.

---

## ⚡ Tech Stack

* **Python 3.x**
* **TensorFlow / Keras** (Deep Learning)
* **scikit-learn** (Preprocessing, Metrics)
* **Pandas & NumPy** (Data Handling)
* **Matplotlib & Seaborn** (Visualization)

---

## 📊 Project Pipeline

### 📊 STEP 1: Data Loading & Exploration

* Reads stock data from a CSV file (`ADANIPORTS.csv`)
* Displays dataset shape, date range, missing values
* Converts 'Date' to datetime format

### 📊 STEP 2: Data Visualization

* Plots:

  * Closing price over time
  * Trading volume over time
  * Close price distribution
  * High-Low daily spread

### ⚙️ STEP 3: Data Preprocessing

* Adds time-based features: Year, Month, Day, Weekday, etc.
* Selects main financial features: `Open`, `High`, `Low`, `Close`, `Volume`
* Scales features to \[0, 1] range using `MinMaxScaler`
* Creates input sequences of 30 days to predict the next day's price

### 🤖 STEP 4: Model Architecture

* **LSTM Neural Network** with:

  * 3 LSTM layers (100, 64, and 32 units)
  * Dropout layers to prevent overfitting
  * Batch Normalization for stable training
  * Dense output layer for regression

### 🎯 STEP 5: Model Training

* Compiles model using:

  * `Adam` optimizer
  * `Mean Squared Error (MSE)` loss
  * `Mean Absolute Error (MAE)` metric
* Uses callbacks:

  * `EarlyStopping` for efficient training
  * `ReduceLROnPlateau` to adjust learning rate dynamically

### 📈 STEP 6: Training Visualization

* Plots loss and MAE curves over epochs

### 🔮 STEP 7: Predictions

* Predicts closing prices on the test set
* Inverse transforms predictions from scaled to actual prices

### 📊 STEP 8: Evaluation

* Evaluation Metrics:

  * MSE, RMSE, MAE, MAPE

### 📈 STEP 9: Results Visualization

* Visualizes:

  * Actual vs Predicted Prices (Line Plot)
  * Scatter plot of Actual vs Predicted
  * Error distribution
  * Last 50 samples (zoomed-in comparison)

### 🔮 STEP 10: Future Prediction

* Uses the latest sequence to predict the **next day's closing price**
* Compares it with the actual last known closing price

### 📅 STEP 11: Summary

* Shows final configuration and model performance
* Highlights disclaimer: **for educational purposes only**

---

## 📀 Output Files

* `adaniports_lstm_model.h5`: Trained LSTM model
* `adaniports_scaler.pkl`: Fitted MinMaxScaler object

---

## ⚠️ Disclaimer

This project is a **research and educational** demonstration. It is not intended for real-world financial decisions. Always consult financial experts before investing.

---

## 🚀 How to Run

1. Clone this repo
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Place `ADANIPORTS.csv` in the correct folder (or update path in script)
4. Run the notebook/script
5. View the predictions and visualizations

---

## 👩‍💼 Author

**Hemprasad A C**
B.Tech CSE @ VIT Chennai
-----

## 📊 Sample Result

* **RMSE**: \~₹X.XX
* **MAE**: \~₹X.XX
* **MAPE**: \~X.XX%
* **Next Day Prediction**: ₹XXXX.XX (vs Actual ₹XXXX.XX)

