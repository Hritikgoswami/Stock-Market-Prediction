# Stock-Market-Prediction 📈📊

![21894download](https://github.com/Hritikgoswami/Stock-Market-Prediction/assets/84679973/405cc395-6d26-46f9-922a-1f9481e6903c)

🌐 A hybrid model for stock price/performance prediction using numerical analysis of historical stock prices, and sentimental analysis of news headlines.
## Table of Contents 📑

1. [Introduction](#introduction) 🎯
2. [Features](#features) ✨
3. [Code Explanation](#code-explanation) 📈
4. [Installation](#installation) 🛠️
5. [Dataset](#dataset) 📊
6. [Model Training](#model-training) 🧠
7. [Results](#results) 📈
8. [Contributing](#contributing) 🤝

## Introduction 📌

This repository demonstrates how to predict stock market prices by combining sentiment analysis of news headlines with historical stock market data. The code aims to show how sentiment from news headlines can be used as a feature for stock market prediction.

## Features ✨

- Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner) 💬📜
- LSTM (Long Short-Term Memory) neural network for prediction 🧠📊
- Data preprocessing and scaling 🔄📈
- Visualization of predicted vs. actual stock prices 📉📈

## Code Explanation 📈

1. **Data Collection and Preprocessing:**
    - Import necessary libraries.
    - Load historical stock market data (BSE Sensex).
    - Load news headlines data.
    - Perform data cleaning, preprocessing, and transformation on both datasets.

2. **Sentiment Analysis:**
    - Use NLTK's VADER sentiment analysis tool to calculate sentiment scores (compound, negative, neutral, positive) for each news headline.
    - Combine news headlines by date and join with stock market data.

3. **Data Scaling:**
    - Normalize and scale the features and target variables using MinMaxScaler.

4. **Data Splitting:**
    - Split the data into training and testing sets for model training and evaluation.

5. **Model Architecture:**
    - Build an LSTM (Long Short-Term Memory) neural network using Keras.
    - Define the architecture with multiple LSTM layers and dropout layers.

6. **Model Training:**
    - Compile the model with Mean Squared Error (MSE) loss function and Adam optimizer.
    - Train the model on the training dataset, monitoring validation loss.

7. **Model Evaluation:**
    - Evaluate the model's performance using various metrics like MSE, RMSE, and more.
    - Calculate root mean squared error for predictions.

8. **Visualization:**
    - Visualize the actual stock prices and predicted stock prices using Matplotlib.

9. **Save Model:**
    - Save the trained model as JSON and weights as H5 files for future use.

## Installation 🛠️

1. Clone the repository:
   ```
   git clone https://github.com/Hritikgoswami/Stock-Market-Prediction.git
   cd stock-market-prediction
   ```

## Dataset 📊

Prepare your historical stock market data and news headlines CSV files. Adjust the file paths in the code accordingly.

Dataset Link - [Times of India News Headlines](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQMQH)

## Model Training 🧠
![image](https://github.com/Hritikgoswami/Stock-Market-Prediction/assets/84679973/ae82e95d-870b-418d-9b0c-3cbebe310cef)

1. Load and preprocess data.
2. Perform sentiment analysis on news headlines.
3. Scale the data.
4. Split into training and testing sets.
5. Build and train the LSTM model.
6. Evaluate model performance.

## Results 📈
![Stock Price Prediction](https://github.com/Hritikgoswami/Stock-Market-Prediction/assets/84679973/b41b497d-c0ab-4a67-8cd8-7a327bb502c4)

- Evaluate the model using metrics like MSE, RMSE.
- Visualize actual vs. predicted stock prices.

## Contributing 🤝

Contributions are welcome! If you'd like to enhance the project or fix any issues, feel free to fork the repository and submit a pull request. Let's make this project better together! 🚀
