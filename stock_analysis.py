import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define stock tickers
stocks = ["AAPL", "GOOGL", "TSLA"]

# Define time period
start_date = "2023-01-01"
end_date = "2024-02-10"

# Step 1: Fetch and Save Data
for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date)
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df.to_csv(f"{stock}_data.csv", index=False)
    print(f"✅ Downloaded {stock} data.")

print("\n✅ Stock data successfully saved.\n")

# Step 2: Read CSV Files and Process Data
data = {}
for stock in stocks:
    df = pd.read_csv(f"{stock}_data.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)  # Drop NaT values
    df = df.apply(pd.to_numeric, errors="coerce")
    data[stock] = df

# Step 3: Train Machine Learning Model (Linear Regression)
stock_to_predict = "AAPL"  # Choose stock for prediction
df = data[stock_to_predict][["Close"]].dropna()

# Convert dates into numerical values for ML training
df["Days"] = np.arange(len(df))

# Split Data into Train and Test Sets
X = df[["Days"]]
y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Future Prices
future_days = np.arange(len(df), len(df) + 10).reshape(-1, 1)
future_prices = model.predict(future_days)

# Step 4: Plot Predictions
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="red", label="Testing Data")
plt.plot(np.arange(len(df)), model.predict(X), color="green", label="Model Prediction")
plt.scatter(future_days, future_prices, color="purple", label="Future Predictions", marker="x")
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.title(f"{stock_to_predict} Stock Price Prediction")
plt.legend()
plt.grid()
plt.show()

# Print Predictions
print("\n📈 Predicted Prices for Next 10 Days:")
for i, price in enumerate(future_prices):
    print(f"Day {i + 1}: ${price:.2f}")

print("\n✅ Machine Learning Analysis Complete.")

