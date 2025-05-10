# CNN-LSTM Stock Price Prediction

![Banner](banner.jpg)
This Jupyter Notebook demonstrates how to build and train a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model using TensorFlow to predict future stock closing prices from historical data.

---

## 📋 Author

- **Name:** Youself Sayed  
- **Student ID:** 4211094

---

## 🗂️ Repository Contents

- **`cnn-lstm.ipynb`**  
  The main Jupyter Notebook that:
  1. Loads and preprocesses the stock dataset  
  2. Visualizes historical closing prices  
  3. Constructs sliding windows of past prices  
  4. Defines and trains a CNN-LSTM model  
  5. Evaluates performance on a held-out test set  

---

## 🛠 Requirements

- **Python 3.7+**  
- **Jupyter Notebook**

### Python packages

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

📊 Dataset

This notebook uses the Stock Market Dataset on Kaggle. You’ll need to download the dataset and place it under:

kaggle/input/stock-market-dataset/stocks/A.csv

The CSV must include at least the following columns:
- Date (YYYY-MM-DD)
- Close (closing price)

If you wish to experiment with other tickers, update the file path in the notebook accordingly.

---

🚀 Getting Started
	1.	Clone this repository

```bash
git clone https://github.com/mrjo10/lstm-and-cnn.git
cd lstm-and-cnn
```

	2.	Ensure dataset is in place
Place the stocks folder under kaggle/input/stock-market-dataset/ so that the notebook can load stocks/A.csv.
	3.	Launch Jupyter

```bash
jupyter notebook cnn-lstm.ipynb
```

	4.	Run all cells in order, or step through to understand each block.

---

🧠 Notebook Workflow
1.	Data Loading & Preprocessing
  - Read CSV, sort by date, extract Close prices array
  - Visualize closing prices over time
2.	Sliding‐Window Creation
  - Use a window size of 100 days
  - Build X (input sequences) and y (next-day price)
3.	Train/Test Split
  - 80% training, 20% testing (no shuffling to preserve time order)
4.	Model Architecture

```python
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
    MaxPooling1D(2),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
```

5.	Training
  - 20 epochs
  - Batch size: 32
  - Validation split: 10%
6.	Evaluation
  - Report test Loss, MSE, MAE

---

📈 Results

After training, the notebook prints out the final test metrics. You can further extend the notebook to:
	•	Plot predictions vs. actual prices
	•	Experiment with different window sizes, architectures or optimizers
	•	Tune hyperparameters via grid search or Bayesian optimization
