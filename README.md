# earthquake-prediction-ml

# 🌍 Earthquake Prediction using Machine Learning

This project is a **modernized implementation** of an earthquake prediction model inspired by the work from AmanXai (2020).
It uses **deep learning regression** to predict **earthquake magnitude and depth** based on historical seismic data.

> ⚠️ This model is for **educational and research purposes only**.
> Earthquake prediction is a complex scientific problem and cannot be guaranteed by ML alone.

---

## 📌 Features

* Predicts **Magnitude** and **Depth**
* Uses **Timestamp, Latitude, Longitude** as inputs
* Neural network built with **TensorFlow/Keras**
* Hyperparameter tuning with **GridSearch + SciKeras**
* Data scaling with **StandardScaler**
* Model evaluation using **MSE & MAE**
* Supports saving and loading trained models

---

## 🗂️ Project Structure

```
earthquake-prediction/
│
├── database.csv
├── earthquake_model.keras
├── scaler.pkl
├── train_model.ipynb
├── README.md
```

---

## 🛠️ Requirements

Install all dependencies:

```bash
pip install pandas numpy scikit-learn scikeras tensorflow matplotlib seaborn joblib
```

---

## 📥 Dataset

The dataset should contain the following columns:

```
Date, Time, Latitude, Longitude, Depth, Magnitude
```

Example:

```
2020-01-01, 12:30:22, 34.56, 76.21, 12.3, 4.8
```

---

## 🚀 How to Run

### 1️⃣ Load Data

```python
final_data = pd.read_csv("database.csv")
```

### 2️⃣ Preprocess

```python
final_data["Datetime"] = pd.to_datetime(final_data["Date"] + " " + final_data["Time"])
final_data["Timestamp"] = final_data["Datetime"].astype(int) // 10**9
```

### 3️⃣ Train Model

Run `train_model.ipynb` or:

```python
python train.py
```

### 4️⃣ Evaluate

The model prints:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

---

## 🧠 Model Architecture

* Dense Neural Network
* 2 Hidden Layers
* ReLU / Tanh activations
* MSE loss (regression)

---

## 💾 Saving the Model

```python
keras_model.save("earthquake_model.keras")
joblib.dump(scaler, "scaler.pkl")
```

---

## 🔮 Predict New Data

```python
from tensorflow.keras.models import load_model
import joblib

model = load_model("earthquake_model.keras")
scaler = joblib.load("scaler.pkl")

X_new = scaler.transform([[timestamp, latitude, longitude]])
pred = model.predict(X_new)

print("Magnitude:", pred[0][0])
print("Depth:", pred[0][1])
```

---

## 📊 Visualization

The project includes:

* Scatter plots for actual vs predicted values
* Map visualizations (optional with Cartopy)

---

## 🧪 Evaluation Metrics

* **MSE** – Mean Squared Error
* **MAE** – Mean Absolute Error

---

## ⚠️ Disclaimer

This project **does not truly predict earthquakes**.
It learns patterns from historical data and should **not be used for real-world disaster prediction**.

---

## 🙌 Acknowledgements

Inspired by:

* AmanXai (2020) – *Earthquake Prediction Model with Machine Learning*

---

## 📬 Contact

Created by: **Shishir Ballal**
For learning & portfolio purposes.

---
