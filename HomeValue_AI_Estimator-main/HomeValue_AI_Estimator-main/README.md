# 🏡HomeValue AI Estimator

This project is a **House Price Prediction Web App** built using **Streamlit** and **Linear Regression**.  
It allows users to input house features and get an estimated price prediction.

---

## 🚀 Features
- Interactive **Streamlit web interface**
- **Dynamic feature inputs** for prediction
- Uses **Linear Regression** model for training
- Displays **model performance metrics** (MSE, R² score)
- Real-time **house price prediction**

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── House_data.csv # Dataset (place inside project folder)
├── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
Install dependencies

bash
Copy code
pip install -r requirements.txt
Add dataset

Place House_data.csv inside the project folder.

Ensure it has a price column as the target variable.

Run the app

bash
Copy code
streamlit run app.py
📊 Model Details
Algorithm: Linear Regression

Training: 80% of dataset

Testing: 20% of dataset

Metrics Displayed:

Mean Squared Error (MSE)

R² Score

📦 Requirements
nginx
Copy code
streamlit
pandas
numpy
scikit-learn
