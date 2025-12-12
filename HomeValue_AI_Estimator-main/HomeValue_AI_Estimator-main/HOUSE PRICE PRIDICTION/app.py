import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("D:\\vs code\\Projects\\HOUSE PRICE PRIDICTION\\House_data.csv")

# Drop unnecessary columns (id, date not useful for prediction)
if 'id' in data.columns:
    data = data.drop(columns=['id'])
if 'date' in data.columns:
    data = data.drop(columns=['date'])

# Features (X) and Target (y)
y = data['price']
X = data.drop(columns=['price'])

# Convert categorical to numerical if needed
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏡 House Price Predictor")
st.write("Enter the features of the house to predict its price.")

# Dynamic feature inputs
user_input = {}
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.number):
        user_input[col] = st.number_input(
            f"{col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    else:
        user_input[col] = st.selectbox(f"{col}", options=X[col].unique())

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Align columns with training data
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ₹ {prediction:,.2f}")

# Model evaluation metrics
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))
