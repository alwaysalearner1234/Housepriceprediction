import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🏡 House Price Predictor")

# Sample dataset
data = {
    "area": [1000, 1500, 1800, 2400, 3000],
    "bedrooms": [2, 3, 3, 4, 5],
    "bathrooms": [1, 2, 2, 3, 4],
    "price": [200000, 300000, 400000, 500000, 600000]
}
df = pd.DataFrame(data)
st.write("### Training Data", df)

# Features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
st.write("### Enter House Details")
area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1500)
bedrooms = st.slider("Bedrooms", 1, 5, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")