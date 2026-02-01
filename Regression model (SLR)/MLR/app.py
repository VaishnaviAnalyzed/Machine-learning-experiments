import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

pd.read_csv(r'C:\Users\PC World\OneDrive\Documents\Practice Machine Learning\Regression model (SLR)\MLR\House_data.csv')
# --- HEADER ---
st.title("🏠 House Price Predictor")
st.write("Enter the house details below to estimate the market value using our **MLR Model**.")
st.markdown("---")

# --- LOAD THE TRAINED MODEL ---
# Replace 'house_model.pkl' with your actual saved model filename
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please ensure 'house_model.pkl' is in the same directory.")
    st.stop()

# --- INPUT SECTION (SIDEBAR OR MAIN) ---
st.subheader("📊 House Features")

col1, col2= st.columns(2)

with col1:
    sqft = st.number_input("📏 Total Square Feet", min_value=300, max_value=10000, value=2000)
    beds = st.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3)

with col2:
    bath = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=8, value=2)
    age = st.number_input("📅 Age of Property (Years)", min_value=0, max_value=100, value=3)

# --- PREDICTION LOGIC ---
if st.button("💰 Predict Price"):
    # 1. Arrange features into a 2D array for the model
    # Note: Ensure the order matches the order used during model training
    features = np.array([[sqft, beds, bath, age]])
    
    # 2. Make Prediction
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    # 3. Display Result
    st.success(f"### Estimated Market Price: ${output:,}")
    
    # Optional visual feedback
    st.balloons()

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Scikit-Learn and Streamlit | Multiple Linear Regression Model. Built by Vaishnavi")