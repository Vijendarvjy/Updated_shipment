import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("shipment_pipeline.pkl")

st.set_page_config(page_title="Shipment Predictor", layout="wide")
st.title("📦 Shipment On-Time Delivery Predictor")

# Threshold slider
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)

with st.form("form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        warehouse = st.selectbox("Warehouse", ['A','B','C','D','F'])
        mode = st.selectbox("Mode", ['Flight','Road','Ship'])
        care = st.slider("Care Calls", 2,7,4)
        rating = st.slider("Rating",1,5,3)

    with col2:
        cost = st.number_input("Cost",90,320,200)
        purchases = st.slider("Prior Purchases",2,10,3)
        importance = st.selectbox("Importance", ['low','medium','high'])
        gender = st.selectbox("Gender", ['M','F'])

    with col3:
        discount = st.number_input("Discount",1,65,10)
        weight = st.number_input("Weight",1000,8000,3000)

    submit = st.form_submit_button("Predict")

if submit:
    df = pd.DataFrame([{
        'Warehouse_block': warehouse,
        'Mode_of_Shipment': mode,
        'Customer_care_calls': care,
        'Customer_rating': rating,
        'Cost_of_the_Product': cost,
        'Prior_purchases': purchases,
        'Product_importance': importance,
        'Gender': gender,
        'Discount_offered': discount,
        'Weight_in_gms': weight
    }])

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= threshold)

    st.subheader("📊 Result")

    if pred == 1:
        st.success(f"✅ On-Time Delivery (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Delayed Delivery (Confidence: {prob:.2f})")

    st.progress(float(prob))
