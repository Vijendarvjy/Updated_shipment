import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("shipment_pipeline.pkl")

st.title("📦 Shipment Predictor")

with st.form("form"):
    warehouse = st.selectbox("Warehouse", ['A','B','C','D','F'])
    mode = st.selectbox("Shipment Mode", ['Flight','Road','Ship'])
    care = st.slider("Care Calls", 2,7,4)
    rating = st.slider("Rating",1,5,3)
    cost = st.number_input("Cost",90,320,200)
    purchases = st.slider("Prior Purchases",2,10,3)
    importance = st.selectbox("Importance", ['low','medium','high'])
    gender = st.selectbox("Gender", ['M','F'])
    discount = st.number_input("Discount",1,65,10)
    weight = st.number_input("Weight",1000,8000,3000)

    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
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

        prob = model.predict_proba(input_df)[0][1]
        pred = 1 if prob >= 0.5 else 0

        if pred == 1:
            st.success(f"✅ On-Time (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Delayed (Confidence: {prob:.2f})")
