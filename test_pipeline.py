import joblib
import pandas as pd

def test_model_load():
    model = joblib.load("model/shipment_pipeline.pkl")
    assert model is not None

def test_prediction():
    model = joblib.load("model/shipment_pipeline.pkl")

    sample = pd.DataFrame([{
        'Warehouse_block': 'A',
        'Mode_of_Shipment': 'Flight',
        'Customer_care_calls': 4,
        'Customer_rating': 3,
        'Cost_of_the_Product': 200,
        'Prior_purchases': 3,
        'Product_importance': 'medium',
        'Gender': 'M',
        'Discount_offered': 10,
        'Weight_in_gms': 3000
    }])

    pred = model.predict(sample)
    assert pred[0] in [0, 1]
