import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from preprocess import preprocess

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load data
df = pd.read_csv("data/train.csv")

X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

# Preprocess
X_processed = preprocess(X)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_processed, y)

# Convert to ONNX
initial_type = [('input', FloatTensorType([None, X_processed.shape[1]]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open("model/shipment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model saved!")
