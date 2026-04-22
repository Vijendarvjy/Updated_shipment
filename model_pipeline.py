import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Train (1).csv")

# Drop ID (not useful)
df = df.drop(columns=['ID'])

# -------------------------------
# FEATURE ENGINEERING FUNCTION
# -------------------------------
def feature_engineering(X):
    X = X.copy()

    X['Cost_to_Weight_ratio'] = X['Cost_of_the_Product'] / (X['Weight_in_gms'] + 1)
    X['Cost*Weight'] = (X['Cost_of_the_Product'] * X['Weight_in_gms']) / 100000
    X['Discount_Ratio'] = X['Discount_offered'] / (X['Cost_of_the_Product'] + 1)
    X['CareCalls_to_Purchases'] = X['Customer_care_calls'] / (X['Prior_purchases'] + 1)
    X['CostWeight_Discount_Interaction'] = X['Cost_to_Weight_ratio'] * (X['Discount_offered'] + 1)

    return X

# -------------------------------
# SPLIT
# -------------------------------
X = df.drop('Reached.on.Time_Y.N', axis=1)
y = df['Reached.on.Time_Y.N']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# COLUMN GROUPS
# -------------------------------
num_cols = [
    'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
    'Prior_purchases', 'Discount_offered', 'Weight_in_gms',
    'Cost_to_Weight_ratio', 'Cost*Weight', 'Discount_Ratio',
    'CareCalls_to_Purchases', 'CostWeight_Discount_Interaction'
]

cat_cols = ['Warehouse_block', 'Mode_of_Shipment']
label_cols = ['Product_importance', 'Gender']

# -------------------------------
# LABEL ENCODING (inside pipeline)
# -------------------------------
from sklearn.preprocessing import OrdinalEncoder

label_transformer = OrdinalEncoder()

# -------------------------------
# PREPROCESSOR
# -------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ("label", OrdinalEncoder(), label_cols)
])

# -------------------------------
# FINAL PIPELINE
# -------------------------------
pipeline = ImbPipeline(steps=[
    ("feature_engineering", FunctionTransformer(feature_engineering)),
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2,
        reg_alpha=0.5,
        eval_metric='logloss',
        random_state=42
    ))
])

# -------------------------------
# TRAIN
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# SAVE PIPELINE
# -------------------------------
joblib.dump(pipeline, "shipment_pipeline.pkl")

print("✅ Pipeline saved successfully!")
