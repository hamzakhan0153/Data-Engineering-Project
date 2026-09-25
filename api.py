from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import dill
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="Customer Churn Prediction",
    description="Customer Churn Prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# Load model and feature dictionary ONCE
# --------------------------------------------------

with open(BASE_DIR / "pipeline.pkl", "rb") as file:
    model = dill.load(file)

my_feature_dict = load(BASE_DIR / "my_feature_dict.pkl")


# --------------------------------------------------
# Repair serialized transformer
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]

senior_citizen_transformer = preprocessor.named_steps["transform_sc"]

column_transformer = preprocessor.named_steps["preprocessor_stage_2"]

categorical_columns = list(
    column_transformer.transformers[0][2]
)


# Find duplicate SENIORCITIZEN column
first_index = categorical_columns.index("SENIORCITIZEN")

try:
    duplicate_index = categorical_columns.index(
        "SENIORCITIZEN",
        first_index + 1
    )

    categorical_columns[duplicate_index] = "SENIORCITIZEN_DUPLICATE"

    column_transformer.transformers[0] = (
        column_transformer.transformers[0][0],
        column_transformer.transformers[0][1],
        categorical_columns,
    )

except ValueError:
    # No duplicate found
    pass


def transform_senior_citizen(self, data):

    transformed = data.copy()

    transformed["SENIORCITIZEN"] = (
        transformed["SENIORCITIZEN"]
        .map({1: "Yes", 0: "No"})
    )

    # Only create duplicate if pipeline expects it
    if "SENIORCITIZEN_DUPLICATE" in transformed.columns:
        transformed["SENIORCITIZEN_DUPLICATE"] = (
            transformed["SENIORCITIZEN"]
        )

    return transformed


senior_citizen_transformer.__class__.transform = (
    transform_senior_citizen
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class CustomerData(BaseModel):
    data: dict


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# --------------------------------------------------
# Get feature information
# --------------------------------------------------

@app.get("/features")
def get_features():
    return my_feature_dict


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        # Convert incoming JSON into DataFrame
        input_data = pd.DataFrame([customer.data])

        # Make prediction
        prediction = model.predict(input_data)

        return {
            "prediction": prediction[0]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )