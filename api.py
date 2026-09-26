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

categorical_columns = list(column_transformer.transformers[0][2])

# Fix duplicated SENIORCITIZEN if the saved pipeline contains it twice
if "SENIORCITIZEN" in categorical_columns:
    first_index = categorical_columns.index("SENIORCITIZEN")
    try:
        duplicate_index = categorical_columns.index("SENIORCITIZEN", first_index + 1)
        categorical_columns[duplicate_index] = "SENIORCITIZEN_DUPLICATE"
        column_transformer.transformers[0] = (
            column_transformer.transformers[0][0],
            column_transformer.transformers[0][1],
            categorical_columns,
        )
    except ValueError:
        pass


def normalize_senior_citizen_value(value):
    if pd.isna(value):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "1"}:
            return "Yes"
        if normalized in {"no", "0"}:
            return "No"
        return value

    if value in (1, "1"):
        return "Yes"
    if value in (0, "0"):
        return "No"

    return value


def transform_senior_citizen(self, data):
    transformed = data.copy()

    if "SENIORCITIZEN" in transformed.columns:
        transformed["SENIORCITIZEN"] = (
            transformed["SENIORCITIZEN"].map(normalize_senior_citizen_value)
        )

    if "SENIORCITIZEN_DUPLICATE" in transformed.columns:
        transformed["SENIORCITIZEN_DUPLICATE"] = transformed["SENIORCITIZEN"]

    return transformed


senior_citizen_transformer.__class__.transform = transform_senior_citizen


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


def _get_expected_columns():
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    categorical_columns = my_feature_dict.get("CATEGORICAL", {}).get("Column Name", [])
    numerical_columns = my_feature_dict.get("NUMERICAL", {}).get("Column Name", [])

    if isinstance(categorical_columns, dict):
        categorical_columns = list(categorical_columns.values())
    if isinstance(numerical_columns, dict):
        numerical_columns = list(numerical_columns.values())

    return list(categorical_columns) + list(numerical_columns)


def normalize_input(raw_data: dict) -> pd.DataFrame:
    cleaned = dict(raw_data)

    if "SENIORCITIZEN" in cleaned:
        cleaned["SENIORCITIZEN"] = cleaned["SENIORCITIZEN"].replace(
            {
                "Yes": 1,
                "No": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                True: 1,
                False: 0,
            }
        )

    df = pd.DataFrame([cleaned])
    expected_columns = _get_expected_columns()
    return df.reindex(columns=expected_columns)


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        input_data = normalize_input(customer.data)
        prediction = model.predict(input_data)

        # Return a stable label format for the front-end
        prediction_label = str(prediction[0]).strip()
        if prediction_label.lower() in {"yes", "true", "1"}:
            prediction_label = "Yes"
        elif prediction_label.lower() in {"no", "false", "0"}:
            prediction_label = "No"

        return {
            "prediction": prediction_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))