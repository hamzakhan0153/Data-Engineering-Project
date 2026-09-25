import streamlit as st
import requests
import os


try:
    API_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

API_URL = API_URL.rstrip("/")


st.title("Customer Churn Prediction App")
st.subheader("Based on Telecom Dataset")


# --------------------------------------------------
# Get feature information from FastAPI
# --------------------------------------------------

try:
    response = requests.get(f"{API_URL}/features")

    if response.status_code != 200:
        st.error("Unable to get feature information from API.")
        st.stop()

    my_feature_dict = response.json()

except requests.exceptions.ConnectionError:
    st.error("FastAPI backend is not running.")
    st.stop()


# --------------------------------------------------
# Categorical Features
# --------------------------------------------------

st.subheader("Categorical Features")

categorical_input = my_feature_dict.get("CATEGORICAL")

categorical_input_vals = {}

columns = categorical_input.get("Column Name")
members = categorical_input.get("Members")


# Column Name is a dictionary like:
# {"0": "GENDER", "1": "SENIORCITIZEN", ...}
if isinstance(columns, dict):
    column_names = list(columns.values())
else:
    column_names = list(columns)


for i, col in enumerate(column_names):

    # Members is a dictionary
    if isinstance(members, dict):

        # Try actual column name first
        if col in members:
            options = members[col]

        # Otherwise use string index
        elif str(i) in members:
            options = members[str(i)]

        # Otherwise use integer index
        elif i in members:
            options = members[i]

        else:
            st.error(f"No members found for column: {col}")
            st.stop()

    # Members is a list
    else:
        options = members[i]

    categorical_input_vals[col] = st.selectbox(
        col,
        options,
        key=f"categorical_{i}_{col}"
    )


# --------------------------------------------------
# Numerical Features
# --------------------------------------------------

st.subheader("Numerical Features")

numerical_input = my_feature_dict.get("NUMERICAL")

numerical_input_vals = {}

numerical_columns = numerical_input.get("Column Name")

if isinstance(numerical_columns, dict):
    numerical_column_names = list(numerical_columns.values())
else:
    numerical_column_names = list(numerical_columns)


for i, col in enumerate(numerical_column_names):

    numerical_input_vals[col] = st.number_input(
        col,
        key=f"numerical_{i}_{col}"
    )


# --------------------------------------------------
# Combine inputs
# --------------------------------------------------

input_data = {
    **categorical_input_vals,
    **numerical_input_vals
}


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json={
                "data": input_data
            }
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            print(f"Prediction: {prediction}")

            translation_dict = {
                "Yes": "Expected",
                "No": "Not Expected"
            }

            prediction_translate = translation_dict.get(
                prediction,
                "Unknown"
            )

            st.write(
                f"The Prediction is **{prediction}**, "
                f"Hence customer is **{prediction_translate}** "
                f"to churn."
            )

        else:

            try:
                error_detail = response.json().get("detail")
            except Exception:
                error_detail = response.text

            st.error(f"API Error: {error_detail}")

    except requests.exceptions.ConnectionError:

        st.error("Could not connect to FastAPI backend.")