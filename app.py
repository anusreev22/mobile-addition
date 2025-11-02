import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------
# Helper function to load models safely with logs
# -------------------------------------------------
def load_model(path):
    abs_path = os.path.abspath(path)
    print(f"Trying to load: {abs_path}")
    try:
        model = joblib.load(path)
        print(f"[SUCCESS] Loaded model: {abs_path}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] File not found: {abs_path}")
        return f"FileNotFoundError: {abs_path}"
    except Exception as e:
        print(f"[ERROR] Failed to load {abs_path}: {e}")
        return str(e)

# -------------------------------------------------
# Load the 4 best models
# -------------------------------------------------
models = {
    'Stacking Ensemble ⭐ (Best)': load_model('stacking_ensemble_pipeline.joblib'),
    'XGBoost': load_model('xgboost_pipeline.joblib'),
    'CatBoost': load_model('catboost_pipeline.joblib'),
    'Random Forest': load_model('random_forest_pipeline.joblib')
}

# Check for loading errors
failed_models = [name for name, model in models.items() if isinstance(model, str)]
if failed_models:
    st.error(f"Error loading models: {', '.join(failed_models)}. "
             "Please run the Jupyter Notebook ('Teen_Addiction_Analysis.ipynb') first to create the model files.")
    # Filter out failed models
    models = {name: model for name, model in models.items() if not isinstance(model, str)}
    print(f"Failed models: {failed_models}")  # Terminal log
    if not models:
        st.stop()  # Stop if no models loaded

# -------------------------------------------------
# Load the feature columns
# -------------------------------------------------
try:
    MODEL_FEATURE_COLUMNS = joblib.load('model_feature_columns.joblib')
    print("[SUCCESS] Loaded feature columns: model_feature_columns.joblib")
except FileNotFoundError:
    st.error("ERROR: 'model_feature_columns.joblib' not found. Please run the Jupyter Notebook.")
    print("[ERROR] Feature columns file not found: model_feature_columns.joblib")
    st.stop()
except Exception as e:
    st.error(f"ERROR loading feature columns: {e}")
    print(f"[ERROR] Failed to load feature columns: {e}")
    st.stop()

# -------------------------------------------------
# Static dropdown options (no CSV needed)
# -------------------------------------------------
gender_options = ["Female", "Other", "Male"]
purpose_options = ["Browsing", "Education", "Social Media", "Gaming", "Other"]

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def get_recommendation(score):
    score = round(score, 2)
    if score < 3.0:
        return (f"**LEVEL: LOW ({score})**",
                "RECOMMENDATION: Habits appear healthy. Continue to monitor and "
                "prioritize offline activities and good sleep hygiene.")
    elif 3.0 <= score <= 7.0:
        return (f"**LEVEL: MODERATE ({score})**",
                "RECOMMENDATION: At-risk. Implement proactive strategies. "
                "Create 'tech-free' zones (like the dinner table) and "
                "encourage setting daily time limits on social media apps.")
    else:  # score > 7.0
        return (f"**LEVEL: HIGH/DANGEROUS ({score})**",
                "RECOMMENDATION: Immediate intervention needed. This level is linked to poor sleep, "
                "high anxiety, and low academic performance. \n\n"
                "**ACTION:** Enforce a strict 'no phone in the bedroom at night' rule. "
                "Replace phone alarms with a real alarm clock. "
                "Seek professional guidance if anxiety or depression are high.")

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Teen Phone Addiction Predictor", layout="wide")
st.title('📱 Teen Phone Addiction Predictor')
st.write("This app predicts the phone addiction level of a teen (on a scale of 0–10) using a trained machine learning model.")

# Model Selection
st.subheader("1. Select Your Model")
model_name = st.selectbox(
    "Choose a model to use for prediction:",
    options=list(models.keys())
)
model = models[model_name]

st.divider()
st.subheader("2. Enter Your Data")

col1, col2 = st.columns(2)
user_input = {}

# -------------------------------------------------
# Input Fields
# -------------------------------------------------
with col1:
    st.header("Behavioral & Academic")
    user_input['Daily_Usage_Hours'] = st.number_input("Daily Usage Hours (Avg)", 0.0, 24.0, 5.0, 0.1)
    user_input['Weekend_Usage_Hours'] = st.number_input("Weekend Usage Hours (Avg)", 0.0, 24.0, 7.0, 0.1)
    user_input['Sleep_Hours'] = st.number_input("Sleep Hours (Avg)", 0.0, 16.0, 7.0, 0.1)
    user_input['Academic_Performance'] = st.number_input("Academic Performance (Score 0–100)", 0, 100, 75)
    user_input['Screen_Time_Before_Bed'] = st.number_input("Screen Time Before Bed (Hours)", 0.0, 10.0, 1.0, 0.1)
    user_input['Phone_Checks_Per_Day'] = st.number_input("Phone Checks Per Day", 0, 1000, 80)
    user_input['Apps_Used_Daily'] = st.number_input("Apps Used Daily", 0, 100, 15)
    user_input['Phone_Usage_Purpose'] = st.selectbox("Main Phone Usage Purpose", options=purpose_options, index=2)

with col2:
    st.header("Personal & Social")
    user_input['Age'] = st.number_input("Age", 10, 19, 15)
    user_input['Gender'] = st.selectbox("Gender", options=gender_options, index=0)
    user_input['School_Grade'] = st.number_input("School Grade", 5, 12, 9)
    user_input['Anxiety_Level'] = st.number_input("Anxiety Level (0–10)", 0, 10, 5)
    user_input['Depression_Level'] = st.number_input("Depression Level (0–10)", 0, 10, 5)
    user_input['Self_Esteem'] = st.number_input("Self Esteem (0–10)", 0, 10, 5)
    user_input['Social_Interactions'] = st.number_input("Social Interactions (Score 0–10)", 0, 10, 5)
    user_input['Exercise_Hours'] = st.number_input("Exercise Hours (Per Week)", 0.0, 20.0, 1.0, 0.1)
    user_input['Family_Communication'] = st.number_input("Family Communication (Score 0–10)", 0, 10, 5)
    user_input['Parental_Control'] = st.number_input("Parental Control (0=No, 1=Yes)", 0, 1, 0)
    user_input['Time_on_Social_Media'] = st.number_input("Time on Social Media (Hours/Day)", 0.0, 24.0, 2.0, 0.1)
    user_input['Time_on_Gaming'] = st.number_input("Time on Gaming (Hours/Day)", 0.0, 24.0, 1.0, 0.1)
    user_input['Time_on_Education'] = st.number_input("Time on Education (Hours/Day)", 0.0, 24.0, 1.0, 0.1)

st.divider()

# -------------------------------------------------
# Prediction Button and Output
# -------------------------------------------------
if st.button('Predict Addiction Level', use_container_width=True, type="primary"):
    user_df = pd.DataFrame([user_input])

    try:
        # Feature Engineering
        user_df['Sleep_Deficit'] = 8.0 - user_df['Sleep_Hours']
        user_df['Screen_vs_Sleep_Ratio'] = user_df['Daily_Usage_Hours'] / (user_df['Sleep_Hours'] + 1e-6)
        user_df['Mental_Health_Score'] = (user_df['Anxiety_Level'] + user_df['Depression_Level']) - user_df['Self_Esteem']
        user_df['Usage_Difference'] = user_df['Weekend_Usage_Hours'] - user_df['Daily_Usage_Hours']
        user_df['Mental_x_Screen'] = user_df['Mental_Health_Score'] * user_df['Daily_Usage_Hours']
        user_df['Sleep_x_Academic'] = user_df['Sleep_Hours'] * user_df['Academic_Performance']
        user_df['Anxiety_x_Usage'] = user_df['Anxiety_Level'] * user_df['Daily_Usage_Hours']
        user_df['Social_x_Exercise'] = user_df['Social_Interactions'] * user_df['Exercise_Hours']
        user_df['Usage_Hours_Squared'] = user_df['Daily_Usage_Hours'] ** 2
        user_df['Sleep_Hours_Squared'] = user_df['Sleep_Hours'] ** 2
        user_df['Mental_Health_Squared'] = user_df['Mental_Health_Score'] ** 2
        user_df['Total_Screen_Time'] = (
            user_df['Time_on_Social_Media'] + user_df['Time_on_Gaming'] + user_df['Daily_Usage_Hours']
        )
        user_df['Phone_Checks_Normalized'] = user_df['Phone_Checks_Per_Day'] / 200.0
        user_df['Academic_Normalized'] = user_df['Academic_Performance'] / 100.0
        user_df['Lifestyle_Balance'] = (
            user_df['Exercise_Hours'] + user_df['Social_Interactions'] + user_df['Family_Communication']
        ) - user_df['Daily_Usage_Hours']
        user_df['Sleep_Quality'] = user_df['Sleep_Hours'] - user_df['Screen_Time_Before_Bed']
        user_df['Risk_Factor_Score'] = (
            user_df['Anxiety_Level'] + user_df['Depression_Level'] + user_df['Daily_Usage_Hours']
            - user_df['Self_Esteem'] - user_df['Sleep_Hours']
        )

        user_df = user_df[MODEL_FEATURE_COLUMNS]

        # Prediction
        prediction = model.predict(user_df)
        predicted_score = prediction[0]

        level_text, rec_text = get_recommendation(predicted_score)

        st.subheader(f"Prediction Result (using {model_name})")
        if predicted_score > 7:
            st.error(level_text)
        elif predicted_score > 3:
            st.warning(level_text)
        else:
            st.success(level_text)
        st.write(rec_text)

        print(f"[SUCCESS] Prediction: {predicted_score} ({model_name})")  # Terminal log

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        print(f"[ERROR] Prediction error: {e}")  # Terminal log
