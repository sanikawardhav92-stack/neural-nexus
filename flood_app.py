import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Flood Risk Prediction",
    page_icon="🌊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_flood_model.pkl")


model = load_model()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🌊 Flood Risk Prediction Dashboard")
st.markdown(
    "### Predict flood risk using environmental and infrastructure factors"
)

st.divider()

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

features = [
    'MonsoonIntensity',
    'TopographyDrainage',
    'RiverManagement',
    'Deforestation',
    'Urbanization',
    'ClimateChange',
    'DamsQuality',
    'Siltation',
    'AgriculturalPractices',
    'Encroachments',
    'IneffectiveDisasterPreparedness',
    'DrainageSystems',
    'CoastalVulnerability',
    'Landslides',
    'Watersheds',
    'DeterioratingInfrastructure',
    'PopulationScore',
    'WetlandLoss',
    'InadequatePlanning',
    'PoliticalFactors'
]

# ---------------------------------------------------
# INPUT FORM
# ---------------------------------------------------

st.subheader("📋 Enter Environmental & Infrastructure Details")

with st.form("flood_prediction_form"):

    col1, col2 = st.columns(2)

    input_data = {}

    for i, feature in enumerate(features):

        column = col1 if i % 2 == 0 else col2

        with column:
            input_data[feature] = st.slider(
                feature,
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

    submitted = st.form_submit_button(
        "🔍 Predict Flood Risk",
        use_container_width=True
    )

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if submitted:

    # Create dataframe in EXACT training feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        flood_probability = probabilities[1]
    else:
        flood_probability = float(prediction)

    flood_percentage = flood_probability * 100

    st.divider()

    st.subheader("🚨 Flood Risk Result")

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    if prediction == 1:

        st.error("🌊 HIGH FLOOD RISK")

        st.metric(
            label="Flood Probability",
            value=f"{flood_percentage:.2f}%"
        )

    else:

        st.success("✅ LOW FLOOD RISK")

        st.metric(
            label="Flood Probability",
            value=f"{flood_percentage:.2f}%"
        )

    # ------------------------------------------------
    # RISK CATEGORY
    # ------------------------------------------------

    if flood_percentage >= 75:
        risk_category = "Very High Risk"

    elif flood_percentage >= 50:
        risk_category = "High Risk"

    elif flood_percentage >= 25:
        risk_category = "Moderate Risk"

    else:
        risk_category = "Low Risk"

    st.info(f"Risk Category: **{risk_category}**")

    # ------------------------------------------------
    # PROBABILITY CHART
    # ------------------------------------------------

    st.subheader("📊 Flood Risk Probability")

    probability_df = pd.DataFrame({
        "Category": ["No Flood", "Flood"],
        "Probability": [
            1 - flood_probability,
            flood_probability
        ]
    })

    fig = px.bar(
        probability_df,
        x="Category",
        y="Probability",
        text="Probability",
        title="Flood Prediction Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------
    # INPUT VALUES CHART
    # ------------------------------------------------

    st.subheader("📈 Input Feature Analysis")

    feature_df = pd.DataFrame({
        "Feature": list(input_data.keys()),
        "Value": list(input_data.values())
    })

    fig2 = px.bar(
        feature_df,
        x="Value",
        y="Feature",
        orientation="h",
        title="Entered Feature Values"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ------------------------------------------------
    # INPUT TABLE
    # ------------------------------------------------

    st.subheader("📋 Input Summary")

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )