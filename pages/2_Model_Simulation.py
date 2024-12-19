import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Model Simulation", page_icon="🔍", layout="wide")

st.header("Model Simulation")

# Sidebar for file uploads
with st.sidebar:
    st.subheader("Upload Files")
    uploaded_model_file = st.file_uploader("Trained Model (Pickle)", type="pkl")
    uploaded_encoder_file = st.file_uploader("Target Encoder (Pickle, optional)", type="pkl")

if uploaded_model_file:
    # Load the model
    model = pickle.load(uploaded_model_file)
    target_encoder = pickle.load(uploaded_encoder_file) if uploaded_encoder_file else None
    st.sidebar.success("Files loaded successfully!")

    # Extract feature names
    feature_names = getattr(model, "feature_names_in_", [f"Feature_{i+1}" for i in range(getattr(model, "n_features_in_", 0))])

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")
        with st.form("input_form"):
            feature_values = {feature: st.number_input(feature, value=0.0) for feature in feature_names}
            submitted = st.form_submit_button("Simulate Prediction")

    if submitted:
        # Prepare input data and make predictions
        input_data = pd.DataFrame([feature_values])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

        if target_encoder:
            prediction = target_encoder.inverse_transform(prediction)

        # Display results in the second column
        with col2:
            st.subheader("Prediction Results")
            st.write(f"Predicted Class: **{prediction[0]}**")

            if prediction_proba is not None:
                proba_df = pd.DataFrame(
                    prediction_proba, 
                    columns=target_encoder.classes_ if target_encoder else model.classes_
                )
                st.bar_chart(proba_df.T, use_container_width=True)
else:
    st.info("Please upload the required files to start.")