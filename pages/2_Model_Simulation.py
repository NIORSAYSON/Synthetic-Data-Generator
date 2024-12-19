import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Model Simulation", page_icon="🔍", layout="wide")

st.header("Model Simulation", divider="rainbow")

# Sidebar for file uploads
with st.sidebar:
    st.subheader("Upload Files")
    uploaded_model_file = st.file_uploader("Trained Model (Pickle)", type="pkl")
    uploaded_encoder_file = st.file_uploader("Target Encoder (Pickle)", type="pkl")

if uploaded_model_file and uploaded_encoder_file:
    # Load the model
    model = pickle.load(uploaded_model_file) 
    target_encoder = pickle.load(uploaded_encoder_file)
    st.sidebar.success("Files loaded successfully!")

    # Extract feature names
    feature_names = getattr(model, "feature_names_in_", [f"Feature_{i+1}" for i in range(getattr(model, "n_features_in_", 0))])

    # Two-column layout
    col1, col2 = st.columns(2)
    # Input Features Form
    with col1:
        st.subheader("Input Features")
        with st.form("input_form"):
            feature_values = {}
            for feature in feature_names:
                feature_values[feature] = st.text_input(feature, value="")  # Blank by default
            
            submitted = st.form_submit_button("Simulate Prediction")

    if submitted:
        # Convert input values to a DataFrame and handle blanks
        try:
            # Convert text inputs to float
            input_data = pd.DataFrame([{feature: float(value) for feature, value in feature_values.items()}])
            
            # Make predictions
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
            
            # Decode prediction if target_encoder is available
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
        except ValueError as e:
            col1, col2 = st.columns([2, 2])  # Split into two equal columns

            with col1:
                st.error(f"Input error: {e}. Please ensure all inputs are valid numbers.")

else:
    st.write("Please upload the required files to start.")