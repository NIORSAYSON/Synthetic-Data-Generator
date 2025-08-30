import streamlit as st
import pandas as pd
import pickle
import numpy as np
import traceback
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_error(error_msg, exception=None):
    """Log error messages for debugging"""
    logger.error(f"Error: {error_msg}")
    if exception:
        logger.error(f"Exception details: {str(exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def safe_pickle_load(file_obj, file_type="file"):
    """Safely load pickle files with comprehensive error handling"""
    try:
        if file_obj is None:
            return None, f"No {file_type} provided"
        
        # Reset file pointer to beginning
        file_obj.seek(0)
        
        # Try to load the pickle file
        try:
            loaded_object = pickle.load(file_obj)
            return loaded_object, None
        except pickle.UnpicklingError as e:
            return None, f"Invalid pickle format for {file_type}: {str(e)}"
        except EOFError:
            return None, f"Incomplete or corrupted {file_type} file"
        except Exception as e:
            return None, f"Error loading {file_type}: {str(e)}"
            
    except Exception as e:
        log_error(f"Critical error loading {file_type}", e)
        return None, f"Critical error loading {file_type}: {str(e)}"

def validate_model(model):
    """Validate that the loaded model is suitable for prediction"""
    try:
        if model is None:
            return False, "Model is None"
        
        # Check if model has required methods
        if not hasattr(model, 'predict'):
            return False, "Model doesn't have predict method"
        
        # Try to get feature information
        if hasattr(model, 'feature_names_in_'):
            return True, "Model validation passed"
        elif hasattr(model, 'n_features_in_'):
            return True, "Model validation passed"
        else:
            return True, "Model validation passed (limited feature info)"
            
    except Exception as e:
        log_error("Error validating model", e)
        return False, f"Model validation error: {str(e)}"

def validate_encoder(encoder):
    """Validate that the loaded encoder is functional"""
    try:
        if encoder is None:
            return False, "Encoder is None"
        
        if not hasattr(encoder, 'inverse_transform'):
            return False, "Encoder doesn't have inverse_transform method"
        
        if not hasattr(encoder, 'classes_'):
            return False, "Encoder doesn't have classes_ attribute"
        
        return True, "Encoder validation passed"
        
    except Exception as e:
        log_error("Error validating encoder", e)
        return False, f"Encoder validation error: {str(e)}"

def safe_feature_extraction(model):
    """Safely extract feature names and count from model"""
    try:
        feature_names = []
        n_features = 0
        
        # Try to get feature names
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            n_features = len(feature_names)
        elif hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        else:
            # Fallback - try to infer from model type
            if hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    n_features = model.coef_.shape[1]
                else:
                    n_features = len(model.coef_)
            elif hasattr(model, 'feature_importances_'):
                n_features = len(model.feature_importances_)
            else:
                # Last resort - assume common number
                n_features = 4
                
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        return feature_names, n_features, None
        
    except Exception as e:
        log_error("Error extracting features from model", e)
        return [f"Feature_{i+1}" for i in range(4)], 4, str(e)

def validate_input_values(feature_values, feature_names):
    """Validate and convert input values to appropriate format"""
    try:
        processed_values = {}
        errors = []
        
        for feature in feature_names:
            if feature not in feature_values:
                errors.append(f"Missing value for {feature}")
                continue
                
            value = feature_values[feature].strip()
            
            if value == "":
                errors.append(f"Empty value for {feature}")
                continue
            
            try:
                # Try to convert to float
                float_value = float(value)
                
                # Check for invalid numbers
                if np.isnan(float_value) or np.isinf(float_value):
                    errors.append(f"Invalid number for {feature}: {value}")
                    continue
                    
                processed_values[feature] = float_value
                
            except ValueError:
                errors.append(f"Could not convert '{value}' to number for {feature}")
                continue
        
        return processed_values, errors
        
    except Exception as e:
        log_error("Error validating input values", e)
        return {}, [f"Validation error: {str(e)}"]

def safe_prediction(model, input_data, target_encoder=None):
    """Safely make predictions with comprehensive error handling"""
    try:
        if model is None:
            return None, None, "No model available"
        
        if input_data is None or input_data.empty:
            return None, None, "No input data provided"
        
        # Validate input data shape
        expected_features = getattr(model, 'n_features_in_', None)
        if expected_features and input_data.shape[1] != expected_features:
            return None, None, f"Expected {expected_features} features, got {input_data.shape[1]}"
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            
            if len(prediction) == 0:
                return None, None, "Model returned empty prediction"
                
        except ValueError as e:
            if "Input contains NaN" in str(e):
                return None, None, "Input contains invalid values (NaN). Please check your inputs."
            else:
                return None, None, f"Prediction error: {str(e)}"
        except Exception as e:
            return None, None, f"Model prediction failed: {str(e)}"
        
        # Get prediction probabilities if available
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            try:
                prediction_proba = model.predict_proba(input_data)
            except Exception as e:
                log_error("Error getting prediction probabilities", e)
                # Continue without probabilities
        
        # Decode prediction if target_encoder is available
        decoded_prediction = prediction.copy()
        if target_encoder:
            try:
                decoded_prediction = target_encoder.inverse_transform(prediction)
            except Exception as e:
                log_error("Error decoding prediction", e)
                # Use original prediction if decoding fails
                decoded_prediction = prediction
        
        return decoded_prediction, prediction_proba, None
        
    except Exception as e:
        log_error("Critical error in prediction", e)
        return None, None, f"Critical prediction error: {str(e)}"

st.set_page_config(page_title="Model Simulation", page_icon="🔍", layout="wide")

st.header("Model Simulation", divider="rainbow")

# Initialize session state for error tracking
if "simulation_errors" not in st.session_state:
    st.session_state.simulation_errors = []

# Sidebar for file uploads with enhanced error handling
with st.sidebar:
    st.subheader("Upload Files")
    
    # File upload section with validation
    uploaded_model_file = st.file_uploader(
        "Trained Model (Pickle)", 
        type="pkl",
        help="Upload a trained machine learning model saved as a pickle file"
    )
    
    uploaded_encoder_file = st.file_uploader(
        "Target Encoder (Pickle)", 
        type="pkl",
        help="Upload the target encoder used during training (if applicable)"
    )
    
    # Display upload status
    if uploaded_model_file:
        st.success("✅ Model file uploaded")
    else:
        st.info("📁 Please upload a model file")
        
    if uploaded_encoder_file:
        st.success("✅ Encoder file uploaded")
    else:
        st.info("📁 Please upload an encoder file (optional)")

# Main content area
if uploaded_model_file and uploaded_encoder_file:
    try:
        # Load the files with comprehensive error handling
        with st.spinner("Loading model and encoder..."):
            
            # Load model
            model, model_error = safe_pickle_load(uploaded_model_file, "model")
            if model is None:
                st.error(f"❌ Failed to load model: {model_error}")
                st.stop()
            
            # Validate model
            model_valid, model_validation_msg = validate_model(model)
            if not model_valid:
                st.error(f"❌ Model validation failed: {model_validation_msg}")
                st.stop()
            
            # Load encoder
            target_encoder, encoder_error = safe_pickle_load(uploaded_encoder_file, "encoder")
            if target_encoder is None:
                st.error(f"❌ Failed to load encoder: {encoder_error}")
                st.stop()
            
            # Validate encoder
            encoder_valid, encoder_validation_msg = validate_encoder(target_encoder)
            if not encoder_valid:
                st.error(f"❌ Encoder validation failed: {encoder_validation_msg}")
                st.stop()
                
        st.sidebar.success("✅ All files loaded and validated successfully!")

        # Extract feature information safely
        feature_names, n_features, feature_error = safe_feature_extraction(model)
        
        if feature_error:
            st.warning(f"⚠️ Feature extraction warning: {feature_error}")
        
        st.info(f"Model expects {n_features} features: {', '.join(feature_names)}")

    except Exception as e:
        log_error("Critical error loading files", e)
        st.error("❌ A critical error occurred while loading the files. Please check your files and try again.")
        st.stop()
        
else:
    st.write("Please upload the required files to start.")
    st.info("💡 **Instructions:**")
    st.write("1. Upload a trained model file (`.pkl` format)")
    st.write("2. Upload the corresponding target encoder file")
    st.write("3. Enter feature values for prediction")
    st.write("4. Click 'Simulate Prediction' to get results")
    st.stop()

# Feature input section with error handling
col1, col2 = st.columns(2)

# Initialize session state for example values
if "use_example_values" not in st.session_state:
    st.session_state.use_example_values = False
if "example_values_generated" not in st.session_state:
    st.session_state.example_values_generated = False

# Generate example values based on feature names
def generate_example_values(feature_names):
    """Generate realistic example values based on feature names"""
    example_vals = {}
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Generate realistic values based on feature name patterns
        if 'vehicle' in feature_lower and 'density' in feature_lower:
            example_vals[feature] = "120.5"
        elif 'pedestrian' in feature_lower and 'density' in feature_lower:
            example_vals[feature] = "45.2"
        elif 'speed' in feature_lower:
            example_vals[feature] = "25.8"
        elif 'emission' in feature_lower:
            example_vals[feature] = "60.3"
        elif 'temperature' in feature_lower:
            example_vals[feature] = "22.5"
        elif 'humidity' in feature_lower:
            example_vals[feature] = "65.0"
        elif 'pressure' in feature_lower:
            example_vals[feature] = "1013.2"
        elif 'wind' in feature_lower:
            example_vals[feature] = "8.5"
        elif 'feature_1' in feature_lower:
            example_vals[feature] = "100.0"
        elif 'feature_2' in feature_lower:
            example_vals[feature] = "75.5"
        elif 'feature_3' in feature_lower:
            example_vals[feature] = "50.2"
        elif 'feature_4' in feature_lower:
            example_vals[feature] = "25.8"
        else:
            # Default random-ish but consistent values
            hash_val = abs(hash(feature)) % 1000
            example_vals[feature] = f"{hash_val / 10:.1f}"
    
    return example_vals

# Input Features Form
with col1:
    st.subheader("Input Features")
    
    # # Display feature requirements
    # st.write(f"**Required Features ({len(feature_names)}):**")
    # for i, feature in enumerate(feature_names, 1):
    #     st.write(f"{i}. {feature}")
    
    # st.divider()
    
    with st.form("input_form"):
        feature_values = {}
        
        # Generate example values if requested
        if st.session_state.use_example_values:
            example_vals = generate_example_values(feature_names)
        else:
            example_vals = {feature: "" for feature in feature_names}
        
        # Create input fields for each feature
        for feature in feature_names:
            feature_values[feature] = st.text_input(
                f"{feature}",
                value=example_vals[feature],
                placeholder="Enter numeric value",
                help=f"Enter a numeric value for {feature}",
                key=f"input_{feature}"
            )
        
        # Add example values button
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            example_clicked = st.form_submit_button("📝 Fill Example Values", type="secondary")
            if example_clicked:
                st.session_state.use_example_values = True
                st.session_state.example_values_generated = True
                st.rerun()
        
        with col_btn2:
            clear_clicked = st.form_submit_button("🗑️ Clear Values", type="secondary")
            if clear_clicked:
                st.session_state.use_example_values = False
                st.session_state.example_values_generated = False
                st.rerun()
            
        submitted = st.form_submit_button("🔮 Simulate Prediction", type="primary")
        
        # Show status message if example values were filled
        if st.session_state.example_values_generated and st.session_state.use_example_values:
            st.success("✅ Example values filled! You can modify them or click 'Simulate Prediction'")
            
            # Show what example values were generated
            with st.expander("📋 View Generated Example Values"):
                example_display = generate_example_values(feature_names)
                for feature, value in example_display.items():
                    st.write(f"• **{feature}**: {value}")
                    
        elif st.session_state.example_values_generated and not st.session_state.use_example_values:
            st.info("ℹ️ Values cleared. Enter your own values or use example values.")

# Results section
with col2:
    st.subheader("Prediction Results")
    
    if submitted:
        try:
            # Validate input values
            processed_values, validation_errors = validate_input_values(feature_values, feature_names)
            
            if validation_errors:
                st.error("❌ **Input Validation Errors:**")
                for error in validation_errors:
                    st.error(f"• {error}")
                st.info("💡 Please correct the errors above and try again.")
            else:
                # Convert to DataFrame for prediction
                try:
                    input_data = pd.DataFrame([processed_values])
                    
                    # Make prediction safely
                    with st.spinner("Making prediction..."):
                        prediction, prediction_proba, prediction_error = safe_prediction(
                            model, input_data, target_encoder
                        )
                    
                    if prediction_error:
                        st.error(f"❌ Prediction failed: {prediction_error}")
                        st.info("💡 Please check your input values and try again.")
                    else:
                        # Display successful results
                        st.success("✅ **Prediction Successful!**")
                        st.write(f"**Predicted Class:** `{prediction[0]}`")
                        
                        # Display confidence scores if available
                        if prediction_proba is not None:
                            try:
                                st.write("**Confidence Scores:**")
                                
                                # Create probability DataFrame
                                if target_encoder and hasattr(target_encoder, 'classes_'):
                                    class_names = target_encoder.classes_
                                else:
                                    class_names = [f"Class_{i}" for i in range(len(prediction_proba[0]))]
                                
                                proba_df = pd.DataFrame(
                                    prediction_proba, 
                                    columns=class_names
                                )
                                
                                # Display as horizontal bar chart
                                st.bar_chart(proba_df.T, use_container_width=True)
                                
                                # Display numerical values
                                st.write("**Detailed Probabilities:**")
                                for class_name, prob in zip(class_names, prediction_proba[0]):
                                    st.write(f"• {class_name}: {prob:.4f} ({prob*100:.2f}%)")
                                    
                            except Exception as e:
                                log_error("Error displaying prediction probabilities", e)
                                st.warning("⚠️ Could not display confidence scores.")
                        else:
                            st.info("ℹ️ This model doesn't provide confidence scores.")
                            
                except Exception as e:
                    log_error("Error creating input DataFrame", e)
                    st.error("❌ Failed to process input data. Please check your values.")
                    
        except Exception as e:
            log_error("Critical error in prediction process", e)
            st.error("❌ A critical error occurred during prediction. Please try again.")
    else:
        # Show placeholder when no prediction has been made
        st.info("👈 Enter feature values and click 'Simulate Prediction' to see results")
        
        # Show example of what results will look like
        st.write("**Example Output:**")
        st.code("""
✅ Prediction Successful!
Predicted Class: High Congestion Zone
Confidence: 85.3%
        """)

# Additional information section
st.divider()
st.subheader("Model Information")

try:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Features", len(feature_names))
        
    with col2:
        if target_encoder and hasattr(target_encoder, 'classes_'):
            st.metric("Classes", len(target_encoder.classes_))
        else:
            st.metric("Classes", "Unknown")
            
    with col3:
        model_type = type(model).__name__
        st.metric("Model Type", model_type)

    # Display available classes
    if target_encoder and hasattr(target_encoder, 'classes_'):
        st.write("**Available Classes:**")
        for i, class_name in enumerate(target_encoder.classes_, 1):
            st.write(f"{i}. {class_name}")
            
except Exception as e:
    log_error("Error displaying model information", e)
    st.warning("⚠️ Could not display complete model information.")

# Tips section
with st.expander("💡 Tips for Better Predictions"):
    st.write("""
    **For accurate predictions:**
    - Ensure all feature values are numeric
    - Use values in a similar range to your training data
    - Avoid extreme outliers unless they represent real scenarios
    - Check that feature names match your training data
    
    **Troubleshooting:**
    - If you get errors, verify your model and encoder files are compatible
    - Make sure you're using the same preprocessing as during training
    - Contact support if you continue experiencing issues
    """)