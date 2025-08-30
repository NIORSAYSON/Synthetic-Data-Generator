import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import traceback
import logging

LOGGER = get_logger(__name__)

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO)

def log_error(error_msg, exception=None):
    """Log error messages for debugging"""
    LOGGER.error(f"Error: {error_msg}")
    if exception:
        LOGGER.error(f"Exception details: {str(exception)}")
        LOGGER.error(f"Traceback: {traceback.format_exc()}")

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float with error handling"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError) as e:
        log_error(f"Failed to convert {value} to float", e)
        return default

def validate_input_parameters(classes, features, class_parameters):
    """Validate input parameters before data generation"""
    try:
        if not classes or len(classes) == 0:
            return False, "No classes provided. Please enter at least one class name."
        
        if not features or len(features) == 0:
            return False, "No features provided. Please enter at least one feature name."
        
        if len(classes) > 10:
            return False, "Too many classes. Please limit to 10 classes or fewer."
        
        if len(features) > 20:
            return False, "Too many features. Please limit to 20 features or fewer."
        
        # Check for valid parameters
        for cls in classes:
            if cls not in class_parameters:
                return False, f"Missing parameters for class: {cls}"
            
            for feature in features:
                feature_key = feature_to_key(feature)
                if f"{feature_key}_mean" not in class_parameters[cls]:
                    return False, f"Missing mean parameter for feature '{feature}' in class '{cls}'"
                if f"{feature_key}_std" not in class_parameters[cls]:
                    return False, f"Missing std parameter for feature '{feature}' in class '{cls}'"
                
                # Validate parameter values
                mean_val = class_parameters[cls][f"{feature_key}_mean"]
                std_val = class_parameters[cls][f"{feature_key}_std"]
                
                if not isinstance(mean_val, (int, float)) or np.isnan(mean_val):
                    return False, f"Invalid mean value for feature '{feature}' in class '{cls}'"
                if not isinstance(std_val, (int, float)) or np.isnan(std_val) or std_val <= 0:
                    return False, f"Invalid standard deviation for feature '{feature}' in class '{cls}'. Must be positive."
        
        return True, "Parameters are valid"
    except Exception as e:
        log_error("Error validating input parameters", e)
        return False, f"Validation error: {str(e)}"

st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🗂️",
    layout="wide",
)

st.sidebar.title("Configuration")

# Input for class and feature names with error handling
st.sidebar.subheader("Classes & Features")

try:
    class_names = st.sidebar.text_input("Enter class names (comma-separated)", "High Congestion Zone, Low Congestion Zone")
    feature_names = st.sidebar.text_input("Enter feature names (comma-separated)", "vehicle density (vehicles/km²), pedestrian density (people/min), average speed (km/h), emission level (µg/m³ NO₂ or CO₂ ppm)")

    # Parse class names and feature names with validation
    if not class_names.strip():
        st.sidebar.error("Please enter at least one class name.")
        st.stop()
    
    if not feature_names.strip():
        st.sidebar.error("Please enter at least one feature name.")
        st.stop()

    classes = [cls.strip() for cls in class_names.split(",") if cls.strip()]
    features = [feat.strip() for feat in feature_names.split(",") if feat.strip()]

    if len(classes) == 0:
        st.sidebar.error("No valid class names found. Please check your input.")
        st.stop()
    
    if len(features) == 0:
        st.sidebar.error("No valid feature names found. Please check your input.")
        st.stop()

except Exception as e:
    log_error("Error parsing class and feature names", e)
    st.sidebar.error("Error processing class and feature names. Please check your input format.")
    st.stop()

# Class-specific parameter settings
st.sidebar.subheader("Class-Specific Parameters")
class_parameters = {}

def feature_to_key(feature_name):
    """Convert feature name to key format used in default_parameters with error handling"""
    try:
        if not feature_name or not isinstance(feature_name, str):
            log_error(f"Invalid feature name: {feature_name}")
            return "invalid_feature"
        
        return (feature_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "/")
                .replace("²", "²")
                .replace("³", "³")
                .replace("₂", "₂"))
    except Exception as e:
        log_error("Error converting feature name to key", e)
        return "invalid_feature"

# Default feature-specific mean and std dev values
default_parameters = {
    "High Congestion Zone": {
        "vehicle_density_vehicles/km²_mean": 160,
        "vehicle_density_vehicles/km²_std": 45,
        "pedestrian_density_people/min_mean": 70,
        "pedestrian_density_people/min_std": 20,
        "average_speed_km/h_mean": 18,
        "average_speed_km/h_std": 6,
        "emission_level_µg/m³_no₂_or_co₂_ppm_mean": 75,
        "emission_level_µg/m³_no₂_or_co₂_ppm_std": 22,
    },
    "Low Congestion Zone": {
        "vehicle_density_vehicles/km²_mean": 100,
        "vehicle_density_vehicles/km²_std": 35,
        "pedestrian_density_people/min_mean": 40,
        "pedestrian_density_people/min_std": 15,
        "average_speed_km/h_mean": 28,
        "average_speed_km/h_std": 8,
        "emission_level_µg/m³_no₂_or_co₂_ppm_mean": 45,
        "emission_level_µg/m³_no₂_or_co₂_ppm_std": 18,
    },
}

for cls in classes:
    with st.sidebar.expander(f"⚙️ {cls} Parameters", expanded=True):
        try:
            st.write(f"**Configure parameters for {cls}:**")
            parameters = {}
            
            for feature in features:
                try:
                    feature_key = feature_to_key(feature)
                    
                    # Get default values (either from predefined defaults or generate random ones)
                    base_mean_default = default_parameters.get(cls, {}).get(f"{feature_key}_mean", 100.0)
                    base_std_default = default_parameters.get(cls, {}).get(f"{feature_key}_std", 10.0)
                    
                    # If randomize was triggered, generate new random values
                    if st.session_state.get("randomize_trigger", 0) > 0:
                        # Use the trigger value as seed for consistent randomization
                        random.seed(st.session_state.randomize_trigger + hash(f"{cls}_{feature_key}"))
                        mean_default = random.uniform(20, 200)
                        std_default = random.uniform(5, 30)
                    else:
                        mean_default = base_mean_default
                        std_default = base_std_default
                    
                    # Ensure defaults are valid numbers
                    mean_default = safe_float_conversion(mean_default, 100.0)
                    std_default = safe_float_conversion(std_default, 10.0)
                    
                    st.write(f"**{feature}:**")
                    
                    # Create two columns for mean and std inputs
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        parameters[f"{feature_key}_mean"] = st.number_input(
                            f"Mean",
                            min_value=0.0,
                            max_value=1000.0,
                            value=float(mean_default),
                            step=1.0,
                            key=f"{cls}_{feature_key}_mean",
                            help=f"Average value for {feature} in {cls}"
                        )
                    
                    with col2:
                        parameters[f"{feature_key}_std"] = st.number_input(
                            f"Std Dev",
                            min_value=0.01,  # Prevent zero std dev
                            max_value=200.0,
                            value=float(std_default),
                            step=0.1,
                            key=f"{cls}_{feature_key}_std",
                            help=f"Standard deviation for {feature} in {cls}"
                        )
                    
                    # Show preview of the distribution
                    if st.checkbox(f"Preview distribution for {feature}", key=f"{cls}_{feature_key}_preview"):
                        try:
                            import matplotlib.pyplot as plt
                            preview_data = np.random.normal(
                                parameters[f"{feature_key}_mean"], 
                                parameters[f"{feature_key}_std"], 
                                1000
                            )
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.hist(preview_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.set_title(f"Preview: {feature} for {cls}")
                            ax.set_xlabel(feature)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            log_error(f"Error creating preview for {feature}", e)
                            st.error("Could not generate preview")
                    
                    st.divider()
                    
                except Exception as e:
                    log_error(f"Error creating input for feature {feature} in class {cls}", e)
                    st.sidebar.error(f"Error with feature '{feature}' settings. Using default values.")
                    # Set safe default values
                    feature_key = feature_to_key(feature)
                    parameters[f"{feature_key}_mean"] = 100.0
                    parameters[f"{feature_key}_std"] = 10.0
            
            class_parameters[cls] = parameters
            
            # Add a reset button for this class
            if st.button(f"Reset {cls} to Defaults", key=f"reset_{cls}"):
                # Clear session state for this class
                keys_to_remove = []
                for key in st.session_state.keys():
                    if key.startswith(f"{cls}_") and (key.endswith('_mean') or key.endswith('_std') or key.endswith('_preview')):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del st.session_state[key]
                
                st.success(f"Reset {cls} parameters!")
                st.rerun()
                
        except Exception as e:
            log_error(f"Error creating settings for class {cls}", e)
            st.sidebar.error(f"Error with class '{cls}' settings. Please check your configuration.")

# Sample size and train/test split
st.sidebar.subheader("Data Settings")
num_samples = st.sidebar.slider("Number of samples", min_value=500, max_value=50000, value=10000)

# Add parameter summary
st.sidebar.subheader("📋 Parameter Summary")
if class_parameters:
    with st.sidebar.expander("View Current Parameters", expanded=False):
        for cls, params in class_parameters.items():
            st.write(f"**{cls}:**")
            for param_name, param_value in params.items():
                if '_mean' in param_name:
                    feature_name = param_name.replace('_mean', '').replace('_', ' ')
                    std_param = param_name.replace('_mean', '_std')
                    std_value = params.get(std_param, 'N/A')
                    st.write(f"• {feature_name}: μ={param_value:.2f}, σ={std_value:.2f}")
            st.divider()

# Add bulk actions
st.sidebar.subheader("🔧 Bulk Actions")

# Initialize randomize trigger in session state
if "randomize_trigger" not in st.session_state:
    st.session_state.randomize_trigger = 0

if st.sidebar.button("🎲 Randomize All Parameters"):
    # Clear all parameter session state to allow new random values
    keys_to_remove = []
    for cls in classes:
        for feature in features:
            feature_key = feature_to_key(feature)
            keys_to_remove.extend([
                f"{cls}_{feature_key}_mean",
                f"{cls}_{feature_key}_std",
                f"{cls}_{feature_key}_preview"
            ])
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Increment trigger to force new random values
    st.session_state.randomize_trigger += 1
    st.sidebar.success("Parameters randomized!")
    st.rerun()

if st.sidebar.button("↩️ Reset All to Defaults"):
    # Clear all parameter keys from session state
    keys_to_remove = []
    for cls in classes:
        for key in st.session_state.keys():
            if key.startswith(f"{cls}_") and (key.endswith('_mean') or key.endswith('_std') or key.endswith('_preview')):
                keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset randomize trigger
    st.session_state.randomize_trigger = 0
    st.sidebar.success("Reset to defaults!")
    st.rerun()
# test_split = st.sidebar.slider("Test size (%)", min_value=10, max_value=50, value=20)

# # Calculate train size percentage
# train_split = 100 - test_split

# # Display train and test size percentages below the sliders
# st.sidebar.write(f"Train size: {train_split}%")
# st.sidebar.write(f"Test size: {test_split}%")

# Store synthetic data in session state
if "synthetic_data" not in st.session_state:
    st.session_state["synthetic_data"] = None

def generate_data():
    """Generate synthetic data with comprehensive error handling"""
    try:
        # Validate inputs before proceeding
        is_valid, error_msg = validate_input_parameters(classes, features, class_parameters)
        if not is_valid:
            st.error(f"Data generation failed: {error_msg}")
            return None
        
        data = []
        samples_per_class = max(1, num_samples // len(classes))
        
        # Loop through each class
        for cls in classes:
            try:
                if cls not in class_parameters:
                    log_error(f"Missing parameters for class: {cls}")
                    continue
                    
                for i in range(samples_per_class):
                    try:
                        sample = {"Class": cls}  # Initialize a sample with the class label
                        
                        # Loop through each feature
                        for feature in features:
                            try:
                                feature_key = feature_to_key(feature)
                                
                                # Get the mean and std dev for this class and feature
                                mean_key = f"{feature_key}_mean"
                                std_key = f"{feature_key}_std"
                                
                                if mean_key not in class_parameters[cls] or std_key not in class_parameters[cls]:
                                    log_error(f"Missing parameters for feature {feature} in class {cls}")
                                    # Use safe defaults
                                    mean = 100.0
                                    std = 10.0
                                else:
                                    mean = safe_float_conversion(class_parameters[cls][mean_key], 100.0)
                                    std = safe_float_conversion(class_parameters[cls][std_key], 10.0)
                                
                                # Ensure std is positive
                                if std <= 0:
                                    std = 1.0
                                
                                # Generate data using normal distribution with error handling
                                try:
                                    value = np.random.normal(mean, std)
                                    # Check for invalid values
                                    if np.isnan(value) or np.isinf(value):
                                        value = mean  # Fallback to mean
                                    sample[feature] = value
                                except Exception as e:
                                    log_error(f"Error generating value for feature {feature}", e)
                                    sample[feature] = mean  # Fallback to mean
                                    
                            except Exception as e:
                                log_error(f"Error processing feature {feature} for class {cls}", e)
                                sample[feature] = 100.0  # Safe default
                        
                        data.append(sample)
                        
                    except Exception as e:
                        log_error(f"Error creating sample {i} for class {cls}", e)
                        continue
                        
            except Exception as e:
                log_error(f"Error processing class {cls}", e)
                continue
        
        if len(data) == 0:
            st.error("No data was generated. Please check your parameters.")
            return None
        
        # Shuffle the data to ensure randomness
        try:
            random.shuffle(data)
        except Exception as e:
            log_error("Error shuffling data", e)
            # Continue without shuffling
        
        # Convert to a Pandas DataFrame with error handling
        try:
            df = pd.DataFrame(data)
            
            # Validate the resulting DataFrame
            if df.empty:
                st.error("Generated DataFrame is empty.")
                return None
            
            # Check for missing values
            if df.isnull().any().any():
                log_error("Generated data contains null values")
                st.warning("Generated data contains some null values. These will be handled appropriately.")
                
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if df[numeric_cols].isin([np.inf, -np.inf]).any().any():
                log_error("Generated data contains infinite values")
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(df.mean())
            
            return df
            
        except Exception as e:
            log_error("Error creating DataFrame from generated data", e)
            st.error("Failed to create dataset from generated data.")
            return None
            
    except Exception as e:
        log_error("General error in data generation", e)
        st.error(f"An unexpected error occurred during data generation: {str(e)}")
        return None

# Function to convert a DataFrame to CSV with error handling
def convert_df_to_csv(df):
    """Convert DataFrame to CSV with error handling"""
    try:
        if df is None or df.empty:
            log_error("Cannot convert empty or None DataFrame to CSV")
            return None
        return df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        log_error("Error converting DataFrame to CSV", e)
        st.error("Failed to convert data to CSV format.")
        return None

# Main display
st.header("Synthetic Data Generator", divider='rainbow')

# Generate data button
if st.sidebar.button("Generate Data"):
    try:
        with st.spinner("Generating synthetic data..."):
            generated_data = generate_data()
            if generated_data is not None:
                st.session_state["synthetic_data"] = generated_data
                st.sidebar.success("Data generated successfully!")
            else:
                st.sidebar.error("Failed to generate data. Please check your parameters.")
    except Exception as e:
        log_error("Error in generate data button handler", e)
        st.sidebar.error("An error occurred while generating data.")
    
synthetic_data = st.session_state["synthetic_data"]

if synthetic_data is not None:
    try:
        st.write("### Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"###### Number of Classes: {len(classes)}")
          
        with col2:
            st.write(f"###### Number of Samples: {len(synthetic_data)}")
           
        # Validate data before processing
        if synthetic_data.empty:
            st.error("The generated dataset is empty.")
        elif len(synthetic_data.columns) < 2:
            st.error("The generated dataset has insufficient columns.")
        else:
            try:
                X = synthetic_data[features]
                y = synthetic_data["Class"]
                
                # Check for missing values in features
                if X.isnull().any().any():
                    st.warning("Some features contain missing values. These will be handled during scaling.")
                    X = X.fillna(X.mean())
                
                # Scale the features with error handling
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
                    X_scaled_df["Class"] = y.values
                    combined_data = pd.concat([synthetic_data.reset_index(drop=True), X_scaled_df], axis=1)
                except Exception as e:
                    log_error("Error during feature scaling", e)
                    st.error("Failed to scale features. Showing original data only.")
                    X_scaled_df = X.copy()
                    X_scaled_df["Class"] = y.values
                    combined_data = synthetic_data.copy()
                
            except KeyError as e:
                log_error(f"Missing expected columns in dataset: {e}", e)
                st.error(f"The dataset is missing expected columns: {str(e)}")
                st.stop()
            except Exception as e:
                log_error("Error processing dataset", e)
                st.error("An error occurred while processing the dataset.")
                st.stop()
                
    except Exception as e:
        log_error("Error in main data processing section", e)
        st.error("An error occurred while displaying dataset information.")
        st.stop()
                
    st.divider()

    st.write("### Generated Data Sample")
    col1, col2 = st.columns(2)
    with col1:
        st.write("###### Original Data")
        try:
            st.dataframe(combined_data.iloc[:, :len(features) + 1])
        except Exception as e:
            log_error("Error displaying original data", e)
            st.error("Failed to display original data.")
    with col2:
        st.write("###### Scaled Data")
        try:
            st.dataframe(combined_data.iloc[:, len(features) + 1:])
        except Exception as e:
            log_error("Error displaying scaled data", e)
            st.error("Failed to display scaled data.")
    
    # Download buttons for original and scaled data
    st.write("### Download Generated Data")

    col1, col2 = st.columns(2)
    with col1:
        try:
            original_csv = convert_df_to_csv(synthetic_data)
            if original_csv is not None:
                st.download_button(
                    label="Download Original Data as CSV",
                    data=original_csv,
                    file_name="original_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to prepare original data for download.")
        except Exception as e:
            log_error("Error preparing original data download", e)
            st.error("Error preparing original data for download.")

    with col2:
        try:
            scaled_csv = convert_df_to_csv(X_scaled_df)
            if scaled_csv is not None:
                st.download_button(
                    label="Download Scaled Data as CSV",
                    data=scaled_csv,
                    file_name="scaled_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to prepare scaled data for download.")
        except Exception as e:
            log_error("Error preparing scaled data download", e)
            st.error("Error preparing scaled data for download.")

    st.divider()
    # EDA Section
    st.write("### Exploratory Data Analysis (EDA)")

    # Descriptive Statistics
    st.write("#### Descriptive Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Data**")
        try:
            st.dataframe(synthetic_data.describe())
        except Exception as e:
            log_error("Error generating descriptive statistics for original data", e)
            st.error("Failed to generate descriptive statistics for original data.")
    with col2:
        st.write("**Scaled Data**")
        try:
            st.dataframe(X_scaled_df.describe())
        except Exception as e:
            log_error("Error generating descriptive statistics for scaled data", e)
            st.error("Failed to generate descriptive statistics for scaled data.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            x_feature = st.selectbox("Select feature for X-axis", features, index=0)
        except Exception as e:
            log_error("Error creating X-axis feature selector", e)
            x_feature = features[0] if features else None
    with col2:
        try:
            y_feature = st.selectbox("Select feature for Y-axis", features, index=1 if len(features) > 1 else 0)
        except Exception as e:
            log_error("Error creating Y-axis feature selector", e)
            y_feature = features[1] if len(features) > 1 else features[0] if features else None

    # 2D Visualization
    st.write("#### 2D Visualization of Features")
    if x_feature and y_feature:
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            sns.scatterplot(
                data=synthetic_data,
                x=x_feature,
                y=y_feature,
                hue="Class",
                palette=["#3A85FF", "#71C9FF", "#FFA0A0", "#FF4040", "#66FF66"],
                alpha=0.7,
                s=20
            )

            plt.xlabel(x_feature, fontsize=12, labelpad=10)
            plt.ylabel(y_feature, fontsize=12, labelpad=10)
            plt.title(f"2D Visualization of {x_feature} vs {y_feature}", fontsize=14, pad=20)
            plt.legend(title="Class", loc="upper right", fontsize=10)

            st.pyplot(fig)
            plt.close()  # Close figure to prevent memory leaks
        except Exception as e:
            log_error("Error creating 2D visualization", e)
            st.error("Failed to create 2D visualization. Please check your data.")
    else:
        st.warning("Please ensure at least two features are available for visualization.")
    st.divider()

    # Distribution Plots
    st.write("#### Distribution of Features")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, feature in enumerate(features[:4]):  # Limit to first 4 features
            try:
                sns.histplot(synthetic_data, x=feature, hue="Class", kde=True, ax=axes[idx])
                axes[idx].set_title(f"Distribution of {feature}")
            except Exception as e:
                log_error(f"Error creating distribution plot for {feature}", e)
                axes[idx].text(0.5, 0.5, f"Error plotting\n{feature}", 
                             ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide unused subplots
        for idx in range(len(features), 4):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()  # Close figure to prevent memory leaks
    except Exception as e:
        log_error("Error creating distribution plots", e)
        st.error("Failed to create distribution plots.")
    st.divider()

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = synthetic_data[features].corr()
        
        # Check if correlation matrix is valid
        if correlation_matrix.empty or correlation_matrix.isnull().all().all():
            st.warning("Cannot compute correlation matrix - insufficient numeric data.")
        else:
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
        
        plt.close()  # Close figure to prevent memory leaks
    except Exception as e:
        log_error("Error creating correlation heatmap", e)
        st.error("Failed to create correlation heatmap.")
    st.divider()

else:
    st.write("Configure the settings in the sidebar and click 'Generate Data' to start.")
    
    # Show any configuration errors
    try:
        if len(classes) == 0 or len(features) == 0:
            st.info("Please enter valid class names and feature names in the sidebar to begin.")
        elif len(class_parameters) == 0:
            st.info("Please configure the parameters for your classes in the sidebar.")
    except Exception as e:
        log_error("Error in final display section", e)
