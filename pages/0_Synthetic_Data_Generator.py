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

LOGGER = get_logger(__name__)

st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🗂️",
    layout="wide",
)

# Sidebar settings
st.sidebar.title("Configuration")

# Input for class and feature names
st.sidebar.subheader("Classes & Features")
class_names = st.sidebar.text_input("Enter class names (comma-separated)", "Male, Female")
feature_names = st.sidebar.text_input("Enter feature names (comma-separated)", "height (cm), weight (kg), shoe size (cm), pitch frequency (Hz)")

# Parse class names and feature names
classes = [cls.strip() for cls in class_names.split(",")]
features = [feat.strip() for feat in feature_names.split(",")]

# Class-specific parameter settings
st.sidebar.subheader("Class-Specific Parameters")
class_parameters = {}

# Default feature-specific mean and std dev values
default_parameters = {
    "Male": {
        "height (cm)_mean": 175,
        "height (cm)_std": 7.5,
        "weight (kg)_mean": 78,
        "weight (kg)_std": 10,
        "shoe size (cm)_mean": 27,
        "shoe size (cm)_std": 1.5,
        "pitch frequency (Hz)_mean": 120,
        "pitch frequency (Hz)_std": 25,
    },
    "Female": {
        "height (cm)_mean": 162,
        "height (cm)_std": 6.5,
        "weight (kg)_mean": 65,
        "weight (kg)_std": 8,
        "shoe size (cm)_mean": 24,
        "shoe size (cm)_std": 1,
        "pitch frequency (Hz)_mean": 220,
        "pitch frequency (Hz)_std": 20,
    },
}

for cls in classes:
    with st.sidebar.expander(f"{cls} Settings"):
        enable_settings = st.checkbox(f"Enable specific settings for {cls}", key=f"{cls}_enabled", value=True)
        if enable_settings:
            parameters = {}
            for feature in features:
                feature_key = feature.lower().replace(" ", "_").replace("(", "").replace(")", "")
                mean_default = default_parameters.get(cls, {}).get(f"{feature}_mean", random.uniform(50, 150))
                std_default = default_parameters.get(cls, {}).get(f"{feature}_std", random.uniform(5, 15))
                parameters[f"{feature_key}_mean"] = st.number_input(
                    f"Mean {feature} for {cls}",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(mean_default),
                    key=f"{cls}_{feature_key}_mean",
                )
                parameters[f"{feature_key}_std"] = st.number_input(
                    f"Std Dev {feature} for {cls}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(std_default),
                    key=f"{cls}_{feature_key}_std",
                )
                st.divider()
            class_parameters[cls] = parameters

# Sample size and train/test split
st.sidebar.subheader("Data Settings")
num_samples = st.sidebar.slider("Number of samples", min_value=500, max_value=50000, value=10000)
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
    data = []
    # Loop through each class
    for cls in classes:
        for _ in range(num_samples // len(classes)):
            sample = {"Class": cls}  # Initialize a sample with the class label
            # Loop through each feature
            for feature in features:
                feature_key = feature.lower().replace(" ", "_").replace("(", "").replace(")", "")
                # Get the mean and std dev for this class and feature
                mean = class_parameters[cls][f"{feature_key}_mean"]
                std = class_parameters[cls][f"{feature_key}_std"]
                # Generate data using normal distribution
                sample[feature] = np.random.normal(mean, std)
            data.append(sample)
    # Shuffle the data to ensure randomness
    random.shuffle(data)
    # Convert to a Pandas DataFrame
    return pd.DataFrame(data)

# Function to convert a DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Main display
st.header("Synthetic Data Generator", divider='rainbow')

# Generate data button
if st.sidebar.button("Generate Data"):
    st.session_state["synthetic_data"] = generate_data()
    
synthetic_data = st.session_state["synthetic_data"]

if synthetic_data is not None:
    st.write("### Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"###### Number of Classes: {len(classes)}")
      
    with col2:
        st.write(f"###### Number of Samples: {num_samples}")
       
    X = synthetic_data[features]
    y = synthetic_data["Class"]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    X_scaled_df["Class"] = y.values
    combined_data = pd.concat([synthetic_data.reset_index(drop=True), X_scaled_df], axis=1)
    st.divider()

    st.write("### Generated Data Sample")
    col1, col2 = st.columns(2)
    with col1:
        st.write("###### Original Data")
        st.dataframe(combined_data.iloc[:, :len(features) + 1])
    with col2:
        st.write("###### Scaled Data")
        st.dataframe(combined_data.iloc[:, len(features) + 1:])
    
    # Download buttons for original and scaled data
    st.write("### Download Generated Data")

    col1, col2 = st.columns(2)
    with col1:
        original_csv = convert_df_to_csv(synthetic_data)
        st.download_button(
            label="Download Original Data as CSV",
            data=original_csv,
            file_name="original_data.csv",
            mime="text/csv",
        )

    with col2:
        scaled_csv = convert_df_to_csv(X_scaled_df)
        st.download_button(
            label="Download Scaled Data as CSV",
            data=scaled_csv,
            file_name="scaled_data.csv",
            mime="text/csv",
        )

    st.divider()
    # EDA Section
    st.write("### Exploratory Data Analysis (EDA)")

    # Descriptive Statistics
    st.write("#### Descriptive Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Data**")
        st.dataframe(synthetic_data.describe())
    with col2:
        st.write("**Scaled Data**")
        st.dataframe(X_scaled_df.describe())
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select feature for X-axis", features, index=0)
    with col2:
        y_feature = st.selectbox("Select feature for Y-axis", features, index=1)

    # 2D Visualization
    st.write("#### 2D Visualization of Features")
    if x_feature and y_feature:
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
    st.divider()

    # Distribution Plots
    st.write("#### Distribution of Features")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, feature in enumerate(features):
        sns.histplot(synthetic_data, x=feature, hue="Class", kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {feature}")
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_matrix = synthetic_data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.divider()

else:
    st.write("Configure the settings in the sidebar and click 'Generate Data' to start.")
