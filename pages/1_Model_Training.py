import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time
import pickle

st.set_page_config(
    page_title="Model Training",
    page_icon="🔄",
    layout="wide",
)

st.header("Model Training", divider="rainbow")

# Function to display classification report
def display_classification_report(y_true, y_pred, target_encoder=None):
    # Decode target classes if they were encoded
    if target_encoder:
        class_names = target_encoder.inverse_transform(np.unique(y_true))
    else:
        class_names = [str(cls) for cls in np.unique(y_true)]
    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def plot_model_performance(performance_df):
    """ Visualize model performance metrics using grouped bar plots """
    # Melt DataFrame to long-form for grouped bar plot
    performance_melted = performance_df.melt(
        id_vars=["Model", "Training Time"], 
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric", 
        value_name="Score"
    )
    
    # Plot grouped barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=performance_melted, palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Models")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    st.pyplot(plt)


def plot_confusion_matrix(y_true, y_pred, model_name, target_encoder=None):
    """ Return confusion matrix figure """
    if target_encoder:
        labels = target_encoder.classes_
    else:
        labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix for {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return fig

def plot_learning_curve(model, X, y, model_name):
    """ Return learning curve figure """
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_sizes, train_mean, label='Training Score', color='blue')
    ax.plot(train_sizes, test_mean, label='Cross-Validation Score', color='orange')
    ax.set_title(f"Learning Curve for {model_name}")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return fig

# File upload
st.sidebar.subheader("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type="csv")
st.sidebar.divider()

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Sample")
    st.dataframe(data.head())
    st.divider()
    
    # Sidebar options
    st.sidebar.subheader("Split Settings")
    test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=50, value=20) / 100
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    # Select target column explicitly
    target_column = st.sidebar.selectbox("Select the target (class) column", data.columns)
    feature_columns = [col for col in data.columns if col != target_column]

    # Encode categorical features and target
    categorical_columns = data[feature_columns].select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Encode target column if it is categorical
    if data[target_column].dtype == 'object':
        target_encoder = LabelEncoder()
        data[target_column] = target_encoder.fit_transform(data[target_column])
        
    else:
        target_encoder = None

    X = data[feature_columns]
    y = data[target_column]

    # Validate target column for classification
    if y.nunique() > 20 and y.dtype != 'object':
        st.sidebar.error("The selected target column seems to be continuous. Please ensure the target is categorical for classification.")
        st.stop()

    st.sidebar.divider()
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.write("### Data Split Information")
    st.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    st.divider()
    st.subheader("Choose Models")
    col1, col2 = st.columns([3, 1])  # Split into two equal columns

    with col1:  # Place multiselect in the first column (50% width)
        selected_models = st.multiselect(
            "Select models to train",
            ["Random Forest", "Logistic Regression", "Support Vector Machine"],
            default=["Random Forest", "Logistic Regression", "Support Vector Machine"],
        )

    # Show Train Models button
    if selected_models:  # Ensure models are selected
       
        # Initialize Session State for persistence
        if "trained_models" not in st.session_state:
            st.session_state.trained_models = {}
        if "encoder_files" not in st.session_state:
            st.session_state.encoder_files = {}
        if "performance" not in st.session_state:
            st.session_state.performance = []
        if "confusion_matrices" not in st.session_state:
            st.session_state.confusion_matrices = {}
        if "learning_curves" not in st.session_state:
            st.session_state.learning_curves = {}
        if "best_model" not in st.session_state:
            st.session_state.best_model = None
        if "best_report" not in st.session_state:
            st.session_state.best_report = None
        if "best_accuracy" not in st.session_state:
            st.session_state.best_accuracy = 0
        if "visualizations_ready" not in st.session_state:
            st.session_state.visualizations_ready = False  # Tracks if visualizations are ready

        # Training Logic
        if st.button("Train Models"):
            with st.spinner("Training models, please wait..."):
                st.session_state.performance = []  # Reset performance metrics
                for model_name in selected_models:
                    start_time = time.time()
                    if model_name == "Random Forest":
                        model = RandomForestClassifier(random_state=random_state)
                    elif model_name == "Logistic Regression":
                        model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif model_name == "Support Vector Machine":
                        model = SVC(random_state=random_state)

                    # Train the model
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    st.session_state.performance.append({
                        "Model": model_name, 
                        "Accuracy": accuracy, 
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1,
                        "Training Time": training_time
                    })

                    # Save model in Session State
                    st.session_state.trained_models[model_name] = model

                    # Save target encoder if it exists
                    if target_encoder and model_name not in st.session_state.encoder_files:
                        st.session_state.encoder_files[model_name] = pickle.dumps(target_encoder)

                    # Store confusion matrix and learning curve
                    st.session_state.confusion_matrices[model_name] = plot_confusion_matrix(y_test, y_pred, model_name, target_encoder)
                    st.session_state.learning_curves[model_name] = plot_learning_curve(model, X, y, model_name)

                    # Save best model
                    if accuracy > st.session_state.best_accuracy:
                        st.session_state.best_model = model_name
                        st.session_state.best_accuracy = accuracy
                        st.session_state.best_report = display_classification_report(y_test, y_pred, target_encoder)

                st.session_state.visualizations_ready = True  # Mark visualizations as ready

        # Display Results if Visualizations Are Ready
        if st.session_state.visualizations_ready:
            st.write("### Training Results")
            st.divider()
            # Display model comparison
            st.write("### Model Comparison")
            performance_df = pd.DataFrame(st.session_state.performance)
            st.table(performance_df.style.format({
                "Accuracy": "{:.4f}", 
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Training Time": "{:.4f}"
            }))
            plot_model_performance(performance_df)

            
            st.divider()
            st.write("### Best Model Performance")
            st.write(f"Best Model: **{st.session_state.best_model}**")
            st.write(f"Accuracy: **{st.session_state.best_accuracy:.4f}**")
            st.write(f"Classification Report of {st.session_state.best_model}:")
            st.dataframe(st.session_state.best_report)

            # Display learning curves
            st.divider()
            st.write("### Learning Curves")
            lc_cols = st.columns(3)
            for idx, (model_name, fig) in enumerate(st.session_state.learning_curves.items()):
                with lc_cols[idx % 3]:
                    st.pyplot(fig)

            # Display confusion matrices
            st.divider()
            st.write("### Confusion Matrices")
            cm_cols = st.columns(3)
            for idx, (model_name, fig) in enumerate(st.session_state.confusion_matrices.items()):
                with cm_cols[idx % 3]:
                    st.pyplot(fig)


            # Add download buttons for models and scalers
            st.divider()
            st.write("### Download Models and Target Encoders")
            col1, col2 = st.columns(2)

            with col1:
                for model_name, model in st.session_state.trained_models.items():
                    model_file = pickle.dumps(model)
                    st.download_button(
                        label=f"Download {model_name} Model",
                        data=model_file,
                        file_name=f"model_{model_name.replace(' ', '_').lower()}.pkl",
                        mime="application/octet-stream"
                    )
            with col2:
                if st.session_state.encoder_files:
                    for model_name, encoder_file in st.session_state.encoder_files.items():
                        st.download_button(
                            label=f"Download Target Encoder ({model_name})",
                            data=encoder_file,
                            file_name="target_encoder.pkl",
                            mime="application/octet-stream"
                        )
            st.divider()
else:
    st.write("Please upload a dataset to begin.")
