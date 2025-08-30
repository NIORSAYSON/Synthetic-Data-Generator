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
from sklearn.impute import SimpleImputer
import time
import pickle
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

def validate_dataframe(df, min_rows=10):
    """Validate uploaded DataFrame for training requirements"""
    try:
        if df is None:
            return False, "No data provided."
        
        if df.empty:
            return False, "The uploaded file is empty."
        
        if len(df) < min_rows:
            return False, f"Dataset too small. Minimum {min_rows} rows required, got {len(df)}."
        
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns (features + target)."
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 50:
            return False, f"Too many missing values ({missing_percentage:.1f}%). Please clean your data."
        
        return True, "Data validation passed."
    except Exception as e:
        log_error("Error validating DataFrame", e)
        return False, f"Validation error: {str(e)}"

def safe_label_encode(series, encoder=None):
    """Safely encode categorical data with error handling"""
    try:
        if encoder is None:
            encoder = LabelEncoder()
            
        # Handle missing values by filling with mode or 'unknown'
        if series.isnull().any():
            mode_value = series.mode()[0] if not series.mode().empty else 'unknown'
            series = series.fillna(mode_value)
        
        # Convert to string to handle mixed types
        series = series.astype(str)
        
        encoded = encoder.fit_transform(series)
        return encoded, encoder
    except Exception as e:
        log_error(f"Error encoding series: {series.name}", e)
        # Return numeric encoding as fallback
        unique_vals = series.unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return series.map(mapping).fillna(0).astype(int), None

def preprocess_features(X, feature_columns):
    """Preprocess features with comprehensive error handling"""
    try:
        # Handle missing values
        if X.isnull().any().any():
            st.warning("Missing values detected in features. Applying imputation...")
            imputer = SimpleImputer(strategy='mean')
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Impute numeric columns
            if len(numeric_columns) > 0:
                X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
            
            # Impute categorical columns
            if len(categorical_columns) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])
        
        # Handle infinite values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
            if X[numeric_columns].isnull().any().any():
                X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
        
        # Validate final result
        if X.isnull().any().any():
            st.error("Failed to handle all missing values in features.")
            return None
        
        return X
    except Exception as e:
        log_error("Error preprocessing features", e)
        st.error("Failed to preprocess features.")
        return None

st.set_page_config(
    page_title="Model Training",
    page_icon="🔄",
    layout="wide",
)

st.header("Model Training", divider="rainbow")

# Function to display classification report with error handling
def display_classification_report(y_true, y_pred, target_encoder=None):
    """Generate classification report with comprehensive error handling"""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            log_error("Empty predictions or true values")
            return pd.DataFrame()
        
        if len(y_true) != len(y_pred):
            log_error(f"Mismatched lengths: y_true={len(y_true)}, y_pred={len(y_pred)}")
            return pd.DataFrame()
        
        # Decode target classes if they were encoded
        if target_encoder:
            try:
                class_names = target_encoder.inverse_transform(np.unique(y_true))
            except Exception as e:
                log_error("Error decoding class names", e)
                class_names = [str(cls) for cls in np.unique(y_true)]
        else:
            class_names = [str(cls) for cls in np.unique(y_true)]
        
        report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        return report_df
    except Exception as e:
        log_error("Error generating classification report", e)
        st.error("Failed to generate classification report.")
        return pd.DataFrame()

def plot_model_performance(performance_df):
    """Visualize model performance metrics using grouped bar plots with error handling"""
    try:
        if performance_df.empty:
            st.warning("No performance data to plot.")
            return
        
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
        plt.close()
    except Exception as e:
        log_error("Error plotting model performance", e)
        st.error("Failed to create performance comparison plot.")

def plot_confusion_matrix(y_true, y_pred, model_name, target_encoder=None):
    """Return confusion matrix figure with error handling"""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            return None
        
        if target_encoder:
            try:
                labels = target_encoder.classes_
            except Exception:
                labels = np.unique(y_true)
        else:
            labels = np.unique(y_true)
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix for {model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        return fig
    except Exception as e:
        log_error(f"Error creating confusion matrix for {model_name}", e)
        return None

def plot_learning_curve(model, X, y, model_name):
    """Return learning curve figure with error handling"""
    try:
        if len(X) < 10:  # Minimum samples for learning curve
            return None
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=min(5, len(X) // 3), scoring='accuracy', n_jobs=1
        )
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
    except Exception as e:
        log_error(f"Error creating learning curve for {model_name}", e)
        return None

def safe_model_training(model, X_train, y_train, X_test, y_test, model_name):
    """Safely train model with comprehensive error handling"""
    try:
        # Validate inputs
        if X_train.empty or len(y_train) == 0:
            return None, "Empty training data"
        
        if len(X_train) != len(y_train):
            return None, "Mismatched training data lengths"
        
        # Check for data quality issues
        if X_train.isnull().any().any():
            return None, "Training data contains NaN values"
        
        if X_test.isnull().any().any():
            return None, "Test data contains NaN values"
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics with error handling
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            log_error(f"Error calculating metrics for {model_name}", e)
            return None, f"Error calculating metrics: {str(e)}"
        
        return {
            "model": model,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_time": training_time
        }, None
        
    except Exception as e:
        log_error(f"Error training {model_name}", e)
        return None, f"Training failed: {str(e)}"

# File upload with error handling
st.sidebar.subheader("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type="csv")
st.sidebar.divider()

if uploaded_file:
    try:
        # Attempt to read the CSV file
        try:
            data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            # Try different encoding
            try:
                data = pd.read_csv(uploaded_file, encoding='latin1')
                st.sidebar.warning("File read with latin1 encoding. Please verify data integrity.")
            except Exception as e:
                st.sidebar.error(f"Failed to read file with multiple encodings: {str(e)}")
                st.stop()
        except pd.errors.EmptyDataError:
            st.sidebar.error("The uploaded file is empty.")
            st.stop()
        except pd.errors.ParserError as e:
            st.sidebar.error(f"Error parsing CSV file: {str(e)}")
            st.stop()
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            st.stop()
        
        # Validate the uploaded data
        is_valid, validation_msg = validate_dataframe(data)
        if not is_valid:
            st.sidebar.error(validation_msg)
            st.stop()
        else:
            st.sidebar.success("✅ Data uploaded successfully!")
        
        st.write("### Uploaded Data Sample")
        st.dataframe(data.head())
        st.divider()
        
    except Exception as e:
        log_error("Unexpected error during file upload", e)
        st.sidebar.error("An unexpected error occurred while processing the file.")
        st.stop()
    
    # Sidebar options with validation
    st.sidebar.subheader("Split Settings")
    test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=50, value=20) / 100
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    # Select target column explicitly with validation
    if len(data.columns) == 0:
        st.sidebar.error("No columns found in the dataset.")
        st.stop()
    
    target_column = st.sidebar.selectbox("Select the target (class) column", data.columns)
    if not target_column:
        st.sidebar.error("Please select a target column.")
        st.stop()
    
    feature_columns = [col for col in data.columns if col != target_column]
    if len(feature_columns) == 0:
        st.sidebar.error("No feature columns available after selecting target.")
        st.stop()

    # Display data information
    st.write(f"**Target column:** {target_column}")
    st.write(f"**Feature columns:** {len(feature_columns)} features")
    st.write(f"**Classes in target:** {data[target_column].nunique()} unique values")

    try:
        # Create a copy for preprocessing to avoid modifying original
        data_processed = data.copy()
        
        # Encode categorical features and target with error handling
        categorical_columns = data_processed[feature_columns].select_dtypes(include=['object']).columns.tolist()
        label_encoders = {}

        if len(categorical_columns) > 0:
            st.info(f"Encoding {len(categorical_columns)} categorical feature(s): {', '.join(categorical_columns)}")
            
            for col in categorical_columns:
                try:
                    encoded_values, encoder = safe_label_encode(data_processed[col])
                    data_processed[col] = encoded_values
                    if encoder is not None:
                        label_encoders[col] = encoder
                except Exception as e:
                    log_error(f"Error encoding column {col}", e)
                    st.error(f"Failed to encode column '{col}'. Removing from features.")
                    feature_columns.remove(col)

        # Encode target column if it is categorical with error handling
        target_encoder = None
        if data_processed[target_column].dtype == 'object':
            try:
                # st.info("Encoding target column...")
                encoded_target, target_encoder = safe_label_encode(data_processed[target_column])
                data_processed[target_column] = encoded_target
            except Exception as e:
                log_error("Error encoding target column", e)
                st.error("Failed to encode target column.")
                st.stop()

        # Extract features and target
        X = data_processed[feature_columns]
        y = data_processed[target_column]

        # Validate target column for classification
        if y.nunique() > 20:
            st.sidebar.warning("Target has >20 unique values. This might be better suited for regression.")
        if y.nunique() == 1:
            st.sidebar.error("Target column has only one unique value. Cannot perform classification.")
            st.stop()

        # Preprocess features
        X = preprocess_features(X, feature_columns)
        if X is None:
            st.stop()

        st.sidebar.divider()
        
        # Split data with error handling
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            # If stratify fails, try without stratification
            st.warning("Stratified split failed, using random split instead.")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            except Exception as e:
                log_error("Error splitting data", e)
                st.error("Failed to split the data. Please check your dataset.")
                st.stop()

        st.write("### Data Split Information")
        st.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Validate split results
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("Data split resulted in empty train or test set.")
            st.stop()

    except Exception as e:
        log_error("Error in data preprocessing", e)
        st.error("An error occurred during data preprocessing. Please check your dataset.")
        st.stop()

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

        # Training Logic with comprehensive error handling
        if st.button("Train Models"):
            if not selected_models:
                st.error("Please select at least one model to train.")
            else:
                try:
                    with st.spinner("Training models, please wait..."):
                        st.session_state.performance = []  # Reset performance metrics
                        st.session_state.trained_models = {}
                        st.session_state.encoder_files = {}
                        st.session_state.confusion_matrices = {}
                        st.session_state.learning_curves = {}
                        
                        training_errors = []
                        
                        for model_name in selected_models:
                            try:
                                st.info(f"Training {model_name}...")
                                
                                # Initialize model with error handling
                                if model_name == "Random Forest":
                                    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
                                elif model_name == "Logistic Regression":
                                    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')
                                elif model_name == "Support Vector Machine":
                                    model = SVC(random_state=random_state, probability=True)
                                else:
                                    log_error(f"Unknown model name: {model_name}")
                                    training_errors.append(f"Unknown model: {model_name}")
                                    continue

                                # Train the model safely
                                result, error_msg = safe_model_training(
                                    model, X_train, y_train, X_test, y_test, model_name
                                )
                                
                                if result is None:
                                    training_errors.append(f"{model_name}: {error_msg}")
                                    continue
                                
                                # Extract results
                                trained_model = result["model"]
                                y_pred = result["y_pred"]
                                accuracy = result["accuracy"]
                                precision = result["precision"]
                                recall = result["recall"]
                                f1 = result["f1"]
                                training_time = result["training_time"]
                                
                                # Store performance metrics
                                st.session_state.performance.append({
                                    "Model": model_name, 
                                    "Accuracy": accuracy, 
                                    "Precision": precision,
                                    "Recall": recall,
                                    "F1-Score": f1,
                                    "Training Time": training_time
                                })

                                # Save model in Session State
                                st.session_state.trained_models[model_name] = trained_model

                                # Save target encoder if it exists
                                if target_encoder and model_name not in st.session_state.encoder_files:
                                    try:
                                        st.session_state.encoder_files[model_name] = pickle.dumps(target_encoder)
                                    except Exception as e:
                                        log_error(f"Error serializing target encoder for {model_name}", e)

                                # Store visualizations with error handling
                                try:
                                    cm_fig = plot_confusion_matrix(y_test, y_pred, model_name, target_encoder)
                                    if cm_fig is not None:
                                        st.session_state.confusion_matrices[model_name] = cm_fig
                                except Exception as e:
                                    log_error(f"Error creating confusion matrix for {model_name}", e)

                                try:
                                    lc_fig = plot_learning_curve(trained_model, X, y, model_name)
                                    if lc_fig is not None:
                                        st.session_state.learning_curves[model_name] = lc_fig
                                except Exception as e:
                                    log_error(f"Error creating learning curve for {model_name}", e)

                                # Update best model
                                if accuracy > st.session_state.best_accuracy:
                                    st.session_state.best_model = model_name
                                    st.session_state.best_accuracy = accuracy
                                    try:
                                        st.session_state.best_report = display_classification_report(y_test, y_pred, target_encoder)
                                    except Exception as e:
                                        log_error(f"Error creating classification report for {model_name}", e)
                                
                                st.success(f"✅ {model_name} trained successfully!")
                                
                            except Exception as e:
                                error_msg = f"Failed to train {model_name}: {str(e)}"
                                log_error(error_msg, e)
                                training_errors.append(error_msg)
                                continue

                        # Display any training errors
                        if training_errors:
                            st.error("Some models failed to train:")
                            for error in training_errors:
                                st.error(f"❌ {error}")

                        # Check if any models were successfully trained
                        if len(st.session_state.performance) == 0:
                            st.error("No models were successfully trained. Please check your data and try again.")
                        else:
                            st.session_state.visualizations_ready = True  # Mark visualizations as ready
                            st.success(f"✅ Successfully trained {len(st.session_state.performance)} model(s)!")

                except Exception as e:
                    log_error("Critical error in training process", e)
                    st.error("A critical error occurred during training. Please check your data and try again.")
                    st.error(f"Error details: {str(e)}")

        # Display Results if Visualizations Are Ready
        if st.session_state.visualizations_ready and len(st.session_state.performance) > 0:
            try:
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
                
                # Plot performance comparison with error handling
                try:
                    plot_model_performance(performance_df)
                except Exception as e:
                    log_error("Error plotting model performance comparison", e)
                    st.error("Failed to create performance comparison plot.")

                st.divider()
                
                # Display best model information
                if hasattr(st.session_state, 'best_model') and st.session_state.best_model:
                    st.write("### Best Model Performance")
                    st.write(f"Best Model: **{st.session_state.best_model}**")
                    st.write(f"Accuracy: **{st.session_state.best_accuracy:.4f}**")
                    
                    if hasattr(st.session_state, 'best_report') and not st.session_state.best_report.empty:
                        st.write(f"Classification Report of {st.session_state.best_model}:")
                        st.dataframe(st.session_state.best_report)
                    else:
                        st.warning("Classification report not available for best model.")

                # Display learning curves with error handling
                if st.session_state.learning_curves:
                    st.divider()
                    st.write("### Learning Curves")
                    lc_cols = st.columns(min(3, len(st.session_state.learning_curves)))
                    for idx, (model_name, fig) in enumerate(st.session_state.learning_curves.items()):
                        try:
                            with lc_cols[idx % len(lc_cols)]:
                                st.pyplot(fig)
                                plt.close(fig)  # Close to prevent memory leaks
                        except Exception as e:
                            log_error(f"Error displaying learning curve for {model_name}", e)

                # Display confusion matrices with error handling
                if st.session_state.confusion_matrices:
                    st.divider()
                    st.write("### Confusion Matrices")
                    cm_cols = st.columns(min(3, len(st.session_state.confusion_matrices)))
                    for idx, (model_name, fig) in enumerate(st.session_state.confusion_matrices.items()):
                        try:
                            with cm_cols[idx % len(cm_cols)]:
                                st.pyplot(fig)
                                plt.close(fig)  # Close to prevent memory leaks
                        except Exception as e:
                            log_error(f"Error displaying confusion matrix for {model_name}", e)

                # Download section with error handling
                st.divider()
                st.write("### Download Models and Target Encoders")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Models:**")
                    for model_name, model in st.session_state.trained_models.items():
                        try:
                            model_file = pickle.dumps(model)
                            st.download_button(
                                label=f"Download {model_name} Model",
                                data=model_file,
                                file_name=f"model_{model_name.replace(' ', '_').lower()}.pkl",
                                mime="application/octet-stream",
                                key=f"download_model_{model_name}"
                            )
                        except Exception as e:
                            log_error(f"Error preparing {model_name} for download", e)
                            st.error(f"Failed to prepare {model_name} for download.")

                with col2:
                    st.write("**Target Encoders:**")
                    if st.session_state.encoder_files:
                        encoder_downloaded = False
                        for model_name, encoder_file in st.session_state.encoder_files.items():
                            try:
                                if not encoder_downloaded:  # Only show one encoder download
                                    st.download_button(
                                        label="Download Target Encoder",
                                        data=encoder_file,
                                        file_name="target_encoder.pkl",
                                        mime="application/octet-stream",
                                        key="download_encoder"
                                    )
                                    encoder_downloaded = True
                                    break
                            except Exception as e:
                                log_error(f"Error preparing target encoder for download", e)
                                st.error("Failed to prepare target encoder for download.")
                    else:
                        st.info("No target encoder available (numeric target)")

                st.divider()
                
            except Exception as e:
                log_error("Error displaying training results", e)
                st.error("An error occurred while displaying training results.")

else:
    st.write("Please upload a dataset to begin.")
    st.info("💡 **Tips for successful training:**")
    st.write("- Ensure your CSV file has a header row")
    st.write("- Make sure your target column is clearly defined")
    st.write("- Remove or handle missing values before upload")
    st.write("- Ensure sufficient data samples (minimum 10 per class)")
