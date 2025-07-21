import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Import models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Import metrics
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Unified AutoML Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius:10px;
        border: 2px solid #ff4b4b;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def load_data(file):
    """Loads data from an uploaded file."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Automated data cleaning pipeline."""
    cleaned_df = df.copy()
    
    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            # Impute numerical columns with median
            if cleaned_df[col].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy='median')
                cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]])
            # Impute categorical columns with most frequent
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]]).flatten()

    # Encode categorical features
    label_encoders = {}
    for col in cleaned_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        cleaned_df[col] = le.fit_transform(cleaned_df[col])
        label_encoders[col] = le
        
    return cleaned_df, label_encoders

def get_problem_type(df, target_column):
    """Automatically determine if it's a classification or regression problem."""
    if df[target_column].nunique() / len(df) < 0.1 and df[target_column].dtype != 'float64':
        # If the number of unique values is small relative to the dataset size, treat as classification
        return "Classification"
    else:
        # Otherwise, treat as regression
        return "Regression"

# --- Main Application ---
st.title("🤖 Unified AutoML Intelligence Platform")
st.write("Upload your dataset, and this app will automatically clean the data, identify the problem type, train multiple models, and select the best one.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)

if df is not None:
    st.header("2. Data Preview & Column Selection")
    st.dataframe(df.head())
    
    st.sidebar.header("2. Select Target Column")
    target_column = st.sidebar.selectbox("Which column do you want to predict?", df.columns)

    if target_column:
        st.sidebar.header("3. Start Analysis")
        if st.sidebar.button("🚀 Analyze and Find Best Model"):
            
            # --- Module 1: Automated Data Cleaning & Prep ---
            st.header("⚙️ Automated Data Processing")
            with st.spinner("Cleaning and preparing data..."):
                cleaned_df, encoders = clean_data(df.drop(columns=[target_column], errors='ignore'))
                target_series, _ = clean_data(pd.DataFrame(df[target_column]))
                
                X = cleaned_df
                y = target_series.iloc[:, 0]
                
                # Feature Scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            st.success("Data cleaning and preparation complete!")
            st.write("Handled missing values, encoded categorical features, and scaled numerical data.")

            # --- Module 2: Auto-Select Algorithm ---
            problem_type = get_problem_type(df, target_column)
            st.header(f"🧠 Problem Identified: **{problem_type}**")
            
            models = {}
            if problem_type == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Support Vector Machine": SVC()
                }
            else: # Regression
                models = {
                    "Linear Regression": LinearRegression(),
                    "K-Nearest Neighbors": KNeighborsRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Support Vector Machine": SVR()
                }

            # --- Module 3: Model Training & Evaluation ---
            st.header("🏋️ Training and Evaluating Models")
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    results[name] = score
                else:
                    score = r2_score(y_test, y_pred)
                    results[name] = score
                
                progress_bar.progress((i + 1) / len(models))
            
            status_text.text("All models trained successfully!")
            st.success("Model training and evaluation complete.")

            # --- Module 4: Display Results ---
            st.header("🏆 Results: Best Model")
            
            # Find the best model
            if problem_type == "Classification":
                best_model_name = max(results, key=results.get)
                best_score = results[best_model_name]
                metric_name = "Accuracy"
            else:
                best_model_name = max(results, key=results.get)
                best_score = results[best_model_name]
                metric_name = "R-squared Score"

            st.subheader(f"🥇 Best Performing Model: **{best_model_name}**")
            st.metric(label=metric_name, value=f"{best_score:.4f}")
            
            st.subheader("Model Performance Comparison")
            results_df = pd.DataFrame(list(results.items()), columns=['Model', metric_name]).sort_values(by=metric_name, ascending=False)
            st.dataframe(results_df)

            st.balloons()

else:
    st.info("Awaiting for CSV file to be uploaded.")
