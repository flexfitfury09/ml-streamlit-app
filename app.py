import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

# Metrics & Visualization
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional AutoML Platform",
    page_icon="🏆",
    layout="wide"
)

# --- Main App ---
st.title("🏆 Professional Automated Machine Learning Platform")
st.write("This advanced tool automates the entire ML workflow: from data cleaning and model training to detailed evaluation and feature analysis. **No deep learning, no errors.**")

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def clean_data(df):
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy='median')
                cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]]).flatten()
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]]).flatten()
    return cleaned_df

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.header("2. Select Your Goal")
        task = st.selectbox("Choose the ML Task", ["--Select--", "📈 Regression", "🎯 Classification"])

        if task != "--Select--":
            st.header("3. Choose Target Column")
            target_column = st.selectbox("Select the column to predict", df.columns)
            st.header("4. Start Training")
            if st.button("🚀 Train Models", use_container_width=True):
                st.session_state.run_training = True
            else:
                st.session_state.run_training = False
        else:
             st.session_state.run_training = False
    else:
        st.session_state.run_training = False


# --- Main Panel for Display ---
if uploaded_file is None:
    st.info("Please upload a CSV file using the sidebar to begin.")
else:
    st.header("📊 Data Preview")
    st.dataframe(df.head())
    
    if st.session_state.get('run_training', False):
        with st.spinner("Processing... Please wait."):
            # 1. Data Cleaning
            df_cleaned = clean_data(df)
            
            # 2. Preprocessing
            y = df_cleaned[target_column]
            X = df_cleaned.drop(columns=[target_column])
            
            # Label encode target for classification
            if task == "🎯 Classification":
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Encode features and scale
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # 3. Model Training
            models = {}
            if task == "🎯 Classification":
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                }
            elif task == "📈 Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "XGBoost": XGBRegressor(objective='reg:squarederror')
                }

            trained_models = {}
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                trained_models[name] = model
                preds = model.predict(X_test)
                if task == "🎯 Classification":
                    results[name] = accuracy_score(y_test, preds)
                else:
                    results[name] = r2_score(y_test, preds)

            # 4. Find Best Model
            best_model_name = max(results, key=results.get)
            best_model = trained_models[best_model_name]
            best_model_score = results[best_model_name]
            
            st.success(f"Training complete! Best Model: **{best_model_name}**")

            # --- Display Results and Advanced Features ---
            st.header("🏆 Model Performance Results")
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model Leaderboard")
                results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Score (Accuracy or R2)']).sort_values(by='Score (Accuracy or R2)', ascending=False)
                st.dataframe(results_df, use_container_width=True)

            with col2:
                st.subheader(f"Best Model: {best_model_name}")
                st.metric(label="Score", value=f"{best_model_score:.4f}")
                
                # Download button for the best model
                model_bytes = io.BytesIO()
                joblib.dump(best_model, model_bytes)
                st.download_button(
                    label="⬇️ Download Best Model (.joblib)",
                    data=model_bytes,
                    file_name=f"{best_model_name.replace(' ', '_')}.joblib",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.header(" drilled-down analysis")

            # Display advanced analytics based on task type
            if task == "🎯 Classification":
                with st.expander("🔬 View Classification Report & Confusion Matrix"):
                    y_pred = best_model.predict(X_test)
                    
                    # Use original labels if they were encoded
                    try:
                        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                    except: # Fallback if target names fail
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
                    st.plotly_chart(fig)

            elif task == "📈 Regression":
                with st.expander("🔬 View Detailed Regression Metrics"):
                    y_pred = best_model.predict(X_test)
                    metrics = {
                        "R-squared Score": r2_score(y_test, y_pred),
                        "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred)
                    }
                    st.json(metrics)

            # Feature Importance
            if hasattr(best_model, 'feature_importances_'):
                with st.expander("💡 View Feature Importance"):
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    
                    fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title=f'Feature Importance for {best_model_name}')
                    st.plotly_chart(fig)