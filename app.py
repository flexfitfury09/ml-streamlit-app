import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoML Co-Pilot",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    """Loads data and returns a pandas DataFrame."""
    return pd.read_csv(file)

# --- Session State Initialization ---
def init_session_state():
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    if 'results' not in st.session_state:
        st.session_state.results = {}

init_session_state()

# --- UI Layout ---
st.title("🤖 AutoML Co-Pilot")
st.write("Your intelligent partner for automated machine learning. From data to deployment-ready models with deep-dive explainability.")

with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.header("2. Analysis Configuration")
        target_column = st.selectbox("Select Target Column", df.columns)
        
        with st.expander("Advanced Options"):
            use_feature_selection = st.toggle("Automated Feature Selection", True)
            use_hyperparameter_tuning = st.toggle("Hyperparameter Tuning (Slower)", False)

        if st.button("🚀 Launch Co-Pilot", use_container_width=True, type="primary"):
            st.session_state.run_analysis = True
            st.session_state.config = {
                "target_column": target_column,
                "use_feature_selection": use_feature_selection,
                "use_hyperparameter_tuning": use_hyperparameter_tuning
            }

# --- Main Application Logic ---
if not st.session_state.run_analysis and not uploaded_file:
    st.info("Upload a dataset and configure your analysis in the sidebar to get started.")

if st.session_state.run_analysis and uploaded_file:
    df = load_data(uploaded_file)
    config = st.session_state.config
    target_column = config['target_column']
    
    with st.spinner("Co-Pilot is at the helm... Performing comprehensive analysis. Please wait."):
        # 1. Intelligent Problem Detection
        if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 20:
            task = "📈 Regression"
        else:
            task = "🎯 Classification"
        
        # 2. Preprocessing Pipeline
        y = df[target_column]
        X = df.drop(columns=[target_column])
        X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True) # Sanitize column names
        
        if task == "🎯 Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.results['label_encoder'] = le
        
        # Data Cleaning & Encoding
        numeric_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(exclude=np.number).columns
        
        for col in numeric_cols: X[col] = SimpleImputer(strategy='median').fit_transform(X[[col]]).flatten()
        for col in categorical_cols: X[col] = LabelEncoder().fit_transform(SimpleImputer(strategy='most_frequent').fit_transform(X[[col]]).flatten())

        # 3. Automated Feature Selection (Optional)
        if config['use_feature_selection']:
            selector_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) if task == "📈 Regression" else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = SelectFromModel(selector_model, threshold='median')
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features)
        
        # Scaling and Splitting
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        st.session_state.results['scaler'] = scaler
        st.session_state.results['features'] = X.columns

        # 4. Model Training & Tuning
        model_to_tune = RandomForestRegressor(random_state=42) if task == "📈 Regression" else RandomForestClassifier(random_state=42)
        
        if config['use_hyperparameter_tuning']:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
            grid_search = GridSearchCV(model_to_tune, param_grid, cv=3, n_jobs=-1, scoring='r2' if task == "📈 Regression" else 'accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            st.session_state.results['best_params'] = grid_search.best_params_
        else:
            best_model = model_to_tune.fit(X_train, y_train)
        
        # 5. Evaluation and Prediction
        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred) if task == "📈 Regression" else accuracy_score(y_test, y_pred)
        
        # 6. SHAP Explainability
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        
        # Storing all artifacts in session state
        st.session_state.results.update({
            'task': task, 'best_model': best_model, 'score': score, 'y_test': y_test, 'y_pred': y_pred,
            'X_test_df': pd.DataFrame(X_test, columns=X.columns), 'explainer': explainer, 'shap_values': shap_values,
            'cleaned_data': pd.concat([X, pd.Series(y, name=target_column)], axis=1)
        })

    st.success("✅ Co-Pilot analysis complete!")

# --- Display Results ---
if st.session_state.run_analysis:
    res = st.session_state.results
    
    # Run Summary
    st.header("📈 Run Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Problem Type Detected", res['task'])
    c2.metric("Features Selected", f"{len(res['features'])}/{len(df.drop(columns=[st.session_state.config['target_column']]).columns)}")
    c3.metric(f"Best Model Score ({'R²' if res['task'] == '📈 Regression' else 'Accuracy'})", f"{res['score']:.4f}")
    
    if 'best_params' in res:
        st.info(f"Optimal hyperparameters found: `{res['best_params']}`")

    # --- Result Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Model Performance", "🧠 Global Explanation", "🔮 Local Prediction Explorer", "💡 What-If Simulator", "📦 Assets"])

    with tab1:
        if res['task'] == '🎯 Classification':
            st.subheader("Classification Report")
            report_df = pd.DataFrame(classification_report(res['y_test'], res['y_pred'], target_names=res['label_encoder'].classes_, output_dict=True)).transpose()
            st.dataframe(report_df)
            
            st.subheader("Confusion Matrix")
            cm_fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(res['y_test'], res['y_pred']), annot=True, fmt='d', ax=ax, cmap='Blues')
            st.pyplot(cm_fig)
        else:
            st.subheader("Regression Metrics")
            st.json({
                "R-squared": r2_score(res['y_test'], res['y_pred']),
                "Mean Absolute Error": np.mean(np.abs(res['y_pred'] - res['y_test'])),
                "Root Mean Squared Error": np.sqrt(np.mean((res['y_pred'] - res['y_test'])**2))
            })

    with tab2:
        st.subheader("SHAP Summary Plot")
        st.write("This plot shows the most significant features and their overall impact on the model's predictions.")
        summary_fig, ax = plt.subplots()
        shap.summary_plot(res['shap_values'], res['X_test_df'], plot_type="bar", show=False)
        st.pyplot(summary_fig)

    with tab3:
        st.subheader("Single Prediction Deconstruction")
        row_index = st.selectbox("Select a row from the test set to explain:", res['X_test_df'].index)
        st.write("The force plot below shows features pushing the prediction higher (red) or lower (blue).")
        st.dataframe(res['X_test_df'].iloc[[row_index]])
        shap.force_plot(res['explainer'].expected_value, res['shap_values'][row_index], res['X_test_df'].iloc[row_index], matplotlib=True, show=False)
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf()

    with tab4:
        st.subheader("Manual Prediction Simulator")
        input_data = {}
        for col in res['features']:
            input_data[col] = st.number_input(f"Enter value for {col}", value=float(res['X_test_df'][col].mean()))
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = res['scaler'].transform(input_df)
            prediction = res['best_model'].predict(input_scaled)
            if res['task'] == "🎯 Classification":
                prediction = res['label_encoder'].inverse_transform(prediction)
            st.success(f"**Predicted Output:** {prediction[0]}")

    with tab5:
        st.subheader("Downloadable Assets")
        st.download_button("⬇️ Download Trained Model (.joblib)", data=joblib.dump(res['best_model'], 'model.joblib'), file_name="best_model.joblib")
        st.download_button("⬇️ Download Cleaned Data (.csv)", data=res['cleaned_data'].to_csv(index=False).encode('utf-8'), file_name="cleaned_data.csv")
        
        predictions_df = res['X_test_df'].copy()
        predictions_df['prediction'] = res['y_pred']
        if res['task'] == "🎯 Classification":
            predictions_df['prediction'] = res['label_encoder'].inverse_transform(predictions_df['prediction'])
        st.download_button("⬇️ Download Predictions on Test Set (.csv)", data=predictions_df.to_csv(index=False).encode('utf-8'), file_name="test_predictions.csv")
