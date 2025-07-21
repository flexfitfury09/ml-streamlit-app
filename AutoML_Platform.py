import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score
import plotly.express as px

st.set_page_config(page_title="AutoML Platform", layout="wide")

st.title("🤖 Automated Machine Learning for Tabular Data")
st.write("This module handles both Supervised and Unsupervised learning on your CSV data.")

# Helper Functions
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def clean_data(df, target_column=None):
    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]])
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                X[col] = imputer.fit_transform(X[[col]]).ravel()
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    return X_scaled, y, X_encoded.columns

uploaded_file = st.sidebar.file_uploader("Upload Your CSV Here", type="csv", key="automl_uploader")

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.header("Choose Your Goal")
    task_type = st.sidebar.selectbox("What do you want to do?", ["--Select--", "📈 Predict a Target (Supervised Learning)", "🧩 Find Patterns (Unsupervised Learning)"])

    st.header("Data Preview")
    st.dataframe(df.head())

    if task_type == "📈 Predict a Target (Supervised Learning)":
        target_column = st.sidebar.selectbox("Select Target Column", df.columns)
        if st.sidebar.button("🚀 Run Supervised AutoML"):
            X_scaled, y, _ = clean_data(df, target_column)
            if df[target_column].dtype == 'object': y = LabelEncoder().fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            problem_type = "Regression" if df[target_column].nunique() > 20 else "Classification"
            st.info(f"**Problem Identified:** {problem_type}")
            models = {}
            if problem_type == "Classification":
                models = {"Logistic Regression": LogisticRegression(), "K-Nearest Neighbors": KNeighborsClassifier(), "Support Vector Machine": SVC(), "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier(), "Naive Bayes": GaussianNB()}
            else:
                models = {"Linear Regression": LinearRegression(), "K-Nearest Neighbors": KNeighborsRegressor(), "Support Vector Machine": SVR(), "Decision Tree": DecisionTreeRegressor(), "Random Forest": RandomForestRegressor()}
            results = {}
            with st.spinner("Training all models..."):
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    if problem_type == "Classification": results[name] = accuracy_score(y_test, preds)
                    else: results[name] = r2_score(y_test, preds)
            best_model_name = max(results, key=results.get)
            metric_name = "Accuracy" if problem_type == "Classification" else "R2 Score"
            results_df = pd.DataFrame(list(results.items()), columns=['Model', metric_name]).sort_values(by=metric_name, ascending=False).reset_index(drop=True)
            st.header("🏆 Supervised Learning Results")
            st.metric(f"Best Model: {best_model_name}", f"{results[best_model_name]:.4f}")
            st.dataframe(results_df)

    elif task_type == "🧩 Find Patterns (Unsupervised Learning)":
        n_clusters = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
        if st.sidebar.button("🚀 Run Unsupervised Analysis"):
            with st.spinner("Running unsupervised analysis..."):
                X_scaled, _, _ = clean_data(df)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(X_scaled)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                results_df = pd.DataFrame(df)
                results_df['Cluster'] = clusters
                results_df['pca-one'] = X_pca[:, 0]
                results_df['pca-two'] = X_pca[:, 1]
            st.header("🧩 Unsupervised Learning Results")
            fig = px.scatter(results_df, x='pca-one', y='pca-two', color='Cluster', title="Clusters Visualized with PCA")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a CSV file in the sidebar to get started on this page.")