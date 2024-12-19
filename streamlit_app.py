
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title and Introduction
st.title("Mental Health Prediction App")
st.markdown("This app predicts the likelihood of mental health issues among university students.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Data Exploration
    st.subheader("Data Exploration")
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())

    if st.checkbox("Show Data Visualization"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

    # Preprocessing
    st.subheader("Data Preprocessing")
    target_column = st.selectbox("Select the target column:", data.columns)
    feature_columns = st.multiselect("Select feature columns:", data.columns, default=data.columns[:-1])

    # Splitting Data
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    st.subheader("Model Training")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Prediction
    st.subheader("Make a Prediction")
    user_input = {col: st.number_input(f"Enter value for {col}:") for col in feature_columns}
    user_df = pd.DataFrame([user_input])
    prediction = model.predict(user_df)
    st.write("Prediction:", prediction[0])
else:
    st.warning("Please upload a dataset to proceed.")
