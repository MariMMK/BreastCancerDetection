import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the trained model, scaler, and selector
mlp_model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
selector = joblib.load('selector.joblib')

# Load dataset for feature names
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Get the selected features
selected_features = df.columns[selector.get_support()]

# Streamlit app
st.title('Breast Cancer Prediction')
st.write("This is a simple Streamlit app to predict breast cancer using a trained ANN model.")

input_data = []

for feature in selected_features:
    input_val = st.number_input(f"Enter value for {feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()))
    input_data.append(input_val)

if st.button('Predict'):
    input_data = [input_data]
    input_data_selected = pd.DataFrame(input_data, columns=selected_features)
    input_data_scaled = scaler.transform(input_data_selected)    # Scale the input data
    prediction = mlp_model.predict(input_data_scaled)
    st.write(f'The prediction is: {"Malignant" if prediction[0] == 0 else "Benign"}')

    # Visualize the input data
    st.write("Input Data:")
    st.write(input_data_selected)

    # Visualize the model prediction probability
    y_prob = mlp_model.predict_proba(input_data_scaled)[0]
    st.write("Prediction Probability:")
    st.write(f"Malignant: {y_prob[0]:.2f}")
    st.write(f"Benign: {y_prob[1]:.2f}")

    # Plot prediction probability
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Malignant', 'Benign'], y=y_prob)
    plt.title('Prediction Probability')
    st.pyplot(plt)