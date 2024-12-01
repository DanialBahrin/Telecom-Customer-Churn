import streamlit as st
import pandas as pd
from PIL import Image
from model import train_model  # Import the train_model function from model.py
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (you can also cache this for performance)
@st.cache_data
def load_data():
    return pd.read_csv("CustomerChurn.csv")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Introduction", "EDA", "Customer Churn Prediction"])

# Load data
data = load_data()

# Page: Introduction
if page == "Introduction":
    st.title("Telecom Customer Churn Analysis")
    
    # Add a subtitle or description
    st.header("Understanding Customer Churn in Telecom Industry")
    st.write("""
    Customer churn, also known as customer attrition, refers to the loss of customers who stop using a service.
    In the telecom industry, customer churn is a significant concern because acquiring new customers is often
    more expensive than retaining existing ones. Companies analyze churn patterns to predict which customers are
    likely to leave and take steps to retain them.
    """)
      
    st.write("""
    In this app, we analyze customer data to identify trends and patterns that lead to churn. We use a machine learning
    model to predict whether a customer is likely to churn or stay based on various features such as customer demographics,
    usage patterns, and more.
    
    ## Goals of this App:
    - **Explore** customer churn data and understand the factors influencing churn.
    - **Predict** customer churn using machine learning models.
    - **Provide insights** that can help businesses in reducing churn and improving customer retention.
    
    By analyzing the customer data in this app, telecom companies can take proactive steps to improve customer satisfaction,
    reduce churn rates, and increase profitability.
    """)

# Page: EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.header("Customer Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    st.bar_chart(churn_counts)

    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())

# Page: Customer Churn Prediction
elif page == "Customer Churn Prediction":
    st.title("Customer Churn Prediction with SMOTEENN")
    
    # Train the model using SMOTEENN resampled data (loaded from model.py)
    model, label_encoders, model_accuracy, X_resampled, y_resampled = train_model(data)
    st.write(f"The trained Random Forest model using SMOTEENN has an accuracy of **{model_accuracy:.2f}** on the test set.")
    
    st.subheader("Resampled Data Distribution")
    st.write("After applying SMOTEENN, the class distribution is as follows:")
    st.bar_chart(pd.Series(y_resampled).value_counts())
    
    # User input form for prediction
    st.subheader("Enter Customer Details")
    input_data = {}
    for column in data.drop(['Churn', 'CustomerID'], axis=1).columns:
        if data[column].dtype == 'object':
            options = data[column].unique()
            input_data[column] = st.selectbox(f"{column}", options)
        else:
            min_val = data[column].min()
            max_val = data[column].max()
            input_data[column] = st.slider(f"{column}", min_value=int(min_val), max_value=int(max_val), value=int(data[column].mean()))
    
    st.write("### Preview of Customer Data Entered:")
    st.write(input_data)
    
    # Convert input data to model format
    input_df = pd.DataFrame([input_data])
    for column, le in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = le.transform(input_df[column])

    # Predict churn
    if st.button("Predict Churn"):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][prediction]
            
            # Show Prediction with Visual Icon
            if prediction == 1:
                result = "Churn"
                st.markdown("### ⚠️ Customer is likely to churn!")
                st.image("https://img.icons8.com/ios/452/close-window.png", width=100)
            else:
                result = "No Churn"
                st.markdown("### ✅ Customer is likely to stay!")
                st.image("https://img.icons8.com/ios/452/ok.png", width=100)
            
            # Show confidence level
            st.write(f"Model Confidence: **{probability * 100:.2f}%**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
