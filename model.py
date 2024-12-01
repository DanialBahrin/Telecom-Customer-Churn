import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to load and preprocess data
def preprocess_data(data):
    if 'CustomerID' in data.columns:
        data = data.drop(columns=['CustomerID'])

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

# Function to train model using SMOTEENN
def train_model(data):
    # Preprocess data
    data, label_encoders = preprocess_data(data)

    if 'Churn' not in data.columns:
        raise ValueError("The dataset must contain a 'Churn' column for prediction.")

    # Split data into features (X) and target (y)
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Use SMOTEENN for resampling the data
    smoteenn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    # Split resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, label_encoders, accuracy, X_resampled, y_resampled
