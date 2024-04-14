import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Setup plot styles
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")



# Load the data
df = pd.read_csv('trainings_data.csv')
print(df.head())

# Exclude unnecessary columns
df.drop(['name', 'email', 'produkt_code_pl', 'lead_id'], axis="columns", inplace=True)  # Assuming 'lead_id' is the name of the column

# Handle missing values
df['geschlecht'] = df['geschlecht'].fillna('Unspecified')
df.dropna(subset=['has_contract'], inplace=True)
columns_to_fill = ['produkt_fachbereich', 'produkt_name', 'studium_beginn', 'product_interest_type']
df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')

# Convert target variable if it's categorical
if df['has_contract'].dtype == object:
    df['has_contract'] = df['has_contract'].astype("category").cat.codes

# Handling categorical columns and encoding
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'has_contract']
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(lambda x: label.fit_transform(x))

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare features and target variable
X = df.drop('has_contract', axis=1)
y = df['has_contract']

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
