# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Load the data
df = pd.read_csv('trainings_data.csv')

# Initial data exploration
print("First few rows of the dataset:")
print(df.head())

# Data types and missing values
print("\nData types and missing value counts:")
print(df.info())
# Missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Data Preprocessing
# Exclude unnecessary columns
df.drop(['name', 'email', 'produkt_code_pl', 'lead_id', 'kontakt_id', 'produkt_id'], axis="columns", inplace=True)

# Handle missing values
df['geschlecht'] = df['geschlecht'].fillna('Unspecified')


# Encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
label = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(lambda x: label.fit_transform(x))

# Splitting the dataset
X = df.drop('has_contract', axis=1)
y = df['has_contract']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, reg_lambda=0.5),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, min_samples_split=10)
}

model_results = {}  # Use a different name for the dictionary

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred.round())
    model_results[name] = accuracy  # Store results in the new dictionary
    print(f"{name} Accuracy: {accuracy}")
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred.round()))

# Identifying the best model
best_model = max(model_results, key=model_results.get)
print(f"Based on our model comparison, the following model showed the best performance: {best_model} with an Overall Accuracy of {model_results[best_model]:.2%}")

# Visualization of feature importance for XGBoost
xgb_model = models['XGBoost']
feature_importances = xgb_model.feature_importances_
plt.barh(np.arange(len(feature_importances)), feature_importances, align='center')
plt.yticks(np.arange(len(X.columns)), X.columns)
plt.xlabel('Feature Importance')
plt.title('Feature Importance for XGBoost Model')
plt.show()

# Visualization of predicted vs actual values for XGBoost
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values for XGBoost Model')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.show()

# Visualization of the model performances
plt.figure(figsize=(10, 5))
plt.bar(model_results.keys(), model_results.values(), color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim([0.6, 0.75])
plt.show()
