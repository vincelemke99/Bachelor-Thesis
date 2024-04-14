# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load the data
df = pd.read_csv('trainings_data.csv')

# Initial data exploration
print("First few rows of the dataset:")
print(df.head())

# Data types and missing values
print("\nData types and missing value counts:")
print(df.info())

# Data Preprocessing
# Exclude unnecessary columns
df.drop(['name', 'email', 'produkt_code_pl', 'lead_id'], axis="columns", inplace=True)

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

# Define the XGBoost regressor with hyperparameters for tuning
xgb_clf = xgb.XGBRegressor(objective='reg:squarederror')

# Setup the parameter grid for GridSearchCV
parameters = {
    'n_estimators': [60, 100, 120, 140],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7],
    'reg_lambda': [0.5]
}

# Initialize and fit the GridSearchCV
xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1)
xgb_reg.fit(X_train_scaled, y_train)

# Output the best performance score and parameters
print("Best score: {:.3f}".format(xgb_reg.best_score_))
print("Best parameters set:", xgb_reg.best_params_)

# Predictions and evaluation
y_pred = xgb_reg.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred.round()))
print("Classification Report:\n", classification_report(y_test, y_pred.round()))

# Visualization of feature importance
feature_importances = xgb_reg.best_estimator_.feature_importances_
plt.barh(np.arange(len(feature_importances)), feature_importances, align='center')
plt.yticks(np.arange(len(X.columns)), X.columns)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.show()

# Visualization of predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.show()
