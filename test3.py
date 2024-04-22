# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

#pd.set_option('display.max_rows', None)  # Replace None with a number if you want to limit to a specific count
pd.set_option('display.max_columns', None)  # Adjusts how many columns are shown
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being split across multiple pages
pd.set_option('display.width', None)  # Use None to automatically adjust to your screen width

# Load the data
df = pd.read_csv('trainings_data.csv')

# create a new column, date_parsed, with the parsed dates
df['date_parsed'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
print(df['date_parsed'].head())
df.drop(['date'], axis=1, inplace=True)
df['Year'] = df['date_parsed'].dt.year
df['Month'] = df['date_parsed'].dt.month
df.drop(['date_parsed'], axis=1, inplace=True)


#df.head
print(df.head)
print(df.shape)
df.drop(['name', 'email', 'produkt_code_pl', 'lead_id', 'kontakt_id', 'produkt_id'], axis="columns", inplace=True)

df.rename(columns={'Year' : 'Jahr', 'Month' : 'Monat', 'Day' : 'Tag' , 'name' : 'Name', 'geschlecht' : 'Geschlecht', 'produkt_zeitraum_c' : 'Studientyp',
       'produkt_art_der_ausbildung_c' : 'Studiengangsart', 'produkt_standort' : 'Studienort',
       'produkt_fachbereich' : 'Fachbereich', 'produkt_name': 'Studiengang', 'studium_beginn' : 'Studien Beginn',
       'product_interest_type' : 'Conversion Type', 'is_converted' : 'Konvertiert', 'has_contract' : 'Vertragsabschluss'},inplace=True)

# get the number of missing data points per column
missing_values_count = df.isnull().sum()
print(missing_values_count)

# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
print(total_missing)
print(total_cells)

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

print(df.info())  # Check data types
print(df.head())

df['Geschlecht'] = df['Geschlecht'].fillna('Unspecified')
columns_to_fill = ['Fachbereich', 'Studiengang', 'Studien Beginn', 'Conversion Type']
df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')



# Label encoding for high cardinality columns
label_encoder = LabelEncoder()
df['Studiengang'] = df['Studiengang'].apply(lambda x: label_encoder.fit_transform([x])[0])

# One-hot encoding for low cardinality columns
df = pd.get_dummies(df, columns=['Geschlecht', 'Studientyp', 'Studiengangsart', 'Conversion Type'])
# Splitting the dataset
X = df.drop('Vertragsabschluss', axis=1)
y = df['Vertragsabschluss']
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
plt.ylim([0.1, 1.0])
plt.show()
