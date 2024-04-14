import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('trainings_data.csv')

# Clean the data
data_cleaned = data.drop(['name', 'email', 'kontakt_id', 'produkt_code_pl'], axis=1)
data_cleaned['geschlecht'].fillna('Unspecified', inplace=True)
data_cleaned = data_cleaned.dropna(subset=['has_contract'])
columns_to_fill = ['produkt_fachbereich', 'produkt_name', 'studium_beginn', 'product_interest_type']
data_cleaned[columns_to_fill] = data_cleaned[columns_to_fill].fillna('Unknown')

# One-hot encoding
data_encoded = pd.get_dummies(data_cleaned, columns=[
    'geschlecht', 'produkt_zeitraum_c', 'produkt_art_der_ausbildung_c',
    'produkt_standort', 'produkt_fachbereich', 'studium_beginn', 'product_interest_type'
])

# Prepare the features and target variable
X = data_encoded.drop(['date', 'lead_id', 'produkt_id', 'produkt_name', 'has_contract'], axis=1)
y = data_encoded['has_contract'].astype('bool')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
