import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
# Read the dataset from the CSV file
data = pd.read_csv('indian_liver_patient.csv')

# 2. Data Preprocessing
# Handling Missing Values: Fill missing 'Albumin_and_Globulin_Ratio' with the mean value
data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

# Encoding Categorical Data: Convert 'Gender' column to numerical values (Male=1, Female=0)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Standardize Target Variable:
# Original dataset uses 1 for Disease and 2 for Healthy.
# We map them to standard binary classification: 1 = Disease, 0 = Healthy
data['Dataset'] = data['Dataset'].map({1: 1, 2: 0})

# 3. Prepare Training Data
X = data.drop('Dataset', axis=1) # Features (Independent variables)
y = data['Dataset']              # Target (Dependent variable)

# Split the data into Training set (80%) and Test set (20%)
# stratify=y ensures the proportion of Disease/Healthy patients is consistent in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training
# Initialize Random Forest Classifier with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training sets
model.fit(X_train, y_train)

# 5. Model Evaluation
# Predict the response for the test dataset
predictions = model.predict(X_test)

# Print accuracy and detailed classification report (Precision, Recall, F1-Score)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))