import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
data_path = os.path.join(base_dir, 'data', 'training_dataset.csv')
df = pd.read_csv(data_path)

# Drop rows with missing target
df = df.dropna(subset=['Attrition'])

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Categorical columns
categorical_cols = ['Gender', 'Department', 'JobRole', 'OverTime', 'BusinessTravel']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Attrition')

# ✅ Include the two new task-related features if not already present
if 'TasksCompletedLastMonth' not in df.columns:
    df['TasksCompletedLastMonth'] = 0  # or insert realistic defaults
if 'TasksCompletedNextMonth' not in df.columns:
    df['TasksCompletedNextMonth'] = 0

numerical_cols.extend(['TasksCompletedLastMonth', 'TasksCompletedNextMonth'])

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split features/target
X = df.drop(columns=['Attrition', 'Name'])  # exclude name from features
y = df['Attrition']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create model directory
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# Save model
with open(os.path.join(model_dir, 'attrition_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save label encoders
with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

# Save scaler
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model, encoders, and scaler saved successfully.")
