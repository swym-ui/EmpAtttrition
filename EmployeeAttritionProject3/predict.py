import pandas as pd
import pickle
import os

# Get base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(base_dir, 'models', 'attrition_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load encoders and scaler
with open(os.path.join(base_dir, 'models', 'label_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

with open(os.path.join(base_dir, 'models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load new employee data
input_csv = os.path.join(base_dir, 'data', 'employee_dataset_2_updated.csv')
df = pd.read_csv(input_csv)

# Keep a copy of original for final output
original_df = df.copy()

# Drop unnecessary columns (except 'Name' for later use)
columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Save Name column separately (if it exists)
names = df['Name'] if 'Name' in df.columns else None

# Drop Name before prediction
df = df.drop(columns=['Name'], errors='ignore')

# Apply label encoding to categorical columns
for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

# Scale numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df[numerical_cols] = scaler.transform(df[numerical_cols])

# Predict
predictions = model.predict(df)
pred_labels = ['Yes' if p == 1 else 'No' for p in predictions]

# Build output DataFrame
output_df = pd.DataFrame()
if names is not None:
    output_df['Name'] = names
output_df['Attrition'] = pred_labels

# ✅ Add back important streamlit-related columns in proper order
for col in ['Age', 'DistanceFromHome', 'Education', 'JobSatisfaction', 'MonthlyIncome',
            'YearsAtCompany', 'TasksCompletedLastMonth', 'TasksCompletedNextMonth',
            'Department', 'JobRole']:
    if col in original_df.columns:
        output_df[col] = original_df[col]

# Save to predictions folder
output_dir = os.path.join(base_dir, 'predictions')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'predicted_attrition.csv')
output_df.to_csv(output_path, index=False)

print(f"\n Prediction completed. Results saved to {output_path}")
