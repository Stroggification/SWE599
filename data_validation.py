# Import necessary libraries
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
"""
this script is used to find and replace missing datas in the dataset and then fixing the imbalance between samples for different users.
"""
# -------------------------------
# Step 1: Load and Process Data (Handle Missing Values)
# -------------------------------
data_path = r"C:\Users\user\Desktop\Training\trainingdata"  # Original data directory
output_path = r"C:\Users\user\Desktop\Training\processed_data"  # Directory to save processed data
os.makedirs(output_path, exist_ok=True)

files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".csv")]

column_names = None  # To store column names
for file in files:
    print(f"Processing file: {file}")
    
    # Load data
    data = pd.read_csv(file)
    if column_names is None:
        column_names = data.columns  # Preserve column names
    
    user_id = data.iloc[0, -1]  # Assuming the last column contains the user ID
    
    # Handle missing values
    print("Checking and imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])  # Impute only feature columns

    # Save the processed data
    output_file = os.path.join(output_path, f"user_{int(user_id)}.csv")
    print(f"Saving processed file: {output_file}")
    data.to_csv(output_file, index=False)

print("Step 1 complete: All individual files processed and saved.")

# -------------------------------
# Step 2: Combine All Files for Balancing
# -------------------------------
print("Combining all processed files for balancing...")

processed_files = [os.path.join(output_path, file) for file in os.listdir(output_path) if file.endswith(".csv")]
all_data = []
all_labels = []

for file in processed_files:
    data = pd.read_csv(file)
    all_data.append(data.iloc[:, :-1].values)  # Features (exclude last column)
    all_labels.append(data.iloc[:, -1].values[0])  # User ID (same for all rows)

X = np.vstack(all_data)  # Combine features from all users
y = np.hstack([[label] * len(data) for label, data in zip(all_labels, all_data)])  # Combine labels

# -------------------------------
# Step 3: Fix Class Imbalance
# -------------------------------
print("Fixing class imbalance using oversampling...")
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)

# Save balanced data per user
print("Saving balanced data per user...")
balanced_output_path = os.path.join(output_path, "balanced_data")
os.makedirs(balanced_output_path, exist_ok=True)

user_ids = np.unique(y_balanced)
for user_id in user_ids:
    # Extract data for the current user
    user_data = X_balanced[y_balanced == user_id]
    user_labels = y_balanced[y_balanced == user_id]
    
    # Create a DataFrame with preserved column names
    user_df = pd.DataFrame(user_data, columns=column_names[:-1])  # Exclude the label column name
    user_df[column_names[-1]] = user_id  # Add the label column with the user ID

    # Save to CSV
    output_file = os.path.join(balanced_output_path, f"user_{int(user_id)}_balanced.csv")
    print(f"Saving balanced file for User ID {int(user_id)}: {output_file}")
    user_df.to_csv(output_file, index=False)

print("Step 3 complete: Balanced data saved per user.")

# -------------------------------
# Further Steps (e.g., Visualization, Feature Importance)
# -------------------------------
print("Further analysis can now be performed on the processed data...")
