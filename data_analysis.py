# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 1: Load Data
# -------------------------------
data_path = r"C:\Users\user\Desktop\Training\processed_data\balanced_data"  # Replace with your actual directory
files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".csv")]

# Combine data from all users
all_data = []
all_labels = []
for file in files:
    data = pd.read_csv(file)
    all_data.append(data.iloc[:, :-1].values)  # Features (exclude last column)
    all_labels.append(data.iloc[:, -1].values[0])  # User ID (same for all rows)

X = np.vstack(all_data)  # Combine features from all users
y = np.hstack([[label] * len(data) for label, data in zip(all_labels, all_data)])  # Combine labels

# -------------------------------
# Step 2: Check for Missing Values
# -------------------------------
print("Checking for missing values...")
missing_values = np.isnan(X).sum()
print(f"Total missing values: {missing_values}")
if missing_values > 0:
    print("Handling missing values by imputing with mean...")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

# -------------------------------
# Step 3: Visualize Feature Distributions
# -------------------------------
print("Visualizing feature distributions...")
num_features = X.shape[1]
for i in range(num_features):
    plt.figure()
    plt.hist(X[:, i], bins=50, alpha=0.7)
    plt.title(f"Feature {i} Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# Step 4: Check for Class Imbalance
# -------------------------------
print("Checking for class imbalance...")
class_counts = Counter(y)
plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.title("User ID Distribution")
plt.xlabel("User ID")
plt.ylabel("Count")
plt.show()
print("Class distribution:")
for user_id, count in class_counts.items():
    print(f"User ID {user_id}: {count} samples")

# -------------------------------
# Step 5: Correlation Analysis
# -------------------------------
print("Performing correlation analysis...")
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# -------------------------------
# Step 6: Feature Importance with Random Forest
# -------------------------------
print("Evaluating feature importance using Random Forest...")
# Subsample data for feature importance
subset_size = 50000  # Adjust based on your system's memory
X_subset = X[:subset_size]
y_subset = y[:subset_size]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_subset)

# Plot feature importance
importances = rf.feature_importances_
plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.show()

# -------------------------------
# Step 7: Sanity Check with Random Labels
# -------------------------------
print("Performing sanity check with random labels...")
random_y = np.random.permutation(y_subset)
rf_random = RandomForestClassifier(n_estimators=50, random_state=42)  # Fewer trees for memory efficiency
rf_random.fit(X_scaled, random_y)
predictions_random = rf_random.predict(X_scaled)
random_accuracy = accuracy_score(random_y, predictions_random)
print(f"Accuracy with random labels: {random_accuracy:.4f}")

# -------------------------------
# Step 8: Overfitting Check on Small Subset
# -------------------------------
print("Checking if the model can overfit on a small subset...")
small_subset_size = 1000
X_small = X_scaled[:small_subset_size]
y_small = y_subset[:small_subset_size]
rf_small = RandomForestClassifier(n_estimators=50, random_state=42)
rf_small.fit(X_small, y_small)
predictions_small = rf_small.predict(X_small)
small_subset_accuracy = accuracy_score(y_small, predictions_small)
print(f"Accuracy on small subset: {small_subset_accuracy:.4f}")

# -------------------------------
# Step 9: Logistic Regression Sanity Check
# -------------------------------
print("Performing sanity check with Logistic Regression...")
logistic = LogisticRegression(max_iter=1000, random_state=42)
logistic.fit(X_small, random_y[:small_subset_size])
predictions_logistic = logistic.predict(X_small)
logistic_accuracy = accuracy_score(random_y[:small_subset_size], predictions_logistic)
print(f"Accuracy with random labels (Logistic Regression): {logistic_accuracy:.4f}")

# -------------------------------
# Summary
# -------------------------------
print("Data analysis complete.")
if missing_values > 0:
    print("Warning: Missing values were found and handled. Consider investigating why they occurred.")

if small_subset_accuracy < 0.9:
    print("Warning: The model failed to overfit on a small subset. This could indicate issues with the model or the data.")

if random_accuracy > 0.1:
    print("Warning: Accuracy with random labels is higher than expected. The data may not be meaningful enough for classification.")
