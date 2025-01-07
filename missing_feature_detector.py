import os
import pandas as pd
"""
Debugging script to see if all the features are persistent in merged data. eg user00123 missing the hearth rate values
they are not in the original dataset either!!

"""
# Define the path to your files
data_path = r"C:\Users\user\Desktop\Training\trainingdata"  # Replace with the actual directory
files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".csv")]

# Define the expected features
expected_features = ['ACC_x', 'ACC_y', 'ACC_z', 'TEMP_Temp', 'GSR_GSR', 'DAYS_ECG_mean_heart_rate']


# Dictionary to track missing features by file
missing_features_report = {}

for file in files:
    # Load the file
    data = pd.read_csv(file)
    
    # Identify missing features
    missing_features = [col for col in expected_features if col not in data.columns]
    
    # If there are missing features, log the file and the missing columns
    if missing_features:
        missing_features_report[file] = missing_features

# Print report
if missing_features_report:
    print("Files with missing features:")
    for file, missing in missing_features_report.items():
        print(f"{file}: Missing columns -> {missing}")
else:
    print("All files have the expected features.")
