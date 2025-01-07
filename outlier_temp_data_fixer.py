import os
import pandas as pd
import matplotlib.pyplot as plt


"""
There were some outlier temparature data observed while visualizing the data with datavalidation.py script. The skin temparature values for some users were above 3000 C adn below -300 C degrees which indicated an error in data collection
these values are replaced using this script

"""

# Path to the directory containing CSV files
data_path = r"C:\Users\user\Desktop\Training\trainingdata"

# Combine all CSV files in the directory
all_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]

# Create an empty DataFrame to store all data
df_list = []

# Loop through files, process, and save corrected data
for file in all_files:
    # Read each file
    temp_df = pd.read_csv(file)

    # Check if TEMP_Temp exists in the columns
    if 'TEMP_Temp' in temp_df.columns:
        # Detect temperature outliers greater than 300
        high_temp_outliers = temp_df[temp_df['TEMP_Temp'] > 300]
        # Detect temperature outliers less than 0
        low_temp_outliers = temp_df[temp_df['TEMP_Temp'] < 0]

        # Correct high temperature outliers by dividing by 100
        if not high_temp_outliers.empty:
            print(f"High temperature outliers detected in file: {file}")
            print(high_temp_outliers)
            temp_df.loc[temp_df['TEMP_Temp'] > 300, 'TEMP_Temp'] /= 100
            print(f"Corrected high temperature outliers in file: {file}")

        # Correct low temperature outliers by replacing them with 0 (or other logic)
        if not low_temp_outliers.empty:
            print(f"Low temperature outliers detected in file: {file}")
            print(low_temp_outliers)
            temp_df.loc[temp_df['TEMP_Temp'] < 0, 'TEMP_Temp'] = 0
            print(f"Corrected low temperature outliers in file: {file}")

        # Save the corrected file back to the same location
        temp_df.to_csv(file, index=False)
        print(f"Saved corrected file: {file}")
    else:
        print(f"TEMP_Temp column not found in file: {file}")

# Re-visualize the corrected feature (for all files combined)
df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

# Check distribution of TEMP_Temp after corrections
if 'TEMP_Temp' in df.columns:
    plt.hist(df['TEMP_Temp'], bins=50, alpha=0.7)
    plt.title("Corrected TEMP_Temp Distribution (Temperature)")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("TEMP_Temp column not found in the combined dataset.")
