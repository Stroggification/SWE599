import os
import pandas as pd

def read_and_downsample(file_path, original_rate, target_rate):
    """
    Reads and downsamples a single file based on the original and target sampling rates.

    Args:
        file_path (str): Path to the file.
        original_rate (float): Original sampling rate of the data.
        target_rate (float): Target sampling rate for downsampling.

    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    # Read the file
    df = pd.read_csv(file_path, header=0)

    # Rename the first column if unnamed
    if df.columns[0] in ['', 'Unnamed: 0']:
        df.rename(columns={df.columns[0]: 'data'}, inplace=True)

    # Downsample based on the ratio of original to target rates
    step = int(original_rate / target_rate)
    downsampled_df = df.iloc[::step].reset_index(drop=True)

    # Remove any timestamp columns (keep only relevant data columns we dont need timstamps since we know the smapling rates for every file)
    downsampled_df = downsampled_df.iloc[:, 1:]

    # Print the downsampled data for debugging
    print(f"Downsampled {file_path}:\n", downsampled_df.head())

    return downsampled_df

def process_user_folder(user_folder, target_rate):
    """
    Processes all files in a single user's folder, downsamples them, and merges them.

    Args:
        user_folder (str): Path to the user folder.
        target_rate (float): Target sampling rate for all data.

    Returns:
        pd.DataFrame: Merged DataFrame for the user.
    """
    # Define file paths and sampling rates
    files_info = {
        "ACC": {"file": os.path.join(user_folder, "MF_ACC.csv"), "rate": 32},
        "TEMP": {"file": os.path.join(user_folder, "MF_T.csv"), "rate": 1},
        "GSR": {"file": os.path.join(user_folder, "MF_GSR.csv"), "rate": 8}
    }

    # Process feature files starting with 'user' 
    user_feature_files = [
        os.path.join(user_folder, file_name)
        for file_name in os.listdir(user_folder)
        if file_name.startswith("user") and file_name.endswith(".csv")
    ]

    for i, file_path in enumerate(sorted(user_feature_files)):
        files_info[f"DAY{i}"] = {"file": file_path, "rate": 1/60}

    # Process each file
    data_frames = []
    for key, info in files_info.items():
        if os.path.exists(info["file"]):
            df = read_and_downsample(info["file"], original_rate=info["rate"], target_rate=target_rate)
            df.columns = [f"{key}_{col}" for col in df.columns]  # Prefix columns with data type
            data_frames.append(df)
        else:
            print(f"{key} file not found in {user_folder}")

    if not data_frames:
        print(f"No valid files found in {user_folder}")
        return pd.DataFrame()

    # Merge all data based on row index
    merged_data = pd.concat(data_frames, axis=1)

    # shortest DataFrame length
    min_length = min(df.shape[0] for df in data_frames)
    merged_data = merged_data.iloc[:min_length]

    return merged_data

def process_all_users(input_folder, output_folder, target_rate):
    """
    Processes all user folders in the input folder, downsamples their data, and saves the merged outputs.

    Args:
        input_folder (str): Path to the folder containing all user folders.
        output_folder (str): Path to the folder to save processed files.
        target_rate (float): Target sampling rate for all data.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each user folder
    for user_folder in os.listdir(input_folder):
        user_path = os.path.join(input_folder, user_folder)
        if os.path.isdir(user_path):
            print(f"Processing {user_folder}...")

            # Process the user's folder
            merged_data = process_user_folder(user_path, target_rate)

            if not merged_data.empty:
                # Save the merged data
                output_file = os.path.join(output_folder, f"{user_folder}_merged.csv")
                merged_data.to_csv(output_file, index=False)
                print(f"Saved merged data for {user_folder} to {output_file}")
            else:
                print(f"No data to save for {user_folder}")

# Input and output paths
input_folder = r"C:\Users\user\Desktop\SWEET TEST"
output_folder = r"C:\Users\user\Desktop\SWEET TEST\SWEETtest1\sweetmerge"
target_rate = 1/60  # Target sampling rate in Hz (once per minute)


process_all_users(input_folder, output_folder, target_rate)
