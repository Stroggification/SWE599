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

    # Adjust sampling rate
    if original_rate > target_rate:  # Downsample
        step = int(original_rate / target_rate)
        downsampled_df = df.iloc[::step].reset_index(drop=True)
    else:  # Upsample
        upsample_factor = int(target_rate / original_rate)
        expanded_data = df.reindex(df.index.repeat(upsample_factor)).reset_index(drop=True)
        downsampled_df = expanded_data.iloc[:int(len(expanded_data) * (original_rate / target_rate))]

    # Remove any timestamp columns (keep only relevant data columns)
    downsampled_df = downsampled_df.iloc[:, 1:]

    return downsampled_df

def merge_day_files(user_folder, target_rate):
    """
    Merges all day-specific files for a user folder into a single DataFrame.

    Args:
        user_folder (str): Path to the user folder.
        target_rate (float): Target sampling rate for downsampling.

    Returns:
        pd.DataFrame: Combined DataFrame for all day files.
    """
    # Process feature files starting with 'user'
    user_feature_files = [
        os.path.join(user_folder, file_name)
        for file_name in os.listdir(user_folder)
        if file_name.startswith("user") and file_name.endswith(".csv")
    ]

    day_frames = []
    for file_path in sorted(user_feature_files):  # Ensure correct order (DAY0, DAY1, ...)
        df = read_and_downsample(file_path, original_rate=1/60, target_rate=target_rate)
        day_frames.append(df)

    if day_frames:
        combined_days = pd.concat(day_frames, axis=0).reset_index(drop=True)
        return combined_days
    else:
        print(f"No day-specific files found in {user_folder}")
        return pd.DataFrame()

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

    # Merge all day files into a single DataFrame
    combined_days = merge_day_files(user_folder, target_rate)
    if not combined_days.empty:
        files_info["DAYS"] = {"file": combined_days, "rate": target_rate}  # Add combined days

    # Process each file
    data_frames = []
    for key, info in files_info.items():
        if isinstance(info["file"], str):  # File path for regular files
            if os.path.exists(info["file"]):
                df = read_and_downsample(info["file"], original_rate=info["rate"], target_rate=target_rate)
                df.columns = [f"{key}_{col}" for col in df.columns]  # Prefix columns with data type
                data_frames.append(df)
            else:
                print(f"{key} file not found in {user_folder}")
        else:  # Already processed DataFrame (combined days)
            df = info["file"]
            df.columns = [f"{key}_{col}" for col in df.columns]
            data_frames.append(df)

    if not data_frames:
        print(f"No valid files found in {user_folder}")
        return pd.DataFrame()

    # Merge all data based on row index
    merged_data = pd.concat(data_frames, axis=1)

    # Truncate to the shortest DataFrame length
    min_length = min(df.shape[0] for df in data_frames)
    merged_data = merged_data.iloc[:min_length]

    return merged_data

def extract_user_id(user_folder):
    """
    Extracts the user ID from the folder name.

    Args:
        user_folder (str): Path to the user folder.

    Returns:
        int: User ID extracted from the folder name.
    """
    base_name = os.path.basename(user_folder)
    user_id = "".join(filter(str.isdigit, base_name))
    return int(user_id) if user_id else None

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

            # Extract user ID
            user_id = extract_user_id(user_folder)
            if user_id is None:
                print(f"Could not extract user ID from {user_folder}, skipping.")
                continue

            # Process the user's folder
            merged_data = process_user_folder(user_path, target_rate)

            if not merged_data.empty:
                # Save only the first six columns
                trimmed_data = merged_data.iloc[:, :6]

                # Add the user ID as the last column
                trimmed_data['user_id'] = user_id

                # Save the merged data
                output_file = os.path.join(output_folder, f"{user_folder}_merged.csv")
                trimmed_data.to_csv(output_file, index=False)
                print(f"Saved merged data for {user_folder} to {output_file}")
            else:
                print(f"No data to save for {user_folder}")

# Input and output paths
input_folder = r"C:\Users\user\Desktop\SWEET DEMO\User Data"
output_folder = r"C:\Users\user\Desktop\SWEET DEMO\merge"
target_rate = 1  # Target sampling rate in Hz

# Run the processing script
process_all_users(input_folder, output_folder, target_rate)
