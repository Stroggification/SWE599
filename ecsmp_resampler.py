import os
import pandas as pd
import numpy as np
"""
!note
changes in the latest version

+add user id as a label(to use in training)
+remove timestamps(we dont ned timestmaps in ml training since we already downsmapled data to a common sampling rate)
+remove the frist and last row for data integrity(someone forget to put a "," for the first row)
+remove the "ibi" column from the merged file but keep the logic. so if needed i can also include it while training using "masking".
+The accelerometer is configured to measure acceleration in the range [-2g, 2g]. Therefore the unit in this file is 1/64g.So the ACC data is divided to 64. 
+Ensured that the naming of the columns is same with the SWEET dataset

"""
# common sampling rate in Hz
COMMON_SAMPLING_RATE = 1

# base directory containing all user folders
BASE_DIR = r"C:\Users\user\Desktop\ECSMP DEMO\User Data"

# output directory for merged files
OUTPUT_DIR = r"C:\Users\user\Desktop\ECSMP DEMO\merge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_acc(file_path, master_timestamps):
    """
    Reads ACC.csv (32 Hz), then resamples it onto the master_timestamps.
    - If COMMON_SAMPLING_RATE < 32, we downsample by averaging over each master timestep.
    - If COMMON_SAMPLING_RATE > 32, we upsample by interpolating.
    """
    acc_data = pd.read_csv(file_path, header=None, names=["x", "y", "z"])
    if acc_data.empty:
        return pd.DataFrame({"timestamp": master_timestamps})

    # Original ACC timestamps (32 Hz)
    acc_data["timestamp"] = np.arange(0, len(acc_data) / 32, 1 / 32)
    acc_data.set_index("timestamp", inplace=True)

    original_rate = 32
    if COMMON_SAMPLING_RATE < original_rate:
        # ----------- DOWNsampling by average -----------
        # 1) Cut into bins based on master_timestamps
        # 2) For each bin, average the ACC data within that bin
        df_list = []
        for i in range(len(master_timestamps) - 1):
            t_start = master_timestamps[i]
            t_end   = master_timestamps[i + 1]
            # Slice data in [t_start, t_end)
            subset = acc_data.loc[(acc_data.index >= t_start) & (acc_data.index < t_end)]
            if not subset.empty:
                mean_vals = subset[["x", "y", "z"]].mean()
                df_list.append([t_start, *mean_vals.values])
            else:
                df_list.append([t_start, np.nan, np.nan, np.nan])

        # Build a DataFrame aligned with master_timestamps
        down_df = pd.DataFrame(df_list, columns=["timestamp", "x", "y", "z"])
        # Last bin edge: align final timestamp exactly
        down_df.loc[len(down_df)] = [master_timestamps[-1], np.nan, np.nan, np.nan]
        final_df = down_df

    else:
        # ----------- UPsampling or same rate -----------
        # Reindex to master_timestamps and interpolate
        final_df = acc_data.reindex(master_timestamps).interpolate().reset_index()
        
        final_df.columns = ["timestamp", "x", "y", "z"]

    # Divide ACC values by 64  The accelerometer is configured to measure acceleration in the range [-2g, 2g]. Therefore the unit in this file is 1/64g.
    final_df[["x", "y", "z"]] = final_df[["x", "y", "z"]] / 64

    return final_df


def process_temp(file_path, master_timestamps):
    """
    Reads TEMP.csv (4 Hz), then resamples it onto the master_timestamps.
    Downsampling = average, Upsampling = interpolate.
    """
    temp_data = pd.read_csv(file_path, header=None, skiprows=2, names=["temperature"])
    if temp_data.empty:
        return pd.DataFrame({"timestamp": master_timestamps})

    original_rate = 4
    temp_data["timestamp"] = np.arange(0, len(temp_data) / original_rate, 1 / original_rate)
    temp_data.set_index("timestamp", inplace=True)

    if COMMON_SAMPLING_RATE < original_rate:
        # Downsample by bins
        df_list = []
        for i in range(len(master_timestamps) - 1):
            t_start = master_timestamps[i]
            t_end   = master_timestamps[i + 1]
            subset = temp_data.loc[(temp_data.index >= t_start) & (temp_data.index < t_end)]
            if not subset.empty:
                df_list.append([t_start, subset["temperature"].mean()])
            else:
                df_list.append([t_start, np.nan])

        down_df = pd.DataFrame(df_list, columns=["timestamp", "temperature"])
        down_df.loc[len(down_df)] = [master_timestamps[-1], np.nan]
        final_df = down_df

    else:
        # Upsample / same rate
        final_df = temp_data.reindex(master_timestamps).interpolate().reset_index()
        final_df.columns = ["timestamp", "temperature"]

    return final_df


def process_eda(file_path, master_timestamps):
    """
    Reads EDA.csv (4 Hz), then resamples it onto the master_timestamps.
    Downsampling = average, Upsampling = interpolate.
    """
    eda_data = pd.read_csv(file_path, header=None, skiprows=2, names=["eda"])
    if eda_data.empty:
        return pd.DataFrame({"timestamp": master_timestamps})

    original_rate = 4
    eda_data["timestamp"] = np.arange(0, len(eda_data) / original_rate, 1 / original_rate)
    eda_data.set_index("timestamp", inplace=True)

    if COMMON_SAMPLING_RATE < original_rate:
        # Downsample by bins
        df_list = []
        for i in range(len(master_timestamps) - 1):
            t_start = master_timestamps[i]
            t_end   = master_timestamps[i + 1]
            subset = eda_data.loc[(eda_data.index >= t_start) & (eda_data.index < t_end)]
            if not subset.empty:
                df_list.append([t_start, subset["eda"].mean()])
            else:
                df_list.append([t_start, np.nan])

        down_df = pd.DataFrame(df_list, columns=["timestamp", "eda"])
        down_df.loc[len(down_df)] = [master_timestamps[-1], np.nan]
        final_df = down_df

    else:
        # Upsample / same rate
        final_df = eda_data.reindex(master_timestamps).interpolate().reset_index()
        final_df.columns = ["timestamp", "eda"]

    return final_df


def process_hr(file_path, master_timestamps):
    """
    Reads HR.csv (1 Hz), then resamples onto master_timestamps.
    If COMMON_SAMPLING_RATE > 1, upsample by interpolation; otherwise same rate or skip.
    """
    hr_data = pd.read_csv(file_path, header=None, skiprows=2, names=["hr"])
    if hr_data.empty:
        return pd.DataFrame({"timestamp": master_timestamps})

    original_rate = 1
    # Timestamps from 0..N at 1 Hz
    hr_data["timestamp"] = np.arange(0, len(hr_data) / original_rate, 1 / original_rate)
    hr_data.set_index("timestamp", inplace=True)

    if COMMON_SAMPLING_RATE > original_rate:
        final_df = hr_data.reindex(master_timestamps).interpolate().reset_index()
        final_df.columns = ["timestamp", "hr"]
    else:
        # If sampling rate is <= 1, effectively keep it or downsample via bins
        # but typically you'd keep it as-is or do the same bin approach as above
        # For simplicity, let's just reindex (which won't change anything if master_timestamps is coarser).
        final_df = hr_data.reindex(master_timestamps, method="ffill").reset_index()
        final_df.columns = ["timestamp", "hr"]

    return final_df


def process_ibi(file_path, max_time, master_timestamps):
    """
    IBI has event-based data (no fixed sampling rate).
    We reindex with method="nearest" or interpolation onto master_timestamps up to max_time.
    """
    ibi_data = pd.read_csv(file_path, header=None, skiprows=1, names=["time", "ibi"])
    if ibi_data.empty:
        return pd.DataFrame({"timestamp": master_timestamps, "ibi": np.nan})

    # cumulative_time: actual second each beat occurs
    ibi_data["cumulative_time"] = ibi_data["time"].cumsum()
    ibi_data.set_index("cumulative_time", inplace=True)

    # Reindex to master_timestamps, which stops at max_time
    # If master_timestamps extends beyond max_time thats okay We ll still do nearest or fill.
    up_to_idx = (master_timestamps <= max_time)
    relevant_times = master_timestamps[up_to_idx]

    ibi_resampled = (
        ibi_data["ibi"]
        .reindex(relevant_times, method="nearest")
        .reindex(master_timestamps, method="ffill")  # fill beyond max_time with last known
        .reset_index()
    )
    ibi_resampled.columns = ["timestamp", "ibi"]
    return ibi_resampled

def merge_data(user_dir):
    # Paths
    acc_file = os.path.join(user_dir, "ACC.csv")
    temp_file = os.path.join(user_dir, "TEMP.csv")
    eda_file = os.path.join(user_dir, "EDA.csv")
    hr_file = os.path.join(user_dir, "HR.csv")
    ibi_file = os.path.join(user_dir, "IBI.csv")

    # Load HR, find max_time
    hr_data = pd.read_csv(hr_file, header=None, skiprows=2, names=["hr"])
    if hr_data.empty:
        print(f"Warning: {hr_file} empty. Skipping.")
        return pd.DataFrame()

    # HR is 1 Hz last index is total duration
    original_rate = 1
    hr_data["timestamp"] = np.arange(0, len(hr_data) / original_rate, 1 / original_rate)
    max_time = hr_data["timestamp"].max()

    # Create the master_timestamps
    master_timestamps = np.arange(0, max_time + 1 / COMMON_SAMPLING_RATE, 1 / COMMON_SAMPLING_RATE)

    # Process each file with master_timestamps
    acc = process_acc(acc_file, master_timestamps)
    temp = process_temp(temp_file, master_timestamps)
    eda = process_eda(eda_file, master_timestamps)
    hr = process_hr(hr_file, master_timestamps)
    ibi = process_ibi(ibi_file, max_time, master_timestamps)  # Still process but don't include in final merge

    # Merge only acc, temp, eda, and hr data
    dfs = [acc, temp, eda, hr]
    merged = acc  # Start with ACC
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="timestamp", how="outer")  # or 'inner' if you prefer

    # Optional: Log IBI for debugging purposes
    if not ibi.empty:
        print(f"Processed IBI for {user_dir}, but not saving to merged file.")

    # Rename columns
    merged.rename(
        columns={
            "x": "ACC_x",
            "y": "ACC_y",
            "z": "ACC_z",
            "temperature": "TEMP_Temp",
            "eda": "GSR_GSR",
            "hr": "DAYS_ECG_mean_heart_rate",
        },
        inplace=True,
    )

    return merged




def main():
    for user_folder in os.listdir(BASE_DIR):
        user_dir = os.path.join(BASE_DIR, user_folder)
        if os.path.isdir(user_dir):
            print(f"Processing user: {user_folder}")
            merged_data = merge_data(user_dir)
            if not merged_data.empty:
                # Extract user ID from the folder name
                user_id = int(user_folder.split("_")[0])  # Assuming the folder starts with the user number
                # Add the user_id column
                merged_data["user_id"] = user_id
                # Drop the 'timestamp' column before saving
                merged_data_no_timestamp = merged_data.drop(columns=["timestamp"])
                # Exclude the first row and the last row
                merged_data_no_timestamp = merged_data_no_timestamp.drop(index=[merged_data_no_timestamp.index[0], merged_data_no_timestamp.index[-1]])
                # Save the updated merged data
                output_path = os.path.join(OUTPUT_DIR, f"{user_folder}_merged.csv")
                merged_data_no_timestamp.to_csv(output_path, index=False)
                print(f"Saved merged data for {user_folder} (user_id: {user_id}) without second and last row")
            else:
                print(f"No data to save for {user_folder}")




if __name__ == "__main__":
    main()
