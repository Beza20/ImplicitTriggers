# data_loader.py

import pandas as pd
import numpy as np
from config import HEAD_FILES, HAND_FILES

# we need this function because we stopped recording every 5 - 7 minutes and we need to adjust the time sequence
# so that the time is continuous when we concatenate the dataframes
# this function takes a dataframe, a time column, and a global offset
# it adjusts the time column by subtracting the minimum time and adding the global offset
def adjust_time_sequence(df, time_column, global_offset):
    """
    Adjust timestamps so each file starts at the end of the previous file.
    """
    if df[time_column].max() > 1e12:  # If still in nanoseconds, convert once
        df[time_column] /= 1e9

    min_time = df[time_column].min()
    df[time_column] = df[time_column] - min_time + global_offset

    max_time = df[time_column].max()
    return df, max_time

# Load and preprocess data
# this function converts the time column to seconds, adjusts the time sequence, and handles missing values
# it returns the head and hand dataframes
def load_and_preprocess_data():
    # ---- PROCESS HEAD FILES ---- #
    global_offset = 0  # Start offset
    head_dfs = []
    for f in HEAD_FILES:
        temp_df = pd.read_csv(f)
        temp_df["time_s"] = temp_df["utc_timestamp_ns"] / 1e9  # Convert nanoseconds to seconds
        #print("Columns after creating time_s:", temp_df.columns)  # Debug statement
        temp_df, global_offset = adjust_time_sequence(temp_df, "time_s", global_offset)
        head_dfs.append(temp_df)

    head_df = pd.concat(head_dfs, ignore_index=True)  # Combine all head files
    #print("Columns after concatenation:", head_df.columns) 
    
    # ---- HANDLE MISSING TRACKING DATA FOR HEAD ---- #
    # Set missing values to NaN
    head_df.loc[
        (head_df["tx_world_device"] == 0) & 
        (head_df["ty_world_device"] == 0) & 
        (head_df["tz_world_device"] == 0), 
        ["tx_world_device", "ty_world_device", "tz_world_device"]
    ] = np.nan
    
    # we're leaving this here but we actually ended up not using any of these interpolated data points 
    # because we only care about the angular differnce

    # Interpolate small gaps
    head_df["tx_world_device"] = head_df["tx_world_device"].interpolate(method="linear", limit=5)
    head_df["ty_world_device"] = head_df["ty_world_device"].interpolate(method="linear", limit=5)
    head_df["tz_world_device"] = head_df["tz_world_device"].interpolate(method="linear", limit=5)

    # Smooth data using a moving average
    window_size = 20
    head_df["tx_world_device"] = head_df["tx_world_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    head_df["ty_world_device"] = head_df["ty_world_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    head_df["tz_world_device"] = head_df["tz_world_device"].rolling(window=window_size, center=True, min_periods=1).mean()


    # ---- PROCESS HAND FILES ---- #
    global_offset = 0  # Reset for hand tracking
    hand_dfs = []
    for f in HAND_FILES:
        temp_df = pd.read_csv(f)
        if "time_s" not in temp_df.columns:
            if temp_df["tracking_timestamp_us"].max() > 1e6:  # Likely in microseconds
                temp_df["time_s"] = temp_df["tracking_timestamp_us"] / 1e6
            elif temp_df["tracking_timestamp_us"].max() > 1e3:  # Likely in milliseconds
                temp_df["time_s"] = temp_df["tracking_timestamp_us"] / 1e3
            else:
                temp_df["time_s"] = temp_df["tracking_timestamp_us"]  # Already in seconds

        temp_df, global_offset = adjust_time_sequence(temp_df, "time_s", global_offset)
        hand_dfs.append(temp_df)

    hand_df = pd.concat(hand_dfs, ignore_index=True)  # Combine all hand files
    
    # ---- HANDLE MISSING TRACKING DATA ---- #
    
    hand_df.loc[
        (hand_df["tx_left_wrist_device"] == 0) & 
        (hand_df["ty_left_wrist_device"] == 0) & 
        (hand_df["tz_left_wrist_device"] == 0), 
        ["tx_left_wrist_device", "ty_left_wrist_device", "tz_left_wrist_device"]
    ] = np.nan

    hand_df.loc[hand_df["left_tracking_confidence"] < 0, ["tx_left_wrist_device", "ty_left_wrist_device", "tz_left_wrist_device"]] = np.nan
    
    hand_df.loc[
        (hand_df["tx_right_wrist_device"] == 0) & 
        (hand_df["ty_right_wrist_device"] == 0) & 
        (hand_df["tz_right_wrist_device"] == 0), 
        ["tx_right_wrist_device", "ty_right_wrist_device", "tz_right_wrist_device"]
    ] = np.nan

    hand_df.loc[hand_df["right_tracking_confidence"] < 0, ["tx_right_wrist_device", "ty_right_wrist_device", "tz_right_wrist_device"]] = np.nan

    # Interpolate small gaps
    #this is very important for hand because hand has a higher probability of having more missing data
    hand_df["tx_left_wrist_device"] = hand_df["tx_left_wrist_device"].interpolate(method="linear", limit=5)
    hand_df["ty_left_wrist_device"] = hand_df["ty_left_wrist_device"].interpolate(method="linear", limit=5)
    hand_df["tz_left_wrist_device"] = hand_df["tz_left_wrist_device"].interpolate(method="linear", limit=5)
    
    hand_df["tx_right_wrist_device"] = hand_df["tx_right_wrist_device"].interpolate(method="linear", limit=5)
    hand_df["ty_right_wrist_device"] = hand_df["ty_right_wrist_device"].interpolate(method="linear", limit=5)
    hand_df["tz_right_wrist_device"] = hand_df["tz_right_wrist_device"].interpolate(method="linear", limit=5)


    # # Smooth data using a moving average
    # we will be smoothing this again later but the double smoothing seems to help for finger movement classifications
    window_size = 5
    hand_df["tx_left_wrist_device"] = hand_df["tx_left_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    hand_df["ty_left_wrist_device"] = hand_df["ty_left_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    hand_df["tz_left_wrist_device"] = hand_df["tz_left_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    
    hand_df["tx_right_wrist_device"] = hand_df["tx_right_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    hand_df["ty_right_wrist_device"] = hand_df["ty_right_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()
    hand_df["tz_right_wrist_device"] = hand_df["tz_right_wrist_device"].rolling(window=window_size, center=True, min_periods=1).mean()

    return head_df, hand_df

if __name__ == "__main__":
    
    raw_hand_df = pd.concat([pd.read_csv(f) for f in HAND_FILES], ignore_index=True)

    print("Original Missing Values in Hand Data:")
    print(raw_hand_df[["tx_left_wrist_device", "tx_right_wrist_device"]].isna().sum())
    head_df, hand_df = load_and_preprocess_data()

    # Check for missing values in left and right wrist positions
    print("Missing values count:")
    print(hand_df[["tx_left_wrist_device", "tx_right_wrist_device"]].isna().sum())

    # Print first few rows to confirm data exists
    print("\nSample of hand data:")
    print(hand_df[["time_s", "tx_left_wrist_device", "tx_right_wrist_device"]].head())
    print("\columns of head data:")
    print(head_df.columns)