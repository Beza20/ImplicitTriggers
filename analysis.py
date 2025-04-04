import pandas as pd
import matplotlib.pyplot as plt
import os
from config import HEAD_VELOCITY_THRESHOLD, LEFT_HAND_VELOCITY_THRESHOLD, RIGHT_HAND_VELOCITY_THRESHOLD, INSTRUCTIONS_TIMESTAMPS, TRIGGER_TIMESTAMPS_FILE, PLOT_OUTPUT_FILE, VALIDATED_TRIGGERS_FILE 
from utils import compute_angular_difference_over_time, compute_velocity, plot_validated_triggers

# Function to convert instruction timestamps from ['m:ss', 'm:ss', ...] to total seconds so that
# we can plot them on the same graph as the trigger timestamps
def convert_instruction_timestamps(instruction_timestamps):
    """Convert ['m:ss', 'm:ss', ...] format to total seconds"""
    converted = []
    for ts in instruction_timestamps:
        if ':' in ts:
            minutes, seconds = map(float, ts.split(':'))
            converted.append(minutes * 60 + seconds)
        else:
            # Handle case where timestamp might already be in seconds
            converted.append(float(ts))
    return converted

# this function takes the head and hand dataframes and computes the angular difference and velocity
# it also plots the results and saves them to a file
# it also saves the trigger timestamps to a file
def analyze_and_plot(head_df, hand_df):
    # Compute Angular difference (if missing)
    if "angular_difference" not in head_df.columns:
        print("Computing angular difference from quaternions...")
        head_df = compute_angular_difference_over_time(head_df)

    # Compute Hand Velocity using `compute_velocity`
    hand_df = compute_velocity(hand_df, "time_s", "tx_left_wrist_device", "ty_left_wrist_device", "tz_left_wrist_device", "velocity_left")
    hand_df = compute_velocity(hand_df, "time_s", "tx_right_wrist_device", "ty_right_wrist_device", "tz_right_wrist_device", "velocity_right")

    window_size_seconds = 0.2  # 0.2-second rolling window

    # Convert to pandas Series with time index for proper time-based rolling
    # Hand data
    hand_velocity_left = pd.Series(
        hand_df["velocity_left"].values,
        index=pd.to_timedelta(hand_df["time_s"], unit='s')
    )
    hand_velocity_right = pd.Series(
        hand_df["velocity_right"].values,
        index=pd.to_timedelta(hand_df["time_s"], unit='s')
    )
    
    # Head data
    head_angular_velocity = pd.Series(
        head_df["angular_difference"].values,
        index=pd.to_timedelta(head_df["time_s"], unit='s')
    )

    # Apply precise time-based rolling for all our data_points
    hand_df["velocity_left_smooth"] = hand_velocity_left.rolling(f'{window_size_seconds}s', min_periods=1).mean().values
    hand_df["velocity_right_smooth"] = hand_velocity_right.rolling(f'{window_size_seconds}s', min_periods=1).mean().values
    head_df["angular_velocity_magnitude_smooth"] = head_angular_velocity.rolling(f'{window_size_seconds}s', min_periods=1).mean().values

    # Define Movement Flags
    head_df["movement_flag"] = (head_df["angular_velocity_magnitude_smooth"] > HEAD_VELOCITY_THRESHOLD).astype(int)
    hand_df["movement_flag_left"] = (hand_df["velocity_left_smooth"] > LEFT_HAND_VELOCITY_THRESHOLD).astype(int)
    hand_df["movement_flag_right"] = (hand_df["velocity_right_smooth"] > RIGHT_HAND_VELOCITY_THRESHOLD).astype(int)

    # NOR Operation: Hands are still only when BOTH are still
    hand_df["movement_flag_nor"] = 1 - (hand_df["movement_flag_left"] | hand_df["movement_flag_right"])

    # Merge DataFrames
    merged_df = pd.merge_asof(
        head_df[["time_s", "movement_flag", "angular_velocity_magnitude_smooth"]],  # Fixed typo here
        hand_df[["time_s", "movement_flag_left", "movement_flag_right", "movement_flag_nor"]],
        on="time_s"
    )

    # Identify when head moves but hands do not
    merged_df["head_no_hand_movement"] = ((merged_df["movement_flag"] == 1) & (merged_df["movement_flag_nor"] == 1)).astype(int)

    # Implement Time-Based Triggering
    # we want the trigger to go off if this condition is met for 2 seconds continuously 
    merged_df["head_no_hand_movement_trigger"] = 0
    trigger_duration_threshold = 2.0  # Require 2 seconds of continuous movement
    current_trigger_time = 0.0

    for i in range(1, len(merged_df)):
        dt = merged_df.loc[i, "time_s"] - merged_df.loc[i - 1, "time_s"]  # Time difference in seconds
    
        if merged_df.loc[i, "head_no_hand_movement"] == 1:
            current_trigger_time += dt
            if current_trigger_time >= trigger_duration_threshold:
                merged_df.loc[i, "head_no_hand_movement_trigger"] = 1
                current_trigger_time = 0.0  # Reset after trigger
        else:
            current_trigger_time = 0.0  # Reset if condition breaks

    # Save Trigger Timestamps
    trigger_times = merged_df.loc[merged_df["head_no_hand_movement_trigger"] == 1, "time_s"].tolist()
    print("\nTrigger happened at these timestamps (seconds):")
    print(trigger_times)

    if not os.path.exists(TRIGGER_TIMESTAMPS_FILE):
        open(TRIGGER_TIMESTAMPS_FILE, "w").close()

    with open(TRIGGER_TIMESTAMPS_FILE, "a") as f:
        for ts in trigger_times:
            f.write(f"{ts}\n")
            
        # add validation plotting step
        # validated trigger files are for when a person has looked at the clips for the trigger points and confirmed them
        # this is a manual process and the validated trigger file is a list of timestamps in seconds
        # if this is the first time you are running it and there is no validated trigger file, it will just show the basic plot
        # if there is a validated trigger file, it will show the validated trigger points on the plot
    if os.path.exists(VALIDATED_TRIGGERS_FILE):
        with open(VALIDATED_TRIGGERS_FILE) as f:
            validated_triggers = [float(line.strip()) for line in f if line.strip()]
        
        instruction_times = convert_instruction_timestamps(INSTRUCTIONS_TIMESTAMPS)
        plot_validated_triggers(
            merged_df=merged_df,
            all_triggers=trigger_times,  
            validated_triggers=validated_triggers,
            instruction_times=instruction_times
        )
    else:
        print("No validated_triggers.txt found - showing basic plot")
        plt.figure(figsize=(12, 4))
        plt.step(merged_df["time_s"], merged_df["head_no_hand_movement_trigger"], where="post", label="Trigger")
        plt.show()

    print(f"\nTrigger timestamps saved to {TRIGGER_TIMESTAMPS_FILE}")

    # ✅ Plot Triggered Events with Instruction Timestamps
    plt.figure(figsize=(12, 6))
    
    # Main trigger plot
    plt.step(merged_df["time_s"], merged_df["head_no_hand_movement_trigger"], where="post", 
             label="Trigger Activated", color="red", alpha=0.7)
    
    # Convert and plot instruction timestamps
    if INSTRUCTIONS_TIMESTAMPS:
        instruction_seconds = convert_instruction_timestamps(INSTRUCTIONS_TIMESTAMPS)
        for i, instruction_time in enumerate(instruction_seconds):
            plt.axvline(x=instruction_time, color='blue', linestyle='--', alpha=0.5,
                       label='Instructions' if i == 0 else "")
            # Optional: Add text labels for each instruction
            plt.text(instruction_time, 0.95, f"Instr {i+1}\n{INSTRUCTIONS_TIMESTAMPS[i]}", 
                    color='blue', ha='center', va='top', rotation=90, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Trigger Flag (1 = Activated)")
    plt.title("Head Movement Triggers with Instruction Timestamps")
    plt.ylim(-0.05, 1.05)  # Add slight padding
    
    # Smart legend handling
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot to file as specified in config
    plt.savefig(PLOT_OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {PLOT_OUTPUT_FILE}")
    
    # Optionally, display the plot
    plt.show()

    # ✅ Print Velocity Statistics
    print("Head Velocity Stats:")
    print(head_df["angular_difference"].describe())

    print("\nLeft Hand Velocity Stats:")
    print(hand_df["velocity_left"].describe())

    print("\nRight Hand Velocity Stats:")
    print(hand_df["velocity_right"].describe())

    print("\nTrigger Stats:")
    print(merged_df["head_no_hand_movement_trigger"].describe())
