import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import analysis  # Your analysis module containing get_merged_df()

def read_validated_triggers(file_path):
    """
    Read a text file with validated trigger timestamps (one per line)
    and return a list of floats.
    """
    validated = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    validated.append(float(line))
                except ValueError:
                    continue
    return validated

def plot_validated_triggers(merged_df, validated_triggers, output_plot_file="validated_plot.png", tolerance=0.01):
    """
    Plots the trigger information using different colors.
    Validated triggers (from validated_triggers list) will be plotted in green,
    while unvalidated triggers remain red.
    
    merged_df: DataFrame that contains at least two columns:
       - "time_s": timestamps in seconds
       - "head_no_hand_movement_trigger": binary indicator (1=trigger, 0=no trigger)
    validated_triggers: List of manually approved trigger timestamps.
    tolerance: Float value for matching trigger timestamps (to account for floating-point differences)
    """
    # Create separate series for validated and unvalidated triggers
    validated_series = pd.Series(0, index=merged_df.index)
    unvalidated_series = merged_df["head_no_hand_movement_trigger"].copy()

    # Iterate over the DataFrame rows to mark validated triggers
    for idx, row in merged_df.iterrows():
        ts = row["time_s"]
        # Check if the timestamp is close to any validated trigger timestamp
        if any(np.isclose(ts, vt, atol=tolerance) for vt in validated_triggers):
            validated_series.at[idx] = 1
            unvalidated_series.at[idx] = 0  # Remove from unvalidated

    # Plotting the triggers
    plt.figure(figsize=(12, 6))
    plt.step(merged_df["time_s"], unvalidated_series, where="post", 
             label="Unvalidated Trigger", color="red", alpha=0.7)
    plt.step(merged_df["time_s"], validated_series, where="post", 
             label="Validated Trigger", color="green", alpha=0.7)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Trigger Flag (1 = Activated)")
    plt.title("Validated vs. Unvalidated Head Movement Triggers")
    plt.grid(True, alpha=0.3)
    
    # Manage the legend to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300, bbox_inches="tight")
    print(f"Validated plot saved to {output_plot_file}")
    plt.show()

if __name__ == "__main__":
    # Call your analysis script to get the merged DataFrame
    # (Assumes your analysis_script.py provides a get_merged_df() function)
    merged_df = analysis_script.get_merged_df()
    
    # Read the validated trigger timestamps from your file (create this manually after reviewing)
    validated_triggers = read_validated_triggers("../P5/P5_validated_trigger.txt")
    
    # Plot the triggers with manual validation highlighting
    plot_validated_triggers(merged_df, validated_triggers, output_plot_file="../P5/P5_validated_plot.png")
