# utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data  
from config import VALIDATED_PLOT_OUTPUT_FILE


def compute_velocity(df, time_col, x_col, y_col, z_col, velocity_col):
    """
    Compute velocity magnitude from position data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the time column.
        x_col (str): X position column.
        y_col (str): Y position column.
        z_col (str): Z position column.
        velocity_col (str): Name of the velocity column to store results.

    Returns:
        pd.DataFrame: Updated DataFrame with computed velocity.
    """
    if time_col != "time_s":
        raise ValueError(f"Unexpected time column: {time_col}. Expected 'time_s'.")

    dt = np.diff(df[time_col], prepend=df[time_col].iloc[0])
    dt[dt == 0] = np.nan  # Prevent division by zero

    vx = np.diff(df[x_col], prepend=df[x_col].iloc[0]) / dt
    vy = np.diff(df[y_col], prepend=df[y_col].iloc[0]) / dt
    vz = np.diff(df[z_col], prepend=df[z_col].iloc[0]) / dt

    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    velocity[np.isnan(velocity)] = 0
    velocity[np.isinf(velocity)] = 0

    df[velocity_col] = velocity  # Store in specified column
    return df

# incase we want to switch back to velocity
# def compute_angular_velocity(head_df, window_size=5):
#     """
#     Compute angular velocity from smoothed quaternion derivatives.
    
#     Args:
#         head_df (pd.DataFrame): DataFrame containing quaternion (qx, qy, qz, qw) and time_s.
#         window_size (int): Size of the moving average window for quaternion smoothing.

#     Returns:
#         pd.DataFrame: Updated DataFrame with computed angular velocity components and magnitude.
#     """
#     if "time_s" not in head_df.columns:
#         raise ValueError("Expected 'time_s' column in head_df but not found!")

#     head_df["qx_world_device"] = head_df["qx_world_device"].rolling(window=window_size, center=True, min_periods=1).median()
#     head_df["qy_world_device"] = head_df["qy_world_device"].rolling(window=window_size, center=True, min_periods=1).median()
#     head_df["qz_world_device"] = head_df["qz_world_device"].rolling(window=window_size, center=True, min_periods=1).median()
#     head_df["qw_world_device"] = head_df["qw_world_device"].rolling(window=window_size, center=True, min_periods=1).median()


#     dt = np.diff(head_df["time_s"], prepend=head_df["time_s"].iloc[0])
#     dt[dt < 1e-6] = np.nan  # Ignore very small time steps

    
#     dq_w = np.diff(head_df["qw_world_device"], prepend=head_df["qw_world_device"].iloc[0]) / dt
#     dq_x = np.diff(head_df["qx_world_device"], prepend=head_df["qx_world_device"].iloc[0]) / dt
#     dq_y = np.diff(head_df["qy_world_device"], prepend=head_df["qy_world_device"].iloc[0]) / dt
#     dq_z = np.diff(head_df["qz_world_device"], prepend=head_df["qz_world_device"].iloc[0]) / dt

#     # Compute angular velocity components
#     wx = 2 * (head_df["qw_world_device"] * dq_x + head_df["qx_world_device"] * dq_w +
#               head_df["qy_world_device"] * dq_z - head_df["qz_world_device"] * dq_y)

#     wy = 2 * (head_df["qw_world_device"] * dq_y - head_df["qx_world_device"] * dq_z +
#               head_df["qy_world_device"] * dq_w + head_df["qz_world_device"] * dq_x)

#     wz = 2 * (head_df["qw_world_device"] * dq_z + head_df["qx_world_device"] * dq_y -
#               head_df["qy_world_device"] * dq_x + head_df["qz_world_device"] * dq_w)

#     # Store computed angular velocities in DataFrame
#     head_df["computed_angular_velocity_x"] = wx
#     head_df["computed_angular_velocity_y"] = wy
#     head_df["computed_angular_velocity_z"] = wz
    
#     # Compute angular velocity magnitude
#     head_df["angular_velocity_magnitude"] = np.sqrt(wx**2 + wy**2 + wz**2)
    

#     # Ensure "time_s" is not dropped
#     return head_df


def compute_angular_difference_over_time(head_df, dt_window=1.0):
    """
    Compute angular difference (net rotation) over a specified time interval (dt_window)
    using raw quaternion data. For each time step, the function finds the quaternion
    from dt_window seconds earlier and computes the rotation difference between that
    quaternion and the current one, similar to Unity's Quaternion.Angle.
    
    Args:
        head_df (pd.DataFrame): DataFrame containing quaternion components 
            (qx_world_device, qy_world_device, qz_world_device, qw_world_device) and time_s.
        dt_window (float): Time interval (in seconds) over which to compute the angular difference.
        
    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'angular_difference' representing
                      the net rotation (in degrees) over the specified time window.
    """
    import numpy as np

    times = head_df['time_s'].values
    angular_diff = np.full(len(head_df), np.nan)
    # basically a dot product
    def quaternion_angle(q1, q2):
        """
        Calculate the angle (in degrees) between two quaternions.
        Normalizes the quaternions and computes:
        
            angle = 2 * arccos(|dot(q1, q2)|)
            
        Args:
            q1, q2: Quaternions in [w, x, y, z] order.
            
        Returns:
            Angle in degrees.
        """
        norm1 = np.linalg.norm(q1)
        norm2 = np.linalg.norm(q2)
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return np.nan
        q1_norm = q1 / norm1
        q2_norm = q2 / norm2
        dot = np.clip(np.abs(np.dot(q1_norm, q2_norm)), -1.0, 1.0)
        # Calculate angle in radians and convert to degrees
        angle_rad = 2 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    # Loop over each row and compute the net rotation over the dt_window interval
    for i, t in enumerate(times):
        target_time = t - dt_window
        # Find the index of the last frame with time <= target_time
        j = np.searchsorted(times, target_time, side='right') - 1
        if j < 0:
            continue  # Not enough earlier data to compute a difference
        q_prev = np.array([
            head_df['qw_world_device'].iloc[j],
            head_df['qx_world_device'].iloc[j],
            head_df['qy_world_device'].iloc[j],
            head_df['qz_world_device'].iloc[j]
        ])
        q_curr = np.array([
            head_df['qw_world_device'].iloc[i],
            head_df['qx_world_device'].iloc[i],
            head_df['qy_world_device'].iloc[i],
            head_df['qz_world_device'].iloc[i]
        ])
        angular_diff[i] = quaternion_angle(q_prev, q_curr)
    
    head_df["angular_difference"] = angular_diff
    return head_df

# this is just to make things look pretty and maybe do a comparison of when the tiggers are 
# and when the instructions are
def plot_validated_triggers(merged_df, all_triggers, validated_triggers, instruction_times):
    """Plot comparing automatic vs validated triggers using y-axis 0-1 for flag display"""
    plt.figure(figsize=(14, 5))
    
    # Plot triggers as vertical lines (flags)
    for trigger in all_triggers:
        if trigger in validated_triggers:
            color = 'limegreen'
            linewidth = 2.5
            alpha = 0.9
            label = 'Validated Trigger'
        else:
            color = 'red'
            linewidth = 1.5
            alpha = 0.4
            label = 'Auto-detected Trigger'
        
        plt.axvline(x=trigger, color=color, linestyle='-', 
                   linewidth=linewidth, alpha=alpha,
                   label=label if trigger == all_triggers[0] else "")
    
    # Plot instructions as vertical lines
    for instr in instruction_times:
        plt.axvline(x=instr, color='blue', linestyle=':', alpha=0.7,
                   label='Instruction' if instr == instruction_times[0] else "")
    
    # Add labels for validated triggers
    for i, trigger in enumerate(validated_triggers, 1):
        plt.text(trigger, 0.97, f"✓{i}", 
                 color='limegreen', ha='center', va='top', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Custom legend (head velocity entry removed)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='limegreen', linewidth=2, label='Validated Triggers'),
        Line2D([0], [0], color='red', alpha=0.7, label='Auto-detected Triggers'),
        Line2D([0], [0], color='blue', linestyle=':', label='Instructions')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Trigger Validation Results")
    plt.xlabel("Time (seconds)")
    plt.ylabel("")  # No y-axis label needed
    plt.ylim(0, 1)  # Set y-axis to just 0-1
    plt.grid(alpha=0.1)
    plt.tight_layout()
    # Save the plot to file as specified in config
    plt.savefig(VALIDATED_PLOT_OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {VALIDATED_PLOT_OUTPUT_FILE}")
    plt.show()


def main():
    # Load Data
    head_df, _ = load_and_preprocess_data()

    print("Columns in head_df:", head_df.columns)

    # Plot comparison for X-axis Angular Velocity
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(head_df["time_s"], wx, label="Computed Angular Velocity X", color="red", alpha=0.7)
    plt.plot(head_df["time_s"], head_df["angular_velocity_x_device"], label="Provided Angular Velocity X", color="blue", linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity X (rad/s)")
    plt.legend()

    # Plot comparison for Y-axis Angular Velocity
    plt.subplot(3, 1, 2)
    plt.plot(head_df["time_s"], wy, label="Computed Angular Velocity Y", color="red", alpha=0.7)
    plt.plot(head_df["time_s"], head_df["angular_velocity_y_device"], label="Provided Angular Velocity Y", color="blue", linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity Y (rad/s)")
    plt.legend()

    # Plot comparison for Z-axis Angular Velocity
    plt.subplot(3, 1, 3)
    plt.plot(head_df["time_s"], wz, label="Computed Angular Velocity Z", color="red", alpha=0.7)
    plt.plot(head_df["time_s"], head_df["angular_velocity_z_device"], label="Provided Angular Velocity Z", color="blue", linestyle="dashed", alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity Z (rad/s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print Summary Statistics
    print(head_df[["time_s", "computed_angular_velocity_x", "computed_angular_velocity_y", "computed_angular_velocity_z",
                   "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]].describe())



    
if __name__ == "__main__":
    main()
