import json
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy
from scipy.spatial.transform import Slerp

# --- Constants ---
OBJECTS_OF_INTEREST = {
    "Bottom2A", "Bottom2B", "Bottom1A", "Bottom1B",
    "BH-leg1", "TH-leg1", "TH-Leg2", "BH-leg2",
    "BasketY", "BasketU", "BasketP"
}
FRAME_RATE = 13.5  # Frames per second

# --- Helper Functions ---
def quaternion_angle(q1, q2):
    """Angle between two quaternions in degrees."""
    d = np.abs(np.dot(q1, q2))
    d = min(max(d, 0.0), 1.0)
    return math.degrees(2 * math.acos(d))

# def slerp(q1, q2, alpha):
#     """Spherical linear interpolation for quaternions."""
#     q1 = np.array(q1)
#     q2 = np.array(q2)
#     dot = np.dot(q1, q2)
#     if dot < 0.0:
#         q2 = -q2
#         dot = -dot
#     dot = np.clip(dot, -1.0, 1.0)
#     theta = np.arccos(dot) * alpha
#     q_rel = q2 - q1 * dot
#     q_rel /= np.linalg.norm(q_rel)
#     return q1 * np.cos(theta) + q_rel * np.sin(theta)

# --- Data Loading ---
def load_json_rotation_data(json_file):
    """Load and extract rotation & position data from JSON."""
    with open(json_file, "r") as f:
        data = json.load(f)
    all_frames = []
    for chunk in data:
        if "frames" in chunk:
            all_frames.extend(chunk["frames"])
    return all_frames

# # --- Frozen Frame Detection ---
# def detect_frozen_sequences(frames, object_name, min_static_frames=3, noise_threshold=1e-3):
#     """Detect sequences where an object's state is unnaturally static."""
#     frozen_ranges = []
#     start_idx = None
#     prev_pos = None
#     prev_rot = None
    
#     for i, frame in enumerate(frames):
#         obj = next((o for o in frame["objects"] if o["name"] == object_name), None)
#         if not obj:
#             continue
        
#         current_pos = np.array([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
#         current_rot = np.array([obj["rotation"]["x"], obj["rotation"]["y"], obj["rotation"]["z"], obj["rotation"]["w"]])
        
#         pos_changed = prev_pos is not None and np.linalg.norm(current_pos - prev_pos) > noise_threshold
#         rot_changed = prev_rot is not None and quaternion_angle(current_rot, prev_rot) > noise_threshold
        
#         if not pos_changed and not rot_changed:
#             if start_idx is None:
#                 start_idx = i
#         else:
#             if start_idx is not None and (i - start_idx) >= min_static_frames:
#                 frozen_ranges.append((start_idx, i-1))
#             start_idx = None
        
#         prev_pos = current_pos
#         prev_rot = current_rot
    
#     if start_idx is not None and (len(frames) - start_idx) >= min_static_frames:
#         frozen_ranges.append((start_idx, len(frames)-1))
    
#     return frozen_ranges

# # --- Gap Interpolation ---
# def interpolate_gaps(frames, object_name, max_gap_frames=20):
#     """Fill gaps (including frozen sequences) with smooth interpolation."""
#     # Find all frames where the object exists
#     obj_frames = [i for i, frame in enumerate(frames) 
#                   if any(o["name"] == object_name for o in frame["objects"])]
    
#     for i in range(1, len(obj_frames)):
#         gap_size = obj_frames[i] - obj_frames[i-1] - 1
#         if 1 <= gap_size <= max_gap_frames:
#             # Get bounding frames
#             prev_frame_idx = obj_frames[i-1]
#             next_frame_idx = obj_frames[i]
#             prev_obj = next(o for o in frames[prev_frame_idx]["objects"] if o["name"] == object_name)
#             next_obj = next(o for o in frames[next_frame_idx]["objects"] if o["name"] == object_name)
            
#             # Interpolate
#             for gap_idx in range(prev_frame_idx + 1, next_frame_idx):
#                 alpha = (gap_idx - prev_frame_idx) / (next_frame_idx - prev_frame_idx)
#                 # Position (linear)
#                 interp_pos = (1 - alpha) * np.array([prev_obj["position"]["x"], prev_obj["position"]["y"], prev_obj["position"]["z"]]) + \
#                              alpha * np.array([next_obj["position"]["x"], next_obj["position"]["y"], next_obj["position"]["z"]])
#                 # Rotation (SLERP)
#                 interp_rot = slerp(
#                     [prev_obj["rotation"]["x"], prev_obj["rotation"]["y"], prev_obj["rotation"]["z"], prev_obj["rotation"]["w"]],
#                     [next_obj["rotation"]["x"], next_obj["rotation"]["y"], next_obj["rotation"]["z"], next_obj["rotation"]["w"]],
#                     alpha
#                 )
#                 # Insert interpolated object
#                 frames[gap_idx]["objects"].append({
#                     "name": object_name,
#                     "position": {"x": interp_pos[0], "y": interp_pos[1], "z": interp_pos[2]},
#                     "rotation": {"x": interp_rot[0], "y": interp_rot[1], "z": interp_rot[2], "w": interp_rot[3]}
#                 })

# --- Main Cleaning Pipeline ---
# def clean_frames_pipeline(raw_frames):
#     """Full cleaning: frozen frames, gaps, and jump interpolation."""
#     cleaned_frames = copy.deepcopy(raw_frames)
    
#     for obj_name in OBJECTS_OF_INTEREST:
#         # Step 1: Detect and delete frozen sequences
#         frozen_ranges = detect_frozen_sequences(cleaned_frames, obj_name)
#         for start, end in frozen_ranges:
#             for i in range(start, end + 1):
#                 cleaned_frames[i]["objects"] = [o for o in cleaned_frames[i]["objects"] if o["name"] != obj_name]
        
#         # Step 2: Interpolate gaps (including frozen sequences)
#         interpolate_gaps(cleaned_frames, obj_name)
    
#     return cleaned_frames
# --- Disabled Cleaning Pipeline ---
def clean_frames_pipeline(raw_frames):
    """Bypass all cleaning steps"""
    return copy.deepcopy(raw_frames)  # Return raw data without modifications

# --- Simplified Computation ---
def compute_rotations(frames):
    """Compute rotations from raw data"""
    object_data = {}
    first_movement_frame = None
    
    for frame_idx in range(len(frames)):
        current_frame = frames[frame_idx]
        
        for obj in current_frame["objects"]:
            obj_name = obj["name"]
            if obj_name not in OBJECTS_OF_INTEREST:
                continue

            # Store raw rotation data
            rot = np.array([
                obj["rotation"]["x"], 
                obj["rotation"]["y"], 
                obj["rotation"]["z"], 
                obj["rotation"]["w"]
            ])
            
            if obj_name not in object_data:
                object_data[obj_name] = {
                    "frames": [],
                    "quaternions": []
                }
            
            object_data[obj_name]["frames"].append(frame_idx)
            object_data[obj_name]["quaternions"].append(rot.tolist())
    
    # First movement frame is always 0 since we're not calculating velocity
    return object_data, 0

# --- Raw Data Trigger Detection ---
def detect_triggers(object_data, obj_name, rotation_threshold_deg=90, return_tolerance_deg=15):
    """Detect triggers using raw rotation data and return frame indices of triggers"""
    if obj_name not in object_data:
        return None, []
    
    return_window_seconds = 3.0
    quaternions = np.array(object_data[obj_name]["quaternions"])
    frame_indices = np.array(object_data[obj_name]["frames"])
    time_seconds = frame_indices / FRAME_RATE
    triggers = np.zeros_like(time_seconds, dtype=int)
    trigger_frames = []
    
    for i in range(1, len(time_seconds)):
        # Check all previous frames within 1 second
        for j in range(max(0, i - int(FRAME_RATE * return_window_seconds)), i):
            angle_diff = quaternion_angle(quaternions[i], quaternions[j])
            if angle_diff < return_tolerance_deg:
                # Check if we had a large rotation between these frames
                max_rotation = max(quaternion_angle(quaternions[j], quaternions[k]) 
                                 for k in range(j, i+1))
                if max_rotation >= rotation_threshold_deg:
                    triggers[i] = 1
                    trigger_frames.append(frame_indices[i])
                    break
                    
    return {
        "time": time_seconds,
        "trigger": triggers
    }, trigger_frames

# --- Simplified Visualization ---
def plot_triggers(triggers, obj_name, save_dir):
    """Plot quaternion-based triggers."""
    plt.figure(figsize=(12, 6))
    plt.step(triggers["time"], triggers["trigger"], where="post", color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Trigger State")
    plt.title(f"Rotation Triggers for {obj_name}")
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{obj_name}_triggers.png"))
    plt.close()

def write_trigger_frames(trigger_data, participant, output_dir):
    """Write trigger frame indices to a text file"""
    filename = os.path.join(output_dir, f"{participant}_framestamps.txt")
    with open(filename, "w") as f:
        for obj_name, frames in trigger_data.items():
            if frames:  # Only write if there are triggers
                f.write(f"{obj_name}: {', '.join(map(str, frames))}\n")

# --- Updated Main Pipeline ---
def process_participant(participant, input_base=".", output_base="output"):
    """Process data for a single participant"""
    json_file = os.path.join(input_base, participant, f"{participant}.json")
    output_dir = os.path.join(output_base, participant, "A2_plots_V5")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and clean data
    raw_frames = load_json_rotation_data(json_file)
    cleaned_frames = clean_frames_pipeline(raw_frames)
    
    # Compute rotations and detect triggers
    object_data, first_movement_frame = compute_rotations(cleaned_frames)
    
    trigger_data = {}  # To store trigger frames for each object
    
    # Generate plots and collect trigger frames for each object
    for obj_name in OBJECTS_OF_INTEREST:
        if obj_name in object_data:
            triggers, trigger_frames = detect_triggers(
                object_data, obj_name,
                rotation_threshold_deg=60,
                return_tolerance_deg=15
            )
            plot_triggers(triggers, obj_name, output_dir)
            trigger_data[obj_name] = trigger_frames
    
    # Write trigger frames to file
    write_trigger_frames(trigger_data, participant, output_dir)

# --- Execution ---
if __name__ == "__main__":
    # Process all participants from P1 to P11, excluding P2
    participants = [f"P{i}" for i in range(1, 12) if i != 2]
    
    for participant in participants:
        try:
            print(f"Processing {participant}...")
            process_participant(participant)
            print(f"Completed {participant}")
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")