# config.py
import re
import os

# Define the base directory where the search should begin
base_dir = "../P11"

# Lists to store file paths
HEAD_FILES = []
HAND_FILES = []

# open the base directory
for sub_dir in os.listdir(base_dir):
    full_subdir_path = os.path.join(base_dir, sub_dir)
    
    # Check if it's a directory
    if os.path.isdir(full_subdir_path):
        # Define expected subpaths
        head_file_path = os.path.join(full_subdir_path, "slam", "closed_loop_trajectory.csv")
        hand_file_path = os.path.join(full_subdir_path, "hand_tracking", "wrist_and_palm_poses.csv")
        
        # Check if the SLAM and hand_tracking  files exist and add them to the list
        if os.path.exists(head_file_path):
            HEAD_FILES.append(head_file_path)
        if os.path.exists(hand_file_path):
            HAND_FILES.append(hand_file_path)

# Print results for verification
print("HEAD_FILES =", HEAD_FILES)
print("HAND_FILES =", HAND_FILES)


# Thresholds
HEAD_VELOCITY_THRESHOLD = 10  # Threshold for head movement this is the angular degrees moved per second
RIGHT_HAND_VELOCITY_THRESHOLD = 0.05  # Threshold for right hand movement in m/s
LEFT_HAND_VELOCITY_THRESHOLD = 0.05 # Threshold for left hand movement in m/s

# Function to extract instruction timestamps from the transcription file
# This function reads a transcription file and extracts timestamps associated with instructions
# It returns a list of timestamps in the format "MM:SS"
# If the file is not found, it returns an empty list
# The function uses regular expressions to search for lines containing the word "instructions" or "next"
# and captures the timestamp at the beginning of those lines
def extract_instruction_timestamps(transcription_file):
    timestamps = []
    pattern = re.compile(r"(\d+:\d+):\s*.*?\b(instructions?|next)\b", re.IGNORECASE)

    try:
        with open(transcription_file, "r") as file:
            content = file.read()
            print(f"File content:\n{content}")  # Print file content for debugging
            
            for line in content.splitlines():
                match = pattern.search(line)
                if match:
                    timestamps.append(match.group(1))  # Extract the timestamp
    except FileNotFoundError:
        print(f"Warning: {transcription_file} not found. Using default timestamps.")

    return timestamps

# Define transcription file path
TRANSCRIPTION_FILE = "../P11/P11_transcription.txt"
TRIGGER_TIMESTAMPS_FILE = "../P11/P11_A1_trigger_timestampsVRmvDblSmthng.txt"
PLOT_OUTPUT_FILE = "../P11/P11_plotVRmvDblSmthng.png"
VALIDATED_PLOT_OUTPUT_FILE = "../P11/P11_validated_plotVRmvDblSmthng.png"
VALIDATED_TRIGGERS_FILE = "../P11/P11_validated_triggerVRmvDblSmthng.txt"

# Extracted timestamps
INSTRUCTIONS_TIMESTAMPS = extract_instruction_timestamps(TRANSCRIPTION_FILE)

# Print extracted timestamps for debugging
print(f"Extracted Instruction Timestamps: {INSTRUCTIONS_TIMESTAMPS}")