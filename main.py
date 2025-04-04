# main.py

from data_loader import load_and_preprocess_data
from analysis import analyze_and_plot

if __name__ == "__main__":
    # Load and preprocess data
    head_df, hand_df = load_and_preprocess_data()

    # Perform analysis and generate plots
    analyze_and_plot(head_df, hand_df)