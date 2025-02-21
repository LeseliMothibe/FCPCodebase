import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load Data
def load_data():
    df_rmppo = pd.read_csv("rmppo_metrics.csv")
    df_vanilla = pd.read_csv("vanilla_metrics.csv")
    df_stopping = pd.read_csv("stopping_metrics.csv")

    # Convert column names to lowercase and strip spaces
    for df in [df_rmppo, df_vanilla, df_stopping]:
        df.columns = df.columns.str.strip().str.lower()

    return df_rmppo, df_vanilla, df_stopping

# Function to calculate intervals between falls
def calculate_fall_intervals(df):
    fall_indices = df[df["falls"] == 1].index
    intervals = np.diff(fall_indices)
    return intervals

# Function to calculate statistical metrics
def calculate_metrics(df, algorithm_name):
    metrics = {}

    # Fall Percentage
    total_episodes = len(df)
    total_falls = df["falls"].sum()
    metrics["Fall Percentage"] = (total_falls / total_episodes) * 100

    # Mean and Median Interval Between Falls
    fall_intervals = calculate_fall_intervals(df)
    if len(fall_intervals) > 0:
        metrics["Mean Interval Between Falls"] = np.mean(fall_intervals)
        metrics["Median Interval Between Falls"] = np.median(fall_intervals)
        metrics["Max Interval Between Falls"] = np.max(fall_intervals)
        metrics["Min Interval Between Falls"] = np.min(fall_intervals)
        metrics["Standard Deviation of Intervals"] = np.std(fall_intervals)
    else:
        metrics["Mean Interval Between Falls"] = np.nan
        metrics["Median Interval Between Falls"] = np.nan
        metrics["Max Interval Between Falls"] = np.nan
        metrics["Min Interval Between Falls"] = np.nan
        metrics["Standard Deviation of Intervals"] = np.nan

    # Speed Metrics

    if algorithm_name != "Stopping":
        metrics["Mean Speed"] = df["speed"].mean()
        metrics["Median Speed"] = df["speed"].median()
        metrics["Max Speed"] = df["speed"].max()
        metrics["Min Speed"] = df["speed"].min()
        metrics["Standard Deviation of Speed"] = df["speed"].std()
        metrics["Skewness of Speed"] = skew(df["speed"])
        metrics["Kurtosis of Speed"] = kurtosis(df["speed"])

    # Deviation Metrics
    metrics["Mean Deviation"] = df["average deviation"].mean()
    metrics["Median Deviation"] = df["average deviation"].median()
    metrics["Max Deviation"] = df["average deviation"].max()
    metrics["Min Deviation"] = df["average deviation"].min()
    metrics["Standard Deviation of Deviation"] = df["average deviation"].std()
    metrics["Skewness of Deviation"] = skew(df["average deviation"])
    metrics["Kurtosis of Deviation"] = kurtosis(df["average deviation"])

    # Stopping Distance Metrics (if applicable)
    if "stopping_distance" in df.columns:
        metrics["Mean Stopping Distance"] = df["stopping_distance"].mean()
        metrics["Median Stopping Distance"] = df["stopping_distance"].median()
        metrics["Max Stopping Distance"] = df["stopping_distance"].max()
        metrics["Min Stopping Distance"] = df["stopping_distance"].min()
        metrics["Standard Deviation of Stopping Distance"] = df["stopping_distance"].std()
        metrics["Skewness of Stopping Distance"] = skew(df["stopping_distance"])
        metrics["Kurtosis of Stopping Distance"] = kurtosis(df["stopping_distance"])
    else:
        metrics["Mean Stopping Distance"] = np.nan
        metrics["Median Stopping Distance"] = np.nan
        metrics["Max Stopping Distance"] = np.nan
        metrics["Min Stopping Distance"] = np.nan
        metrics["Standard Deviation of Stopping Distance"] = np.nan
        metrics["Skewness of Stopping Distance"] = np.nan
        metrics["Kurtosis of Stopping Distance"] = np.nan

    # Add algorithm name to metrics
    metrics["Algorithm"] = algorithm_name

    return metrics

# Main function to perform statistical analysis
def perform_statistical_analysis(df_rmppo, df_vanilla, df_stopping):
    # Calculate metrics for each algorithm
    metrics_rmppo = calculate_metrics(df_rmppo, "RMPPO")
    metrics_vanilla = calculate_metrics(df_vanilla, "Vanilla PPO")
    metrics_stopping = calculate_metrics(df_stopping, "Stopping")

    # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame([metrics_rmppo, metrics_vanilla, metrics_stopping])

    # Save metrics to a CSV file
    metrics_df.to_csv("statistical_analysis.csv", index=False)
    print("Statistical analysis saved to 'statistical_analysis.csv'.")

    return metrics_df

if __name__ == "__main__":
    # Load data
    df_rmppo, df_vanilla, df_stopping = load_data()

    # Perform statistical analysis
    metrics_df = perform_statistical_analysis(df_rmppo, df_vanilla, df_stopping)

    # Print the results
    print(metrics_df)