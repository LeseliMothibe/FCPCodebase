import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

# Ensure results directory exists
output_dir = "results_plots"
os.makedirs(output_dir, exist_ok=True)

# Load Data
def load_data():
    df_rmppo = pd.read_csv("rmppo_metrics.csv")
    df_vanilla = pd.read_csv("vanilla_metrics.csv")
    df_stopping = pd.read_csv("stopping_metrics.csv")

    # Convert column names to lowercase and strip spaces
    for df in [df_rmppo, df_vanilla, df_stopping]:
        df.columns = df.columns.str.strip().str.lower()

    return df_rmppo, df_vanilla, df_stopping

# Helper function to save plots
def save_plot(fig, title):
    fig.savefig(os.path.join(output_dir, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# Principal Component Analysis (PCA) 3D Plot
def plot_pca_3d(data, title):
    num_features = min(3, data.shape[1])
    pca = PCA(n_components=num_features)
    transformed = pca.fit_transform(data)

    df_pca = pd.DataFrame(transformed, columns=[f'PC{i+1}' for i in range(num_features)])
    
    if num_features == 3:
        fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', title=title)
    elif num_features == 2:
        fig = px.scatter(df_pca, x='PC1', y='PC2', title=title)
    else:
        print(f"⚠ Not enough features for PCA in {title}. Skipping...")
        return
    
    fig.write_html(os.path.join(output_dir, f"{title}.html"))

# 3D Trajectory Plot
def plot_3d_trajectory(df, title):
    fig = px.line_3d(df, x="speed", y="average deviation", z="falls", title=title)
    fig.write_html(os.path.join(output_dir, f"{title}.html"))

# Rolling Mean Speed Plot
def rolling_mean_speed(df, label, window=10):
    df["rolling_speed"] = df["speed"].rolling(window=window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.index, df["rolling_speed"], label=label)
    ax.set_title(f"Rolling Mean Speed ({window} Episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    save_plot(fig, f"Rolling_Mean_Speed_{label}")

# Pair Plot (Correlation Analysis)
def plot_pairplot(df, label):
    sns.pairplot(df, diag_kind='kde')
    save_plot(plt, f"Pairplot_{label}")

# Cumulative Distribution Function (CDF) for Speed
def plot_cdf(df, label):
    speed_sorted = np.sort(df["speed"])
    cdf = np.arange(len(speed_sorted)) / float(len(speed_sorted))
    plt.plot(speed_sorted, cdf, label=label)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Speed")
    plt.legend()
    save_plot(plt, f"CDF_Speed_{label}")

# Heatmap for Speed Variations
def plot_speed_heatmap(df_rmppo, df_vanilla):
    min_length = min(len(df_rmppo["speed"]), len(df_vanilla["speed"]))
    df_speed = pd.DataFrame({
        "Episode": np.arange(1, min_length + 1),
        "RMPPO Speed": df_rmppo["speed"].iloc[:min_length].values,
        "Vanilla Speed": df_vanilla["speed"].iloc[:min_length].values,
    })
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_speed.set_index("Episode").T, cmap="coolwarm", annot=False)
    plt.title("Speed Variations Heatmap")
    plt.xlabel("Episode")
    plt.ylabel("Algorithm")
    save_plot(plt, "Heatmap_Speed_Variations")

# Quiver Plot for Direction Changes
def plot_quiver(df, title):
    plt.figure(figsize=(6, 6))
    x = np.arange(len(df))
    y = np.zeros_like(x)
    u = np.cos(np.radians(df["average deviation"]))
    v = np.sin(np.radians(df["average deviation"]))
    plt.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Direction Change")
    save_plot(plt, f"Quiver_{title.replace(' ', '_')}")

# Radial Histogram for Final Directions
def radial_histogram(df, title):
    angles = np.radians(df["average deviation"])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=30, color="blue", alpha=0.7)
    ax.set_title(title)
    save_plot(fig, f"Radial_Histogram_{title.replace(' ', '_')}")

# Fall Trend and Time Series Plots
def plot_fall_trend(df, label, window=200):
    df["rolling_falls"] = df["falls"].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["rolling_falls"], label=f"{label} (Rolling {window})", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Fall Frequency (Rolling Mean)")
    plt.title(f"Fall Frequency Trend - {label}")
    plt.legend()
    plt.grid(True)
    save_plot(plt, f"fall_trend_{label.replace(' ', '_')}")

def plot_fall_time_series(df, label):
    plt.figure(figsize=(12, 4))
    plt.scatter(df.index, df["falls"], alpha=0.5, marker='o', s=10, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Fall (1 = Yes, 0 = No)")
    plt.title(f"Binary Time Series of Falls - {label}")
    plt.yticks([0, 1], ["No Fall", "Fall"])
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    save_plot(plt, f"fall_series_{label.replace(' ', '_')}")

# Box Plots for Speed Comparison
def plot_speed_boxplot(df_rmppo, df_vanilla):
    df_rmppo["Algorithm"] = "RMPPO"
    df_vanilla["Algorithm"] = "Vanilla PPO"
    df_speed = pd.concat([df_rmppo[["speed", "Algorithm"]], df_vanilla[["speed", "Algorithm"]]])
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Algorithm", y="speed", data=df_speed)
    plt.title("Speed Box Plot")
    plt.ylabel("Speed (m/s)")
    plt.xlabel("Algorithm")
    save_plot(plt, "speed_box_plots")

# Speed Distribution Comparison
def plot_speed_distribution(df_rmppo, df_vanilla):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_rmppo["speed"], bins=30, color="blue", kde=True, label="RMPPO", stat="density")
    sns.histplot(df_vanilla["speed"], bins=30, color="red", kde=True, label="Vanilla PPO", stat="density", alpha=0.6)
    plt.title("Speed Distribution Comparison")
    plt.xlabel("Average Speed (m/s)")
    plt.ylabel("Frequency")
    plt.legend()
    save_plot(plt, "speed_comparison")

# Deviation During Running Comparison
def plot_deviation_running_comparison(df_rmppo, df_vanilla):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_rmppo["average deviation"], bins=30, color="blue", kde=True, label="RMPPO", stat="density")
    sns.histplot(df_vanilla["average deviation"], bins=30, color="red", kde=True, label="Vanilla PPO", stat="density", alpha=0.6)
    plt.title("Deviation During Running")
    plt.xlabel("Average Deviation (degrees)")
    plt.ylabel("Frequency")
    plt.legend()
    save_plot(plt, "deviation_running_comparison")

# Deviation During Stopping Distribution
def plot_deviation_stopping_distribution(df_stopping):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_stopping["average deviation"], bins=30, kde=True, color="green")
    plt.title("Deviation During Stopping")
    plt.xlabel("Average Deviation (degrees)")
    plt.ylabel("Frequency")
    save_plot(plt, "deviation_stopping_distribution")

# Fall Frequency Comparison
def plot_fall_frequency_comparison(df_rmppo, df_vanilla, df_stopping):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_rmppo["falls"], bins=30, color="blue", kde=True, label="RMPPO", stat="density")
    sns.histplot(df_vanilla["falls"], bins=30, color="red", kde=True, label="Vanilla PPO", stat="density", alpha=0.6)
    sns.histplot(df_stopping["falls"], bins=30, color="green", kde=True, label="Stopping", stat="density", alpha=0.6)
    plt.title("Fall Frequency Comparison")
    plt.xlabel("Number of Falls")
    plt.ylabel("Frequency")
    plt.legend()
    save_plot(plt, "fall_frequency_comparison")

# Stopping Distance Distribution
def plot_stopping_distance_distribution(df_stopping):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_stopping["stopping_distance"], bins=30, kde=True, color="purple")
    plt.title("Stopping Distance Distribution")
    plt.xlabel("Stopping Distance (meters)")
    plt.ylabel("Frequency")
    save_plot(plt, "stopping_distance_distribution")

# Bar graph to compare fall frequency (percentage)
def plot_fall_comparison_bar(df_rmppo, df_vanilla, df_stopping):
    # Calculate percentage of falls for each algorithm
    total_episodes_rmppo = len(df_rmppo)
    total_episodes_vanilla = len(df_vanilla)
    total_episodes_stopping = len(df_stopping)

    percentage_falls_rmppo = (df_rmppo["falls"].sum() / total_episodes_rmppo) * 100
    percentage_falls_vanilla = (df_vanilla["falls"].sum() / total_episodes_vanilla) * 100
    percentage_falls_stopping = (df_stopping["falls"].sum() / total_episodes_stopping) * 100

    # Data for the bar graph
    algorithms = ["Vanilla PPO", "RMPPO", "Stopping"]
    percentage_falls = [percentage_falls_vanilla, percentage_falls_rmppo, percentage_falls_stopping]

    # Create the bar graph
    plt.figure(figsize=(8, 6))
    bars = plt.bar(algorithms, percentage_falls, color=["red", "blue", "green"])

    # Add labels and title
    plt.title("Percentage of Falls Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Percentage of Falls (%)")

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}%", ha="center", va="bottom")

    # Save the plot
    save_plot(plt, "fall_comparison_bar_percentage")

# Box and Whisker Plot for Deviation
def plot_deviation_boxplot(df_rmppo, df_vanilla, df_stopping):
    # Add algorithm labels
    df_rmppo["Algorithm"] = "RMPPO"
    df_vanilla["Algorithm"] = "Vanilla PPO"
    df_stopping["Algorithm"] = "Stopping"

    # Combine data
    df_deviation = pd.concat([
        df_rmppo[["average deviation", "Algorithm"]],
        df_vanilla[["average deviation", "Algorithm"]],
        df_stopping[["average deviation", "Algorithm"]]
    ])

    # Create the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Algorithm", y="average deviation", data=df_deviation, palette=["blue", "red", "green"])
    plt.title("Deviation Comparison (Box and Whisker Plot)")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Deviation (degrees)")
    save_plot(plt, "deviation_boxplot")

# Main function to generate all plots
def generate_plots(df_rmppo, df_vanilla, df_stopping):
    plot_pca_3d(df_rmppo[["speed", "average deviation"]], "PCA 3D RMPPO")
    plot_pca_3d(df_vanilla[["speed", "average deviation"]], "PCA 3D Vanilla PPO")

    plot_3d_trajectory(df_rmppo, "3D Trajectory - RMPPO")
    plot_3d_trajectory(df_vanilla, "3D Trajectory - Vanilla PPO")

    rolling_mean_speed(df_rmppo, "RMPPO")
    rolling_mean_speed(df_vanilla, "Vanilla PPO")

    plot_pairplot(df_rmppo, "RMPPO")
    plot_pairplot(df_vanilla, "Vanilla_PPO")

    plot_cdf(df_rmppo, "RMPPO")
    plot_cdf(df_vanilla, "Vanilla PPO")

    plot_speed_heatmap(df_rmppo, df_vanilla)

    plot_quiver(df_rmppo, "Quiver Plot - RMPPO")
    plot_quiver(df_vanilla, "Quiver Plot - Vanilla PPO")

    radial_histogram(df_rmppo, "Final Direction - RMPPO")
    radial_histogram(df_vanilla, "Final Direction - Vanilla PPO")

    plot_fall_trend(df_rmppo, "RMPPO")
    plot_fall_trend(df_vanilla, "Vanilla PPO")
    plot_fall_trend(df_stopping, "Stopping")

    plot_fall_time_series(df_rmppo, "RMPPO")
    plot_fall_time_series(df_vanilla, "Vanilla PPO")
    plot_fall_time_series(df_stopping, "Stopping")

    plot_speed_boxplot(df_rmppo, df_vanilla)
    plot_speed_distribution(df_rmppo, df_vanilla)

    plot_deviation_running_comparison(df_rmppo, df_vanilla)
    plot_deviation_stopping_distribution(df_stopping)
    plot_fall_frequency_comparison(df_rmppo, df_vanilla, df_stopping)
    plot_stopping_distance_distribution(df_stopping)

    # Add the new bar graph for fall comparison (percentage)
    plot_fall_comparison_bar(df_rmppo, df_vanilla, df_stopping)

    # Add the new box and whisker plot for deviation
    plot_deviation_boxplot(df_rmppo, df_vanilla, df_stopping)

if __name__ == "__main__":
    df_rmppo, df_vanilla, df_stopping = load_data()
    generate_plots(df_rmppo, df_vanilla, df_stopping)
    print("All plots have been generated and saved in the 'results_plots' directory.")