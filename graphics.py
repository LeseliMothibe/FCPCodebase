import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
from sklearn.decomposition import PCA

# Ensure results directory exists
output_dir = "results_plots"
os.makedirs(output_dir, exist_ok=True)

# Load Data
df_rmppo = pd.read_csv("rmppo_metrics.csv")
df_vanilla = pd.read_csv("vanilla_metrics.csv")
df_stopping = pd.read_csv("stopping_metrics.csv")

# Convert column names to lowercase and strip spaces
df_rmppo.columns = df_rmppo.columns.str.strip().str.lower()
df_vanilla.columns = df_vanilla.columns.str.strip().str.lower()
df_stopping.columns = df_stopping.columns.str.strip().str.lower()

# Helper function to save plots
def save_plot(fig, title):
    fig.savefig(os.path.join(output_dir, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.close()  # Close the current figure properly


# ==========================
# 1️⃣ Principal Component Analysis (PCA) 3D Plot
# ==========================
def plot_pca_3d(data, title):
    num_features = min(3, data.shape[1])  # Ensure we don't ask for more components than available
    pca = PCA(n_components=num_features)
    transformed = pca.fit_transform(data)

    # Create a DataFrame with transformed values
    df_pca = pd.DataFrame(transformed, columns=[f'PC{i+1}' for i in range(num_features)])
    
    # Generate appropriate 2D or 3D plot
    if num_features == 3:
        fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', title=title)
    elif num_features == 2:
        fig = px.scatter(df_pca, x='PC1', y='PC2', title=title)
    else:
        print(f"⚠ Not enough features for PCA in {title}. Skipping...")
        return
    
    fig.write_html(os.path.join(output_dir, f"{title}.html"))

plot_pca_3d(df_rmppo[["speed", "average deviation"]], "PCA 3D RMPPO")
plot_pca_3d(df_vanilla[["speed", "average deviation"]], "PCA 3D Vanilla PPO")

# ==========================
# 2️⃣ 3D Trajectory Plot
# ==========================
def plot_3d_trajectory(df, title):
    fig = px.line_3d(df, x="speed", y="average deviation", z="falls", title=title)
    fig.write_html(os.path.join(output_dir, f"{title}.html"))

plot_3d_trajectory(df_rmppo, "3D Trajectory - RMPPO")
plot_3d_trajectory(df_vanilla, "3D Trajectory - Vanilla PPO")

# ==========================
# 3️⃣ Rolling Mean Speed Plot
# ==========================
def rolling_mean_speed(df, label, window=10):
    df["rolling_speed"] = df["speed"].rolling(window=window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure explicitly
    ax.plot(df.index, df["rolling_speed"], label=label)
    ax.set_title(f"Rolling Mean Speed ({window} Episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    
    save_plot(fig, f"Rolling_Mean_Speed_{label}")  # Pass the figure object
    plt.close(fig)  # Ensure proper closure

# Call the function
rolling_mean_speed(df_rmppo, "RMPPO")
rolling_mean_speed(df_vanilla, "Vanilla PPO")

# ==========================
# 4️⃣ Pair Plot (Correlation Analysis)
# ==========================
sns.pairplot(df_rmppo, diag_kind='kde')
save_plot(plt, "Pairplot_RMPPO")

sns.pairplot(df_vanilla, diag_kind='kde')
save_plot(plt, "Pairplot_Vanilla_PPO")

# ==========================
# 5️⃣ Cumulative Distribution Function (CDF) for Speed
# ==========================
def plot_cdf(df, label):
    speed_sorted = np.sort(df["speed"])
    cdf = np.arange(len(speed_sorted)) / float(len(speed_sorted))
    plt.plot(speed_sorted, cdf, label=label)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Speed")
    plt.legend()
    save_plot(plt, f"CDF_Speed_{label}")

plot_cdf(df_rmppo, "RMPPO")
plot_cdf(df_vanilla, "Vanilla PPO")

# ==========================
# 6️⃣ Heatmap for Speed Variations
# ==========================
# Ensure all columns have the same length before creating the DataFrame
min_length = min(len(df_rmppo["speed"]), len(df_vanilla["speed"]), len(df_stopping["stopping_time"]))

df_speed = pd.DataFrame({
    "Episode": np.arange(1, min_length + 1),  # Ensure episodes are sequential
    "RMPPO Speed": df_rmppo["speed"].iloc[:min_length].values,
    "Vanilla Speed": df_vanilla["speed"].iloc[:min_length].values,
    "Stopping Time": df_stopping["stopping_time"].iloc[:min_length].values
})

plt.figure(figsize=(10, 5))
sns.heatmap(df_speed.set_index("Episode").T, cmap="coolwarm", annot=False)
plt.title("Speed Variations Heatmap")
plt.xlabel("Episode")
plt.ylabel("Algorithm")
save_plot(plt, "Heatmap_Speed_Variations")

# ==========================
# 7️⃣ Quiver Plot for Direction Changes
# ==========================
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

plot_quiver(df_rmppo, "Quiver Plot - RMPPO")
plot_quiver(df_vanilla, "Quiver Plot - Vanilla PPO")

# ==========================
# 8️⃣ Radial Histogram for Final Directions
# ==========================
def radial_histogram(df, title):
    angles = np.radians(df["average deviation"])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=30, color="blue", alpha=0.7)
    ax.set_title(title)
    save_plot(fig, f"Radial_Histogram_{title.replace(' ', '_')}")

radial_histogram(df_rmppo, "Final Direction - RMPPO")
radial_histogram(df_vanilla, "Final Direction - Vanilla PPO")

print("All plots have been generated and saved in the 'results_plots' directory.")
