import pandas as pd
import matplotlib.pyplot as plt

# Load the data
falls_data = pd.read_csv("falls_running_vanilla.csv")
running_deviation_data = pd.read_csv("running_deviation_vanilla.csv")
average_speeds_data = pd.read_csv("average_speeds_vanilla.csv")

# Graph 1: Cumulative Falls Over Episodes
falls_data['Cumulative Falls'] = falls_data['Fall State'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(falls_data['Episode'], falls_data['Cumulative Falls'], marker='o', linestyle='-', color='b')
plt.title("Cumulative Falls Over Episodes (Vanilla PPO)")
plt.xlabel("Episode")
plt.ylabel("Cumulative Falls")
plt.grid(True)
plt.savefig("cumulative_falls_vanilla.png")
plt.close()

# Graph 2: Running Deviation vs. Episode
plt.figure(figsize=(10, 6))
plt.scatter(running_deviation_data['Episode'], running_deviation_data['Running Deviation'], color='r', alpha=0.6)
plt.title("Running Deviation vs. Episode (Vanilla PPO)")
plt.xlabel("Episode")
plt.ylabel("Running Deviation (m)")
plt.grid(True)
plt.savefig("running_deviation_vanilla.png")
plt.close()

# Graph 3: Average Speed Over Episodes
plt.figure(figsize=(10, 6))
plt.plot(average_speeds_data['Episode'], average_speeds_data['Average Speed'], marker='o', linestyle='-', color='g')
plt.title("Average Speed Over Episodes (Vanilla PPO)")
plt.xlabel("Episode")
plt.ylabel("Average Speed (m/s)")
plt.grid(True)
plt.savefig("average_speed_vanilla.png")
plt.close()