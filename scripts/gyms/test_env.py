from Basic_Run import Basic_Run

if __name__ == "__main__":
    # Test a single environment instance
    env = Basic_Run(
        ip="127.0.0.1",
        server_p=3100,
        monitor_p=3200,
        r_type=1,
        enable_draw=True,  # Enable rendering to visualize the simulation
        env_id=0,
        mode="run"  # Start in "run" mode for testing
    )

    try:
        # Reset the environment
        obs = env.reset()
        print("Environment reset successfully.")

        # Take some steps in the environment
        for step in range(10):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, info = env.step(action)
            print(f"Step {step}: Reward = {reward}, Done = {done}")

            if done:
                print("Episode finished. Resetting environment.")
                obs = env.reset()
    except Exception as e:
        print(f"Error during environment testing: {e}")
    finally:
        env.close()
