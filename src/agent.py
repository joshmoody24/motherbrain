def run_agent(env):
    """
    Runs the main agent loop.
    For now, it takes random actions.
    """
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()
