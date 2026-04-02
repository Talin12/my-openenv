from env import TrafficEnv


def run_random_agent(env):
    """Run one episode using random actions. Returns total reward and steps."""
    obs, _ = env.reset()
    total_reward = 0

    print("=" * 42)
    print("  Episode 1: Random Agent")
    print("=" * 42)

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        action_str = "Keep Light" if action == 0 else "Switch Light"
        print(env.render())
        print(f"  Action: {action_str} | Reward: {reward}\n")

        if terminated or truncated:
            reason = "Gridlock (Terminated)" if terminated else "Max Steps (Truncated)"
            print(f"  Ended: {reason}")
            break

    return total_reward, env.current_step


def run_heuristic_agent(env):
    """
    Run one episode using a greedy heuristic:
        - Observe the current green and red direction queue totals.
        - If the red direction has MORE waiting cars, switch (Action 1).
        - Otherwise, keep the current light (Action 0).
    Returns total reward and steps.
    """
    obs, _ = env.reset()
    total_reward = 0

    print("=" * 42)
    print("  Episode 2: Heuristic Agent")
    print("=" * 42)

    while True:
        cars_n, cars_s, cars_e, cars_w, light = obs

        if light == 0:  # N-S currently green
            green_total = int(cars_n) + int(cars_s)
            red_total   = int(cars_e) + int(cars_w)
        else:           # E-W currently green
            green_total = int(cars_e) + int(cars_w)
            red_total   = int(cars_n) + int(cars_s)

        # Switch only if the waiting red direction has more cars
        action = 1 if red_total > green_total else 0

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        action_str = "Keep Light" if action == 0 else "Switch Light"
        print(env.render())
        print(f"  Action: {action_str} | Reward: {reward}\n")

        if terminated or truncated:
            reason = "Gridlock (Terminated)" if terminated else "Max Steps (Truncated)"
            print(f"  Ended: {reason}")
            break

    return total_reward, env.current_step


if __name__ == "__main__":
    env = TrafficEnv(render_mode="ansi")

    random_reward, random_steps     = run_random_agent(env)
    heuristic_reward, heuristic_steps = run_heuristic_agent(env)

    print("\n" + "=" * 42)
    print("  Final Comparison")
    print("=" * 42)
    print(f"  {'Agent':<20} {'Steps':>5}   {'Total Reward':>12}")
    print(f"  {'-'*20}  {'-'*5}   {'-'*12}")
    print(f"  {'Random':<20} {random_steps:>5}   {random_reward:>12}")
    print(f"  {'Heuristic':<20} {heuristic_steps:>5}   {heuristic_reward:>12}")
    winner = "Heuristic" if heuristic_reward > random_reward else "Random"
    print(f"\n  ✅ Better Agent: {winner}")
    print("=" * 42)

    env.close()