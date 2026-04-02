from env import TrafficEnv

env = TrafficEnv(render_mode="ansi")
obs, info = env.reset()

total_reward = 0
step_count = 0

print("=" * 40)
print("  Random Agent — Traffic Control Env")
print("=" * 40)

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    action_str = "Keep Light" if action == 0 else "Switch Light"
    print(env.render())
    print(f"  Action: {action_str} | Reward: {reward}")
    print()

    if terminated or truncated:
        reason = "Gridlock (Terminated)" if terminated else "Max Steps (Truncated)"
        print("=" * 40)
        print(f"  Episode Finished — {reason}")
        print(f"  Steps: {step_count} | Total Reward: {total_reward}")
        print("=" * 40)
        break

env.close()