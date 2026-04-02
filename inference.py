from env import TrafficEnv

env = TrafficEnv(render_mode="ansi")
obs, _ = env.reset()

print("=" * 42)
print("  Inference — Heuristic Agent (10 steps)")
print("=" * 42)

for step in range(10):
    cars_n, cars_s, cars_e, cars_w, light = obs

    if light == 0:  # N-S currently green
        green_total = int(cars_n) + int(cars_s)
        red_total   = int(cars_e) + int(cars_w)
    else:           # E-W currently green
        green_total = int(cars_e) + int(cars_w)
        red_total   = int(cars_n) + int(cars_s)

    action = 1 if red_total > green_total else 0

    obs, reward, terminated, truncated, _ = env.step(action)

    print(env.render())
    print(f"  Action: {'Switch' if action else 'Keep'} | Reward: {reward}\n")

    if terminated or truncated:
        print("  Episode ended early.")
        break

print("=" * 42)
print("  Inference complete.")
print("=" * 42)

env.close()