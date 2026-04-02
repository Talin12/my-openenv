import gradio as gr
from env import TrafficEnv

# Initialize global environment
env = TrafficEnv()

def format_state(state):
    n, s, e, w, light = state
    light_str = "🟢 North-South" if light == 0 else "🟢 East-West"
    return f"🚦 Green Light: {light_str}\n\n🚗 Cars Waiting:\n- North: {n}\n- South: {s}\n- East: {e}\n- West: {w}"

def reset_simulation():
    state = env.reset()
    return format_state(state), "Environment Reset", 0

def step_simulation(action_type):
    action = 0 if action_type == "Keep Light" else 1
    state, reward = env.step(action)
    
    action_log = f"Agent Action Taken: {action_type} (Action {action})"
    return format_state(state), action_log, reward

# Build the UI
with gr.Blocks(title="Smart City Traffic RL Environment") as demo:
    gr.Markdown("# 🚦 Traffic Control RL Environment")
    gr.Markdown("A real-world simulation environment for the OpenEnv × MetAI hackathon. Act as the AI agent below to control the intersection.")
    
    with gr.Row():
        with gr.Column():
            state_display = gr.Textbox(label="Current Environment State", lines=6, interactive=False)
            action_log = gr.Textbox(label="Event Log", interactive=False)
            reward_display = gr.Number(label="Last Step Reward (Congestion Penalty)", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Agent Controls")
            btn_keep = gr.Button("Step: Keep Light (Action 0)", variant="primary")
            btn_switch = gr.Button("Step: Switch Light (Action 1)", variant="stop")
            gr.Markdown("---")
            btn_reset = gr.Button("Reset Environment")

    # Wire up the buttons
    btn_keep.click(fn=lambda: step_simulation("Keep Light"), outputs=[state_display, action_log, reward_display])
    btn_switch.click(fn=lambda: step_simulation("Switch Light"), outputs=[state_display, action_log, reward_display])
    btn_reset.click(fn=reset_simulation, outputs=[state_display, action_log, reward_display])

    # Run reset on initial load so the UI isn't blank
    demo.load(fn=reset_simulation, outputs=[state_display, action_log, reward_display])

if __name__ == "__main__":
    demo.launch()