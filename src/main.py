import retro
import os
from .agent import run_agent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "data"))

env = retro.make(
    "Metroid-Nes", inttype=retro.data.Integrations.ALL, render_mode="human"
)

obs, info = env.reset()

run_agent(env)


#
env.close()
