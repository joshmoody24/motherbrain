import sys
import os
import pathlib
import gymnasium as gym
import numpy as np
import retro
import ruamel.yaml as yaml

# Add the submodule path to the python path to ensure we use the bundled libraries.
submodule_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dreamerv3")
)
sys.path.insert(0, submodule_path)

import elements
import embodied
from embodied.envs import from_gym
from embodied.core import wrappers
from dreamerv3.agent import Agent
from .agent import run_agent


class Discretizer(gym.Wrapper):
    """A custom wrapper to convert discrete actions to multi-binary actions."""

    def __init__(self, env):
        super().__init__(env)
        # B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A
        self._actions = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No-op
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: A (Jump)
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: B (Shoot)
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 5: Right + Jump
            [1, 0, 0, 0, 0, 0, 0, 1, 0],  # 6: Right + Shoot
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def step(self, action):
        """Converts the discrete action and conforms to the old gym API."""
        obs, reward, terminated, truncated, info = self.env.step(self._actions[action])
        done = terminated or truncated
        return obs, reward, done, info


class BatchWrapper:
    """A wrapper to make a single environment look like a batch of one."""

    def __init__(self, env):
        self.env = env

    def __len__(self):
        return 1

    def __getattr__(self, name):
        return getattr(self.env, name)


def wrap_env(env, config):
    """Applies observation and action space wrappers."""
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.NormalizeAction(env, name)
    env = wrappers.UnifyDtypes(env)
    env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


def main():
    """
    Main entry point for the application.
    """
    # Load DreamerV3 config from the submodule
    config_path = os.path.join(submodule_path, "dreamerv3", "configs.yaml")
    configs = yaml.YAML(typ="safe").load(pathlib.Path(config_path).read_text())

    # Use the robust config loading from the original DreamerV3 script
    config = elements.Config(configs["defaults"])
    parsed, _ = elements.Flags(
        configs=["defaults", "atari", "debug", "size1m"]
    ).parse_known([])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse([])

    # Increase train ratio for smoother visual experience
    config = config.update({"run.train_ratio": 512})

    # Remove CPU restriction - let JAX auto-detect the best platform
    # config = config.update({"jax.platform": "cpu"})

    # Print which platform JAX is using
    import jax

    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}\n")

    # Create replay buffer
    replay_length = config.batch_length + config.replay_context
    replay_dir = os.path.join(submodule_path, "replay_buffer")
    replay = embodied.replay.Replay(
        length=replay_length,
        capacity=config.replay.size,
        directory=replay_dir,
        chunksize=config.replay.chunksize,
    )

    # Run the agent
    run_agent(config, replay)


if __name__ == "__main__":
    main()

