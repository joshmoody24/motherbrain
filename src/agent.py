import embodied
import numpy as np
import os
import elements
import dreamerv3.agent as dreamerv3_agent
import pathlib
import gymnasium as gym
import retro

# Imports from main.py that are now needed here
from embodied.envs import from_gym
from embodied.core import wrappers


# Custom wrappers from main.py that are now needed here
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
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 7: START
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def step(self, action):
        """Converts the discrete action and conforms to the old gym API."""
        obs, reward, terminated, truncated, info = self.env.step(self._actions[action])
        done = terminated and truncated
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


def run_agent(config, replay):
    """
    Runs the main agent training loop using the embodied.Driver.
    """

    # Environment creation function. This will be passed to the Driver.
    def env_ctor():
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "..", "data"))
        env = retro.make(
            "Metroid-Nes", inttype=retro.data.Integrations.ALL, render_mode="human"
        )
        env = Discretizer(env)
        env = from_gym.FromGym(env, obs_key="image")
        env = wrap_env(env, config)
        env = wrappers.TimeLimit(env, 1000)
        env = BatchWrapper(env)
        return env

    # Create a dummy environment to get the obs_space and act_space
    # This is how the original dreamerv3 code does it.
    dummy_env = env_ctor()
    obs_space = dummy_env.obs_space
    act_space = dummy_env.act_space
    dummy_env.close()  # Close the dummy env immediately after getting spaces

    # Filter out the 'reset' action from the space passed to the agent
    act_space = {k: v for k, v in act_space.items() if k != "reset"}

    # Create a specific config for the agent by unpacking the agent section
    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )

    # Create the agent
    agent = dreamerv3_agent.Agent(obs_space, act_space, agent_config)

    # The driver handles the environment interaction loop.
    # It expects a list of environment creation functions.
    driver = embodied.Driver([env_ctor])
    driver.on_step(replay.add)

    # Initialize the driver with an initial observation.
    # This will call env.reset() and set driver.obs.
    driver.reset()

    # Create the training dataset iterator.
    fn = lambda: replay.sample(config.batch_size, "train")
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length,
        consec=config.consec_train,
        prefix=config.replay_context,
        strict=True,
        contiguous=True,
    )

    # Use the agent's stream method to properly add seeds and handle device placement
    stream = agent.stream(stream)
    dataset = iter(stream)

    # Initialize training state.
    train_state = agent.init_train(config.batch_size)

    print("Prefilling replay buffer...")
    # Use a random policy to collect initial data.
    # We need to wrap the random agent to include the expected output keys
    random_agent = embodied.RandomAgent(obs_space, act_space)

    # Get the RSSM dimensions from the agent's configuration
    rssm_config = agent.config.dyn.rssm
    deter_size = rssm_config.deter
    stoch_size = rssm_config.stoch
    classes = rssm_config.classes

    print(f"RSSM config - deter: {deter_size}, stoch: {stoch_size}, classes: {classes}")

    def random_policy_with_outputs(carry, obs):
        carry, acts, _ = random_agent.policy(carry, obs)
        batch_size = len(obs["is_first"])
        # For size1m: classes=4, and the expected shape is (2, 4) based on the Extras output
        # This means it's using a discrete latent with 4 classes, but only 2 categorical variables
        outs = {
            "dyn/deter": np.zeros((batch_size, deter_size), dtype=np.float32),
            "dyn/stoch": np.zeros((batch_size, 2, 4), dtype=np.float32),
        }
        return carry, acts, outs

    driver(random_policy_with_outputs, steps=config.batch_size * 10)
    print("Done prefilling.")

    print("\nStarting training...")
    # Initialize policy state.
    policy_state = agent.init_policy(1)  # batch_size=1 for single environment

    # This policy will be used by the driver to collect new data.
    action_names = [
        "No-op",
        "Right",
        "Left",
        "A(Jump)",
        "B(Shoot)",
        "Right+Jump",
        "Right+Shoot",
        "START",
    ]
    action_counter = 0

    def policy_with_logging(state, obs):
        nonlocal action_counter
        result = agent.policy(
            state if state is not None else policy_state, obs, mode="train"
        )

        # Log actions periodically
        if action_counter % 100 == 0:
            action_idx = result[1]["action"][0]  # Get first env's action
            print(f"Action chosen: {action_names[action_idx]} (idx: {action_idx})")
        action_counter += 1

        return result

    policy = policy_with_logging

    # Main training loop.
    step = 0
    env_steps = 0
    import time

    train_ratio = config.run.train_ratio  # Train once every N environment steps
    print(f"Training ratio: 1 train step per {train_ratio} environment steps")

    while True:
        step += 1

        # Time environment interaction
        env_start = time.time()
        # Let the driver run for a few steps to collect data.
        driver(policy, steps=train_ratio)
        env_steps += train_ratio
        env_time = time.time() - env_start

        # Time training
        train_start = time.time()
        # Perform a training step.
        batch = next(dataset)
        train_state, outs, metrics = agent.train(train_state, batch)
        train_time = time.time() - train_start

        if step % 10 == 0:
            total_time = env_time + train_time
            actual_fps = train_ratio / total_time  # Real FPS including training pauses
            print(
                f"Step {step}: env={env_time:.3f}s ({env_time/total_time*100:.1f}%), "
                f"train={train_time:.3f}s ({train_time/total_time*100:.1f}%), "
                f"total={total_time:.3f}s, visual_FPS={actual_fps:.1f}, "
                f"env_steps={env_steps}"
            )
