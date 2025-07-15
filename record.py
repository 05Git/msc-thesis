"""
record.py: Record episode trajectories for imitation learning.
"""
import diambra.arena
import configs
import os
import random
import time
import argparse
import numpy as np

from diambra.arena import RecordingSettings
from diambra.arena.utils.controller import get_diambra_controller
from stable_baselines3.common.utils import set_random_seed
from utils import load_agent

# diambra run -g python record.py --use_controller --num_eps_to_record _ --agent_path _ --rec_folder _ --username _ --deterministic

def main(
    use_controller: bool,
    agent_path: str,
    num_episodes_to_record: int, 
    recording_folder: str,
    rec_username: str,
    deterministic: bool,
):
    assert not (use_controller and agent_path), "Haven't implemented human vs agent yet"

    # Set up seeds
    seeds = list(range(0, 10000))
    # step_ratio > 1 will make gameplay too fast to play normally
    if use_controller:
        configs.train_settings.step_ratio = 1

    # Recording settings
    base_path = os.path.dirname(os.path.abspath(__file__))
    recording_settings = RecordingSettings()
    recording_settings.username = rec_username
    if use_controller:
        ep_recording_path = os.path.join("recordings/human", recording_folder)
    elif agent_path:
        ep_recording_path = os.path.join("recordings/agent", recording_folder)
    else:
        ep_recording_path = os.path.join("recordings/random", recording_folder)
    recording_settings.dataset_path = os.path.join(base_path, ep_recording_path, configs.train_settings.game_id)
    
    env = diambra.arena.make(
        game_id=configs.train_settings.game_id,
        env_settings=configs.train_settings,
        wrappers_settings=configs.train_wrappers,
        episode_recording_settings=recording_settings,
        render_mode="human",
    )

    if use_controller:
        # Controller initialization
        controller = get_diambra_controller(env.get_actions_tuples())
        controller.start()

    if agent_path and os.path.isfile(agent_path):
        agent = load_agent(env=env, policy_path=agent_path, force_load=True)

    for _ in range(num_episodes_to_record):
        seed = seeds.pop(random.randint(0, len(seeds)))
        if not len(seeds) > 0:
            seeds = list(range(0, 10000)) # Reset seeds
        set_random_seed(seed)

        obs = env.reset()
        while True:
            # env.render()
            actions = env.action_space.sample()
            if use_controller:
                if configs.misc["num_players"] == 2:
                    actions["agent_0"] = controller.get_actions()
                    actions["agent_1"] = [0, 0]
                else:
                    actions = controller.get_actions()
            if agent_path:
                if configs.misc["num_players"] == 2:
                    p1_actions, _ = agent.predict(obs, deterministic=deterministic)
                    p2_actions = [0, 0]
                    actions = [np.append(p1_actions, p2_actions)]
                else:
                    actions, _ = agent.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
            if use_controller:
                time.sleep(1e-2)

    if use_controller:
        controller.stop()

    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eps_to_record", type=int, required=False, help="Number of episodes to record", default=1)
    parser.add_argument("--use_controller", action=argparse.BooleanOptionalAction, required=True, help="Flag to activate use of controller")
    parser.add_argument("--agent_path", type=str, required=False, help="Path to trained agent", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Whether to follow a deterministic or stochastic policy", default=True)
    parser.add_argument("--rec_folder", type=str, required=False, help="Folder to save recordings in", default="episode_recording")
    parser.add_argument("--username", type=str, required=True, help="Username to save recordings to")
    opt = parser.parse_args()

    main(
        use_controller=opt.use_controller,
        agent_path=opt.agent_path,
        num_episodes_to_record=opt.num_eps_to_record,
        deterministic=opt.deterministic,
        recording_folder=opt.rec_folder,
        rec_username=opt.username,
    )
