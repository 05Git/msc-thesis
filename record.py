import diambra.arena
from diambra.arena.utils.controller import get_diambra_controller
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, RecordingSettings, SpaceTypes, load_settings_flat_dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import argparse
import yaml
import custom_wrappers
import time
import utils
import os
import random
import numpy as np

# diambra run -g python sb3/record.py --gameID _ --settingsCfg config_files/transfer-cfg-settings.yaml --use_controller/--no-use_controller --numEpisodesToRecord _

def main(
    game_id: str, 
    settings_cfg: str, 
    policy_cfg: str,
    use_controller: bool, 
    agent_path: str | None,
    multiplayer_env: bool, 
    num_episodes_to_record: int, 
    recording_folder: str,
    rec_username: str
):
    assert not (use_controller and agent_path), "Haven't implemented human vs agent yet"

    # Read cfg files
    policy_file = open(policy_cfg)
    policy_params = yaml.load(policy_file, Loader=yaml.FullLoader)
    policy_file.close()

    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    # Set up seeds
    seeds = list(range(0, 10000))
    utils.set_global_seed(0) # Set up reproducibility before popping random seeds

    # Load shared settings
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1 if use_controller else settings["step_ratio"]
    game_settings = settings_params["settings"][game_id]
    game_settings["characters"] = game_settings["characters"]["train"][0]
    if multiplayer_env:
        settings["action_space"] = (settings["action_space"], settings["action_space"])
        settings["outfits"] = (settings["outfits"], settings["outfits"])
        game_settings["characters"] = (game_settings["characters"][0], game_settings["characters"][1])
        if game_id == "sfiii3n":
            game_settings["super_art"] = (game_settings["super_art"], game_settings["super_art"])
        if game_id == "kof98umh":
            game_settings["fighting_style"] = (game_settings["fighting_style"], game_settings["fighting_style"])
            game_settings["ultimate_style"] = (game_settings["ultimate_style"], game_settings["ultimate_style"])    
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, settings) if multiplayer_env else load_settings_flat_dict(EnvironmentSettings, settings)
    wrappers_settings = settings_params["wrappers_settings"]
    if multiplayer_env:
        wrappers_settings["role_relative"] = False
    wrappers_settings = load_settings_flat_dict(WrappersSettings, wrappers_settings)

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
    recording_settings.dataset_path = os.path.join(base_path, ep_recording_path, game_id)
    
    env = diambra.arena.make(
        game_id=settings.game_id,
        env_settings=settings,
        wrappers_settings=wrappers_settings,
        episode_recording_settings=recording_settings,
        render_mode="human",
    )
    if multiplayer_env:
        if settings.action_space == SpaceTypes.DISCRETE:
            env = custom_wrappers.MultiAgentDiscreteTransferWrapper(
                env=env, 
                stack_frames=wrappers_settings.stack_frames,
            )
        else:
            env = custom_wrappers.MultiAgentMDTransferWrapper(
                env=env, 
                stack_frames=wrappers_settings.stack_frames,
            )
    else:
        if settings.action_space == SpaceTypes.DISCRETE:
            env = custom_wrappers.DiscreteTransferWrapper(
                env=env, 
                stack_frames=wrappers_settings.stack_frames, 
                characters=[settings.characters]
            )
        else:
            env = custom_wrappers.MDTransferWrapper(
                env=env, 
                stack_frames=wrappers_settings.stack_frames, 
                characters=[settings.characters]
            )
    env = VecTransposeImage(DummyVecEnv([lambda: env]))

    if use_controller:
        # Controller initialization
        controller = get_diambra_controller(env.get_actions_tuples())
        controller.start()

    if agent_path and os.path.isfile(agent_path):
        policy_kwargs = policy_params["policy_kwargs"]
        if not policy_kwargs:
            policy_kwargs = {}
        agent = PPO.load(
            path=agent_path,
            env=env,
            policy_kwargs=policy_kwargs,
            custom_objects={
                "action_space" : env.action_space,
                "observation_space" : env.observation_space,
            }
        )

    for _ in range(num_episodes_to_record):
        seed = seeds.pop(random.randint(0, len(seeds)))
        if not len(seeds) > 0:
            seeds = list(range(0, 10000)) # Reset seeds
        utils.set_global_seed(seed)

        # Player-Environment interaction loop
        observation = env.reset()
        while True:
            # env.render()
            actions = env.action_space.sample()
            if use_controller:
                if multiplayer_env:
                    actions["agent_0"] = controller.get_actions()
                    actions["agent_1"] = [0, 0]
                else:
                    actions = controller.get_actions()
            if agent_path:
                if multiplayer_env:
                    p1_actions, _ = agent.predict(observation, deterministic=False)
                    p2_actions = [0, 0]
                    actions = [np.append(p1_actions, p2_actions)]
                else:
                    actions, _ = agent.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(actions)
            if done:
                break
            if use_controller:
                time.sleep(1e-2)

    if use_controller:
        controller.stop()

    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameID", type=str, required=False, help="Specific game to record for", default="sfiii3n")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/settings-cfg.yaml")
    parser.add_argument("--policyCfg", type=str, required=False, help="Policy config", default="config_files/ppo-cfg.yaml")
    parser.add_argument("--numEpisodesToRecord", type=int, required=False, help="Number of episodes to record", default=1)
    parser.add_argument("--useController", action=argparse.BooleanOptionalAction, required=True, help="Flag to activate use of controller")
    parser.add_argument("--agentPath", type=str, required=False, help="Path to trained agent", default=None)
    parser.add_argument("--multiagentEnv", action=argparse.BooleanOptionalAction, required=False, help="Use single or multiagent env", default=False)
    parser.add_argument("--recordingFolder", type=str, required=False, help="Folder to save recordings in", default="episode_recording")
    parser.add_argument("--username", type=str, required=True, help="Username to save recordings to")
    opt = parser.parse_args()

    main(
        game_id=opt.gameID,
        settings_cfg=opt.settingsCfg,
        policy_cfg=opt.policyCfg,
        use_controller=opt.useController,
        agent_path=opt.agentPath,
        num_episodes_to_record=opt.numEpisodesToRecord,
        multiplayer_env=opt.multiagentEnv,
        recording_folder=opt.recordingFolder,
        rec_username=opt.username,
    )
