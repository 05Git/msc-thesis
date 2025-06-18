import os
import random
from diambra.arena.utils.controller import get_diambra_controller
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings, RecordingSettings
from diambra.arena import EnvironmentSettingsMultiAgent, SpaceTypes, load_settings_flat_dict
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import argparse
import yaml
import custom_wrappers
import time
import utils

# diambra run -g python sb3/record.py --gameID _ --settingsCfg config_files/transfer-cfg-settings.yaml --use_controller/--no-use_controller --numEpisodesToRecord _

def main(game_id: str, settings_cfg: str, use_controller: bool, multiplayer_env: bool, num_episodes_to_record: int, recording_folder: str, rec_username: str):
    # Settings
    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    # Set up seeds
    seeds = list(range(0, 10000))
    utils.set_global_seed(0) # Set up reproducibility before popping random seeds

    # Recording settings
    base_path = os.path.dirname(os.path.abspath(__file__))
    recording_settings = RecordingSettings()
    recording_settings.username = rec_username
    if use_controller:
        ep_recording_path = os.path.join("recordings/human", recording_folder)
    else:
        ep_recording_path = os.path.join("recordings/random", recording_folder)
    recording_settings.dataset_path = os.path.join(base_path, ep_recording_path, game_id, settings["action_space"])

    # Load shared settings
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1 if use_controller else settings["step_ratio"]
    settings["characters"] = settings["characters"]["train"]
    game_settings = settings_params["settings"][game_id]
    settings.update(game_settings)
    if multiplayer_env:
        settings["action_space"] = (settings["action_space"], settings["action_space"])
        settings["outfits"] = (settings["outfits"], settings["outfits"])
    settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, settings) if multiplayer_env else load_settings_flat_dict(EnvironmentSettings, settings)
    wrappers_settings = settings_params["wrappers_settings"]
    wrappers_settings = load_settings_flat_dict(WrappersSettings, wrappers_settings)

    for _ in range(num_episodes_to_record):
        seed = seeds.pop(random.randint(0, len(seeds)))
        if not len(seeds) > 0:
            seeds = list(range(0, 10000)) # Reset seeds

        env, _ = make_sb3_env(
            settings.game_id,
            env_settings=settings,
            wrappers_settings=wrappers_settings,
            episode_recording_settings=recording_settings,
            render_mode="human",
            seed=seed,
            # no_vec=True,
        )
        utils.set_global_seed(seed)
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
        if use_controller:
            # Controller initialization
            controller = get_diambra_controller(env.get_actions_tuples())
            controller.start()

        # Player-Environment interaction loop
        observation, info = env.reset()
        while True:
            # env.render()
            if use_controller:
                actions = controller.get_actions()
                if settings.action_space == SpaceTypes.DISCRETE:
                    if actions[1] > 0:
                        actions = 8 + actions[1] # In discrete action sets, action indices come after the movement indices
                    else:
                        actions = actions[0] # If no action idx, use the movement idx
            else:
                actions = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(actions)
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
    parser.add_argument("--gameID", type=str, required=False, help="Specific game to record for", default="sfiii3n")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/transfer-cfg-settings.yaml")
    parser.add_argument("--numEpisodesToRecord", type=int, required=False, help="Env seed", default=1)
    parser.add_argument("--useController", action=argparse.BooleanOptionalAction, required=True, help="Flag to activate use of controller")
    parser.add_argument("--multiagentEnv", action=argparse.BooleanOptionalAction, required=False, help="Use single or multiagent env", default=False)
    parser.add_argument("--recordingFolder", type=str, required=False, help="Folder to save recordings in", default="episode_recording")
    parser.add_argument("--username", type=str, required=True, help="Username to save recordings to")
    opt = parser.parse_args()

    main(
        game_id=opt.gameID,
        settings_cfg=opt.settingsCfg,
        useController=opt.use_controller,
        num_episodes_to_record=opt.numEpisodesToRecord,
        multiplayer_env=opt.multiagentEnv,
        recording_folder=opt.recordingFolder,
        rec_username=opt.username,
    )
