import os
from diambra.arena.utils.controller import get_diambra_controller
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings, RecordingSettings
from diambra.arena import SpaceTypes, load_settings_flat_dict
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import argparse
import yaml
import custom_wrappers
import time
import utils

# diambra run -g python sb3/record.py --gameID _ --settingsCfg config_files/transfer-cfg-settings.yaml --use_controller/--no-use_controller --numEpisodesToRecord _

def main(game_id: str, settings_cfg: str, use_controller: bool, num_episodes_to_record: int):
    # Settings
    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    # Load shared settings
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.DISCRETE if settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1
    game_settings = settings_params["settings"][game_id]
    game_settings["characters"] = game_settings["characters"][0]
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)
    wrappers_settings = settings_params["wrappers_settings"]
    wrappers_settings["flatten"] = True
    stack_frames = wrappers_settings["stack_frames"]
    wrappers_settings = load_settings_flat_dict(WrappersSettings, wrappers_settings)

    # Set up seeds
    seeds = list(range(0, 10000))
    seed_idx = 0

    # Recording settings
    base_path = os.path.dirname(os.path.abspath(__file__))
    recording_settings = RecordingSettings()
    recording_settings.username = "tgbn"
    if use_controller:
        ep_recording_path = "recordings/human/episode_recording"
    else:
        ep_recording_path = "recordings/random/episode_recording"
    action_space = "discrete" if settings.action_space == SpaceTypes.DISCRETE else "multi_discrete"
    recording_settings.dataset_path = os.path.join(base_path, ep_recording_path, game_id, action_space)

    for _ in range(num_episodes_to_record):
        seed = seeds[seed_idx]
        env, _ = make_sb3_env(
            settings.game_id,
            env_settings=settings,
            wrappers_settings=wrappers_settings,
            episode_recording_settings=recording_settings,
            render_mode="human",
            seed=seed,
            no_vec=True,
        )
        utils.set_global_seed(seed)
        if settings.action_space == SpaceTypes.DISCRETE:
            env = custom_wrappers.DiscreteTransferWrapper(env, stack_frames)
        else:
            env = custom_wrappers.MDTransferWrapper(env, stack_frames)

        if use_controller:
            # Controller initialization
            controller = get_diambra_controller(env.get_actions_tuples())
            controller.start()

        observation, info = env.reset()

        # Player-Environment interaction loop
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
            # Episode end (Done condition) check
            if terminated or truncated:
                break
            if use_controller:
                time.sleep(1e-2)

        if use_controller:
            controller.stop()

        env.close()

        seed_idx += 1
        if seed_idx > len(seeds) - 1:
            seed_idx = 0 # Just in case we record more than 10000 episodes

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameID", type=str, required=False, help="Specific game to record for", default="sfiii3n")
    parser.add_argument("--settingsCfg", type=str, required=False, help="Env settings config", default="config_files/transfer-cfg-settings.yaml")
    parser.add_argument("--numEpisodesToRecord", type=int, required=False, help="Env seed", default=1)
    parser.add_argument('--use_controller', action=argparse.BooleanOptionalAction, required=True, help='Flag to activate use of controller')
    opt = parser.parse_args()

    main(
        game_id=opt.gameID,
        settings_cfg=opt.settingsCfg,
        use_controller=opt.use_controller,
        num_episodes_to_record=opt.numEpisodesToRecord,
    )
