import diambra.arena
import os
from diambra.arena.utils.controller import get_diambra_controller
from diambra.arena import EnvironmentSettings, SpaceTypes, RecordingSettings, load_settings_flat_dict
import argparse
import yaml
import custom_wrappers
import time

# diambra run -g python sb3/record.py --gameID _ --settingsCfg config_files/transfer-cfg-settings.yaml --use_controller/--no-use_controller

def main(game_id: str, settings_cfg: str, use_controller: bool):
    # Settings
    settings_file = open(settings_cfg)
    settings_params = yaml.load(settings_file, Loader=yaml.FullLoader)
    settings_file.close()

    # Load shared settings
    settings = settings_params["settings"]["shared"]
    settings["action_space"] = SpaceTypes.MULTI_DISCRETE
    settings["step_ratio"] = 1
    game_settings = settings_params["settings"][game_id]
    game_settings["characters"] = game_settings["characters"][0]
    settings.update(game_settings)
    settings = load_settings_flat_dict(EnvironmentSettings, settings)

    # Recording settings
    base_path = os.path.dirname(os.path.abspath(__file__))
    recording_settings = RecordingSettings()
    recording_settings.username = "tgbn"
    if use_controller:
        recording_settings.dataset_path = os.path.join(base_path, "recordings/human/episode_recording", game_id)
    else:
        recording_settings.dataset_path = os.path.join(base_path, "recordings/random/episode_recording", game_id)

    env = diambra.arena.make(game_id, settings, episode_recording_settings=recording_settings, render_mode="human")
    # if settings.action_space == SpaceTypes.DISCRETE:
    #     env = custom_wrappers.DiscreteTransferWrapper(env)
    # else:
    #     env = custom_wrappers.MDTransferWrapper(env)

    if use_controller:
        # Controller initialization
        controller = get_diambra_controller(env.get_actions_tuples())
        controller.start()

    observation, info = env.reset(seed=42)

    # Player-Environment interaction loop
    while True:
        # env.render()
        if use_controller:
            actions = controller.get_actions()
        else:
            actions = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        # Episode end (Done condition) check
        if done:
            observation, info = env.reset()
            break
        time.sleep(1e-2)

    if use_controller:
        controller.stop()
    env.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameID", type=str, required=False, help="Specific game to record for")
    parser.add_argument("--settingsCfg", type=str, required=True, help="Env settings config")
    parser.add_argument('--use_controller', action=argparse.BooleanOptionalAction, required=True, help='Flag to activate use of controller')
    opt = parser.parse_args()
    print(opt)

    main(opt.gameID, opt.settingsCfg, opt.use_controller)