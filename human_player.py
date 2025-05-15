import diambra.arena
import os
from diambra.arena.utils.controller import get_diambra_controller
from diambra.arena import EnvironmentSettings, SpaceTypes, RecordingSettings

def main():
    # Environment Initialization
    # Settings
    settings = EnvironmentSettings()
    settings.frame_shape = (256, 256, 1)
    settings.characters = ("Ryu")
    settings.step_ration = 1
    settings.action_space = SpaceTypes.MULTI_DISCRETE

    # Recording settings
    home_dir = os.path.expanduser("~")
    game_id = "sfiii3n"
    recording_settings = RecordingSettings()
    recording_settings.dataset_path = os.path.join(home_dir, "test/episode_recording", game_id)
    recording_settings.username = "tgbn"

    env = diambra.arena.make(game_id, settings, episode_recording_settings=recording_settings, render_mode="human")

    # Controller initialization
    controller = get_diambra_controller(env.get_actions_tuples())
    controller.start()

    # Player-Environment interaction loop
    while True:
        env.render()
        actions = controller.get_actions()
        print(actions)
        observation, reward, done, info = env.step(actions)

        # Episode end (Done condition) check
        if done:
            observation = env.reset()
            break

    env.close()
    controller.stop()

    return 0

if __name__ == '__main__':
    main()