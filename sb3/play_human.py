import diambra.arena
from diambra.arena import EnvironmentSettings
from diambra.arena.utils.controller import get_diambra_controller
import time

def main():
    # Settings
    settings = EnvironmentSettings()
    settings.characters = "Ryu"
    settings.step_ratio = 1
    settings.difficulty = 1
    settings.super_art = 1
    settings.continue_game = 0.

    # Environment creation
    env = diambra.arena.make("sfiii3n", env_settings=settings, render_mode="human")

    # Controller initialization
    controller = get_diambra_controller(env.get_actions_tuples())
    controller.start()

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    while True:
        # (Optional) Environment rendering
        # env.render()

        # Action random sampling
        actions = controller.get_actions()

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        if info["round_done"]:
            print("̣\n-------------")
            print(f"env_done: {info['env_done']}")
            print(f"game_done: {info['game_done']}")
            print(f"episode_done: {info['episode_done']}")
            print(f"round_done: {info['round_done']}")
            print(f"stage_done: {info['stage_done']}")
            print(f"difficulty: {info['settings'].episode_settings.difficulty}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print("̣\n-------------")

        time.sleep(6e-3)

        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()