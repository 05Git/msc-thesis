from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Settings
    settings = EnvironmentSettings()
    settings.frame_shape = (84, 84, 1)
    settings.characters = ("Ryu")

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 5
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 12
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = []

    # Create environment
    env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    # Instantiate the agent
    agent = PPO("MultiInputPolicy", env, verbose=1, device=device)
    # Train the agent
    agent.learn(total_timesteps=200)
    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Run trained agent
    observation = env.reset()
    cumulative_reward = [0.0 for _ in range(num_envs)]
    while True:
        action, _state = agent.predict(observation, deterministic=True)

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if any(x != 0 for x in reward):
            print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()