from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
import argparse
import os
import cv2
import numpy as np
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout

# diambra run python sb3/imitation_dataset_loader.py --dataset_path _

def get_transitions(data_loader: DiambraDataLoader):
    n_episodes = len(data_loader.episode_files) # Number of datasets
    trajectories = []
    for _ in range(n_episodes):
        _ = data_loader.reset() # Load next episode data
        obs = np.array([
            cv2.imdecode( # Decode frames to correct shape
                np.frombuffer(
                    data["obs"]["frame"], dtype=np.uint8 # Read frame data for each step in the episode
                ),
                cv2.IMREAD_UNCHANGED,
            )
            for data in data_loader.episode_data
        ])
        obs = np.concatenate([obs, [np.zeros_like(obs[-1])]], axis=0) # Obs expected to be one entry longer than other arrays
        acts = np.array([data["action"] for data in data_loader.episode_data], dtype=np.uint8)
        rews = np.array([data["reward"] for data in data_loader.episode_data], dtype=np.float16)
        dones = np.array([data["terminated"] for data in data_loader.episode_data])
        terminal = np.any([done == True for done in dones]) # Does this episode end in a terminal state?
        infos = np.array([data["info"] for data in data_loader.episode_data])

        trajectories.append(TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            terminal=terminal,
            infos=infos
        ))
    
    return rollout.flatten_trajectories_with_rew(trajectories)

def main(dataset_path_input):
    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")

    data_loader = DiambraDataLoader(dataset_path)

    n_loops = data_loader.reset()
    transitions = get_transitions(data_loader)
    print(f"Transitions obs shape: {transitions.obs.shape}")
    print(f"Transitions acts shape: {transitions.acts.shape}")
    print(f"Transitions infos shape: {transitions.infos.shape}")
    print(f"Transitions rews shape: {transitions.rews.shape}")
    
    return 0

    i = 0
    max_i = 5
    while n_loops == 0:
        observation, action, reward, terminated, truncated, info = data_loader.step()
        # print("Observation: {}".format(observation["frame"]))
        # print("Action: {}".format(action))
        # print("Reward: {}".format(reward))
        # print("Terminated: {}".format(terminated))
        # print("Truncated: {}".format(truncated))
        # print("Info: {}".format(info))
        data_loader.render()
        i += 1

        if terminated or truncated or not i < max_i:
            n_loops = data_loader.reset()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset')
    opt = parser.parse_args()
    print(opt)

    main(opt.dataset_path)
