from diambra.arena.utils.diambra_data_loader import DiambraDataLoader

from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout

import os
import argparse
import numpy as np
import cv2

def get_transitions(data_loader: DiambraDataLoader, agent_num: int | None):
    n_episodes = len(data_loader.episode_files) # Number of datasets
    trajectories = []
    for i in range(n_episodes):
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
        obs = np.stack([obs[i:i+4] for i in range(len(obs) - 3)], axis=0) # Stack frames together, this is what the policy expects as input
        last_frame_stack = np.stack([obs[-1]] * 4)
        obs = np.concatenate([obs, last_frame_stack], axis=0) # Concat 4 extra frame stacks to match trajectory's required shape
        if agent_num is not None:
            acts = np.array([data["action"][f"agent_{agent_num}"] for data in data_loader.episode_data], dtype=np.uint8)
        else:
            acts = np.array([data["action"] for data in data_loader.episode_data], dtype=np.uint8)
        rews = np.array([data["reward"] for data in data_loader.episode_data], dtype=np.float16)
        dones = np.array([data["terminated"] for data in data_loader.episode_data])
        terminal = np.any(dones) # Does this episode end in a terminal state?
        infos = np.array([data["info"] for data in data_loader.episode_data])

        trajectories.append(TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            terminal=terminal,
            infos=infos
        ))
        print(f"Trajectory {i + 1} loaded")
        break
    
    return rollout.flatten_trajectories_with_rew(trajectories)

def main(dataset_path_input: str, agent_num: int | None):
    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")
    
    data_loader = DiambraDataLoader(dataset_path)
    _ = data_loader.reset()
    print(data_loader.episode_data[0])

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetPath", type=str, required=True, help="Path to imitation trajectories")
    parser.add_argument("--agentNum", type=int, required=False, help="Agent number (if trajectories come from multiagent env)", default=None)
    opt = parser.parse_args()

    main(
        dataset_path_input=opt.datasetPath,
        agent_num=opt.agentNum
    )
