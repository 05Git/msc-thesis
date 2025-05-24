from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
import argparse
import os
import numpy as np
from imitation.data.types import TrajectoryWithRew

# diambra run python sb3/imitation_dataset_loader.py --dataset_path _

def get_trajectories(dataset: DiambraDataLoader):
    obs = np.array([data["obs"] for data in dataset.episode_data])
    obs = np.concatenate([obs, [np.zeros_like(dataset.episode_data[-1]["obs"])]], axis=0)
    acts = np.array([data["action"] for data in dataset.episode_data])
    rews = np.array([data["reward"] for data in dataset.episode_data], dtype=np.float16)
    dones = np.array([data["terminated"] or data["truncated"] for data in dataset.episode_data])
    infos = np.array([data["info"] for data in dataset.episode_data])

    return TrajectoryWithRew(
        obs=obs,
        acts=acts,
        rews=rews,
        terminal=dones,
        infos=infos
    )

def main(dataset_path_input):
    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")

    data_loader = DiambraDataLoader(dataset_path)

    n_loops = data_loader.reset()
    trajectories = get_trajectories(data_loader)
    print(trajectories.rews[np.where(trajectories.rews != 0)[0]])

    while n_loops == 0:
        observation, action, reward, terminated, truncated, info = data_loader.step()
        # print("Observation: {}".format(observation))
        # print("Action: {}".format(action))
        # print("Reward: {}".format(reward))
        # print("Terminated: {}".format(terminated))
        # print("Truncated: {}".format(truncated))
        # print("Info: {}".format(info))
        data_loader.render()

        if terminated or truncated:
            n_loops = data_loader.reset()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset')
    opt = parser.parse_args()
    print(opt)

    main(opt.dataset_path)
