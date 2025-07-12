"""
imitate.py: Train a policy through behavioural cloning or adversarial imitation
"""
import os
import argparse
import numpy as np
import cv2
import configs
import tempfile
import torch as th
import torch.nn as nn
import json
import custom_wrappers as cw
import gymnasium.spaces as spaces

from diambra.arena import SpaceTypes
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
from stable_baselines3.common.vec_env import VecTransposeImage
from utils import load_agent, evaluate_policy_with_arcade_metrics, train_eval_split

from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import CnnRewardNet
from imitation.data import rollout
from imitation.util import logger as imit_logger

# diambra run -s _ python imitation.py --dataset_path _ --train_id _ --agent_num _ --policy_path _ --deterministic --num_players _ 

def get_transitions(data_loader: DiambraDataLoader, agent_num: int = None):
    """
    Transform episode recordings into trajectories for imitation learning algorithms.

    :param data_loader: (DiambraDataLoader) Diambra's data loader class, for loading episode recordings.
    :param agent_num: If episode recording data features two agents, need to specify the ID of the agent
    you want to imitate.
    """
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
    
    return rollout.flatten_trajectories_with_rew(trajectories)


def main(
    dataset_path_input: str, 
    train_id: str,
    agent_num: int,
    policy_path: str,
    deterministic: bool,
    num_players: int,
):
    assert train_id in configs.game_ids, f"Invalid game id ({train_id}), available ids: [{configs.game_ids}]"

    # Load configs
    settings_config = configs.env_settings
    ppo_config = configs.ppo_settings

    if dataset_path_input is not None:
        dataset_path = dataset_path_input
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, "dataset")

    # Set up imitation transitions
    imitation_data_loader = DiambraDataLoader(dataset_path)
    transitions = get_transitions(imitation_data_loader, agent_num=agent_num)
    print("\nTransitions loaded")

    # Load envs
    assert num_players in [1,2]
    if num_players == 1:
        train_settings, eval_settings, train_wrappers, eval_wrappers = configs.load_1p_settings(game_id=train_id)
    else:
        train_settings, eval_settings, train_wrappers, eval_wrappers = configs.load_2p_settings(game_id=train_id)
    if deterministic:
        # Due to DIAMBRA's implementation, selecting the same action each frame corresponds to holding down tthe button for that action.
        # This means that an attack the policy is trying to spam each frame won't come out after the first button press, as holding
        # an attack button leads to nothing happening. To allow the policy to play the game as it intends, this wrapper helps set any attack
        # which is the same as the previous frame's to 0, which stops the env interpreting that attack button as held down and allows the
        # policy to send out another attack. This is particularly important for deterministic policies, as they often converge to relying
        # on one or two attacks no matter the situation, and are particularly prone to this button hold "bug".
        eval_wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if eval_settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])

    num_train_envs = settings_config["num_train_envs"] 
    num_eval_envs = settings_config["num_eval_envs"]
    train_env, eval_env = train_eval_split(
        game_id=train_id,
        num_train_envs=num_train_envs,
        num_eval_envs=num_eval_envs,
        train_settings=train_settings,
        eval_settings=eval_settings,
        train_wrappers=train_wrappers,
        eval_wrappers=eval_wrappers,
        seed=settings_config["seed"]
    )
    # Transpose the env's images so that they have shape (C,H,W) instead of (H,W,C) (stable_baselines3 requires channel first observations)
    train_env, eval_env = VecTransposeImage(train_env), VecTransposeImage(eval_env)
    print(f"\nActivated {num_train_envs + num_eval_envs} environment(s)")

    model_checkpoint = ppo_config["model_checkpoint"]
    save_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(env=train_env, seed=settings_config["seed"], policy_path=checkpoint_path)

    # Set new imitation logger
    imit_log = imit_logger.configure(configs.tensor_board_folder, ["stdout", "tensorboard"])

    # Train behavioural clone trainer on transitions
    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(seed=configs.env_settings["seed"]),
        policy=agent.policy,
        device=configs.ppo_settings["device"],
        custom_logger=imit_log,
    )

    # Set up GAIL trainer
    reward_net = CnnRewardNet(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        hwc_format=False,
        use_action=False,
        use_next_state=True,
    )
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=512,
        gen_replay_buffer_capacity=256,
        n_disc_updates_per_round=8,
        venv=train_env,
        gen_algo=agent,
        reward_net=reward_net,
        custom_logger=imit_log,
        allow_variable_horizon=True,
    )

    eval_results = {}
    reward_infos, episode_lengths, stages_infos, arcade_infos = evaluate_policy_with_arcade_metrics(
        model=agent,
        env=eval_env,
        n_eval_episodes=configs.callbacks_settings["n_eval_episodes"],
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    eval_results.update({
        "results_before_imitation": {
            "rewards": reward_infos,
            "ep_lens": episode_lengths,
            "stages": stages_infos,
            "arcade": arcade_infos,
        }
    })

    # Imitation training
    try:
        if configs.imitation_settings["type"] == "imitate":
            trainer = bc_trainer
            trainer.train(n_epochs=configs.imitation_settings["bc"]["n_epochs"])
        elif configs.imitation_settings["type"] == "adv":
            trainer = gail_trainer
            trainer.train(configs.imitation_settings["gail"]["n_steps"], callback=None)
        else:
            raise ValueError(f"Expected imitation type to be 'imitate' or 'adv', received {configs.imitation_settings['type']} instead.")
    except KeyboardInterrupt:
        print("Ending imitation learning early, saving model")

    # Save imitated policy
    imitation_folder = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}", "trainer")
    agent.policy = trainer.policy
    agent.save(os.path.join(imitation_folder, "trainer_policy"))

    reward_infos, episode_lengths, stages_infos, arcade_infos = evaluate_policy_with_arcade_metrics(
        model=agent,
        env=eval_env,
        n_eval_episodes=configs.callbacks_settings["n_eval_episodes"],
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    eval_results.update({
        "results_after_imitation": {
            "rewards": reward_infos,
            "ep_lens": episode_lengths,
            "stages": stages_infos,
            "arcade": arcade_infos,
        }
    })

    train_env.close()
    eval_env.close()

    # Save evaluation results
    base_path = os.path.dirname(os.path.abspath(__file__))
    if policy_path:
        # Assumes relative path rather than absolute path
        policy_path_parts = policy_path.split(os.sep)
        model_path = os.path.join(*policy_path_parts[:2])
    else:
        model_path = os.path.join(configs.folders["parent_dir"], configs.folders["model_name"])
    file_path = os.path.join(
        base_path,
        model_path,
        "model",
        f"seed_{configs.env_settings['seed']}",
        "trainer",
        "imitation_learning_results.json"
    )
    with open(file_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to imitation trajectories")
    parser.add_argument("--train_id", type=str, required=False, help="Specific game to train on", default="sfiii3n")
    parser.add_argument("--num_players", type=int, required=False, help="Number of players in the env", default=1)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Whether to follow a deterministic or stochastic policy", default=True)
    parser.add_argument("--agent_num", type=int, required=False, help="Agent number (if trajectories come from multiagent env)", default=None)
    opt = parser.parse_args()

    main(
        dataset_path_input=opt.dataset_path,    # Path to episode recordings
        train_id=opt.train_id,                  # ID of game to train on
        agent_num=opt.agent_num,                # Agent number to retrieve actions from (if recordings are of 2p env)
        policy_path=opt.policy_path,            # Path to specific policy 
        deterministic=opt.deterministic,        # Whether to evaluate deterministic or stochastic policy
        num_players=opt.num_players,            # 1 or 2 player env
    )
