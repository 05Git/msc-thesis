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
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import RewardNet, BasicRewardNet, CnnRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import rollout
from imitation.util import logger as imit_logger

# diambra run -s _ python imitation.py --dataset_path _ --train_id _ --agent_num _ --policy_path _ --deterministic --num_players _

class MultiDiscreteCnnRewardNet(CnnRewardNet):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        hwc_format: bool = True,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.hwc_format = hwc_format

        if not (self.use_state or self.use_next_state):
            raise ValueError("CnnRewardNet must take current or next state as input.")

        if not preprocessing.is_image_space(observation_space):
            raise ValueError(
                "CnnRewardNet requires observations to be images.",
            )
        assert isinstance(observation_space, spaces.Box)

        if self.use_action and not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError(
                "MultiDiscreteCnnRewardNet can only use MultiDiscrete action spaces.",
            )

        input_size = 0
        output_size = 1

        if self.use_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_action:
            assert isinstance(action_space, spaces.Discrete)
            output_size = int(action_space.n)

        if self.use_next_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_done:
            output_size *= 2

        full_build_cnn_kwargs: Dict[str, Any] = {
            "hid_channels": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_channels": input_size,
            "out_size": output_size,
            "squeeze_output": output_size == 1,
        }

        self.cnn = networks.build_cnn(**full_build_cnn_kwargs)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        inputs = []
        if self.use_state:
            state_ = cnn_transpose(state) if self.hwc_format else state
            inputs.append(state_)
        if self.use_next_state:
            next_state_ = cnn_transpose(next_state) if self.hwc_format else next_state
            inputs.append(next_state_)

        inputs_concat = th.cat(inputs, dim=1)
        outputs = self.cnn(inputs_concat)
        if self.use_action and not self.use_done:
            # for discrete action spaces, action is passed to forward as a one-hot
            # vector.
            rewards = th.sum(outputs * action, dim=1)
        elif self.use_action and self.use_done:
            # here, we double the size of the one-hot vector, where the first entries
            # are for done=False and the second are for done=True.
            action_done_false = action * (1 - done[:, None])
            action_done_true = action * done[:, None]
            full_acts = th.cat((action_done_false, action_done_true), dim=1)
            rewards = th.sum(outputs * full_acts, dim=1)
        elif not self.use_action and self.use_done:
            # here we turn done into a one-hot vector.
            dones_binary = done.long()
            dones_one_hot = nn.functional.one_hot(dones_binary, num_classes=2)
            rewards = th.sum(outputs * dones_one_hot, dim=1)
        else:
            rewards = outputs
        return rewards
    

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
    train_env, eval_env = VecTransposeImage(train_env), VecTransposeImage(eval_env)
    print(f"\nActivated {num_train_envs + num_eval_envs} environment(s)")

    model_checkpoint = ppo_config["model_checkpoint"]
    save_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(env=train_env, seed=settings_config["seed"], policy_path=checkpoint_path)

    # Set new imitation logger
    imit_log = imit_logger.configure(configs.tensor_board_folder, ["stdout", "tensorboard"])

    # Set up GAIL trainer
    # reward_net = CnnRewardNet(
    #     observation_space=train_env.observation_space,
    #     action_space=train_env.action_space,
    #     normalize_input_layer=RunningNorm,
    #     hwc_format=False,
    # )
    # gail_trainer = GAIL(
    #     demonstrations=transitions,
    #     demo_batch_size=1028,
    #     gen_replay_buffer_capacity=512,
    #     n_disc_updates_per_round=8,
    #     venv=train_env,
    #     gen_algo=agent,
    #     reward_net=reward_net,
    #     custom_logger=imit_log,
    # )
    
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
        bc_trainer.train(n_epochs=configs.imitation_settings["bc"]["n_epochs"])
        # gail_trainer.train(configs.imitation_settings["gail"]["n_steps"], callback=None)
        # with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        #     print(tmpdir)
        #     dagger_trainer = SimpleDAggerTrainer(
        #         venv=train_env,
        #         scratch_dir=tmpdir,
        #         expert_policy=agent,
        #         bc_trainer=bc_trainer,
        #         rng=np.random.default_rng(seed=seed),
        #         custom_logger=imit_log,
        #     )
        #     dagger_trainer.train(configs.imitation_settings["dagger"]["n_steps"])
    except KeyboardInterrupt:
        print("Ending imitation learning early, saving model")

    # Save imitated policy
    imitation_folder = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}", "bc_trainer")
    agent.policy = bc_trainer.policy
    agent.save(os.path.join(imitation_folder, "bc_agent_policy"))

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
        policy_path_parts = policy_path.split(os.sep)
        model_path = os.path.join(*policy_path_parts[:2])
    else:
        model_path = os.path.join(configs.folders["parent_dir"], configs.folders["model_name"])
    file_path = os.path.join(
        base_path,
        model_path,
        "model",
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
        dataset_path_input=opt.dataset_path,
        train_id=opt.train_id,
        agent_num=opt.agent_num,
        policy_path=opt.policy_path,
        deterministic=opt.deterministic,
        num_players=opt.num_players,
    )
