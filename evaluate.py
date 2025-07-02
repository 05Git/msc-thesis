import os
import json
import argparse
import configs
import torch as th
import numpy as  np
import custom_wrappers as cw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from utils import evaluate_policy_with_arcade_metrics, eval_student_teacher_likelihood
from diambra.arena import SpaceTypes

# diambra run -s _ python evaluate.py --eval_id _ --num_players _ --dir_name _ --policy_path _ --deterministic

def main(
    eval_id: str,
    num_players: int,
    deterministic: bool,
    dir_name: str,
    policy_path: str,
):
    assert eval_id in configs.game_ids, f"Invalid game id ({eval_id}), available ids: [{configs.game_ids}]"

    # Load configs
    settings_config = configs.env_settings
    ppo_config = configs.ppo_settings

    # Load envs
    assert num_players in [1,2]
    if num_players == 1:
        _, eval_settings, _, eval_wrappers = configs.load_1p_settings(game_id=eval_id)
    else:
        _, eval_settings, _, eval_wrappers = configs.load_2p_settings(game_id=eval_id)
    if deterministic:
        eval_wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if eval_settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])

    eval_env, num_envs = make_sb3_env(
        game_id=eval_id,
        env_settings=eval_settings,
        wrappers_settings=eval_wrappers,
        seed = settings_config["seed"]
    )
    eval_env = VecTransposeImage(eval_env)
    set_random_seed(settings_config["seed"])

    model_checkpoint = ppo_config["model_checkpoint"]
    save_path = os.path.join(configs.model_folder, f"seed_{settings_config['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = PPO.load(
        path=checkpoint_path,
        env=eval_env,
        policy_kwargs=configs.policy_kwargs,
        device=configs.ppo_settings["device"],
        custom_objects={
            "action_space" : eval_env.action_space,
            "observation_space" : eval_env.observation_space,
        }
    )

    reward_infos, episode_lengths, stages_infos, arcade_infos = evaluate_policy_with_arcade_metrics(
        model=agent,
        env=eval_env,
        n_eval_episodes=configs.callbacks_settings["n_eval_episodes"],
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    
    if configs.teacher_paths:
        teacher_act_counts, teacher_act_means, teacher_act_stds = eval_student_teacher_likelihood(
            student=agent,
            num_teachers=len(configs.teacher_paths),
            env=eval_env,
            n_eval_episodes=configs.callbacks_settings["n_eval_episodes"],
            deterministic=deterministic,
        )

    eval_env.close()

    eval_results = {
        "model": checkpoint_path,
        "characters": eval_settings.characters,
        "rewards_infos": reward_infos,
        "episode_lengths": episode_lengths,
        "stages_infos": stages_infos,
        "arcade_runs_infos": arcade_infos,
        "mean_reward": np.mean(reward_infos),
        "std_reward": np.std(reward_infos),
        "mean_stages": np.mean(stages_infos),
        "std_stages": np.std(stages_infos),
        "mean_arcade_runs": np.mean(arcade_infos),
        "std_arcade_runs": np.std(arcade_infos),
    }

    if configs.teacher_paths:
        eval_results.update({
            "teacher_likelihood_scores": teacher_act_counts,
            "teacher_likelihood_means": teacher_act_means,
            "teacher_likelihood_stds": teacher_act_stds,
        })

    # Save evaluation results
    base_path = os.path.dirname(os.path.abspath(__file__))
    if policy_path:
        policy_path_parts = policy_path.split(os.sep)
        model_path = os.path.join(*policy_path_parts[:2])
    else:
        model_path = os.path.join(configs.folders["parent_dir"], configs.folders["model_name"])
    results_save_path = os.path.join(
        base_path,
        model_path,
        "evaluations",
        dir_name,
    )
    os.makedirs(results_save_path, exist_ok=True)
    with open(os.path.join(results_save_path, "results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir_name", type=str, required=False, help="Name of evaluations directory", default="evaluation_results")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Evaluate deterministic or stochastic policy", default=True)
    parser.add_argument("--eval_id", type=str, required=False, help="Specific game to eval on", default="sfiii3n")
    parser.add_argument("--num_players", type=int, required=False, help="Number of players in the env", default=1)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    opt = parser.parse_args()

    main(
        deterministic=opt.deterministic,
        dir_name=opt.eval_dir_name,
        eval_id=opt.eval_id,
        num_players=opt.num_players,
        policy_path=opt.policy_path,
    )
