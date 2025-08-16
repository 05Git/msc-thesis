"""
evaluate.py: Evaluate a policy and return the metrics. 
"""
import os
import json
import argparse
import numpy as  np
import custom_wrappers as cw

from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from utils import load_agent, evaluate_policy_with_arcade_metrics, eval_student_teacher_likelihood
from diambra.arena import SpaceTypes
from settings import load_settings

# diambra run -s _ python evaluate.py --cfg _ --eval_dir_name _ --policy_path _ --deterministic

def main(cfg: str, deterministic: bool, dir_name: str, policy_path: str):
    configs = load_settings(cfg)
    settings = configs["eval_settings"]
    wrappers = configs["eval_wrappers"]
    if deterministic:
        wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])

    eval_env, _ = make_sb3_env(
        game_id=settings.game_id,
        env_settings=settings,
        wrappers_settings=wrappers,
        seed = configs["misc"]["seed"]
    )
    eval_env = VecTransposeImage(eval_env)
    set_random_seed(configs["misc"]["seed"])

    model_checkpoint = configs["misc"]["model_checkpoint"]
    save_path = os.path.join(configs["folders"]["model_folder"], f"seed_{configs['misc']['seed']}")
    checkpoint_path = os.path.join(save_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(settings_config=configs, env=eval_env, policy_path=checkpoint_path, force_load=True)
    """
    reward_infos, episode_lengths, stages_infos, arcade_infos, kl_divs, \
    teacher_act_counts, teacher_act_means, teacher_act_stds = evaluate_policy_with_arcade_metrics(
        model=agent,
        env=eval_env,
        teachers=configs["teachers"],
        n_eval_episodes=configs["misc"]["n_eval_episodes"],
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    eval_results = {
        "model": checkpoint_path,
        "characters": settings.characters,
        "rewards_infos": reward_infos,
        "episode_lengths": episode_lengths,
        "stages_infos": stages_infos,
        "arcade_runs_infos": arcade_infos,
        "kl_divergences": kl_divs,
        "mean_reward": np.mean(reward_infos),
        "std_reward": np.std(reward_infos),
        "mean_stages": np.mean(stages_infos),
        "std_stages": np.std(stages_infos),
        "mean_arcade_runs": np.mean(arcade_infos),
        "std_arcade_runs": np.std(arcade_infos),
        "mean_kl_divs": {id: np.mean(kl_div) for id, kl_div in kl_divs.items()},
        "std_kl_divs": {id: np.std(kl_div) for id, kl_div in kl_divs.items()},
        "teacher_likelihood_scores": teacher_act_counts,
        "teacher_likelihood_means": teacher_act_means,
        "teacher_likelihood_stds": teacher_act_stds,
    }
    """
    teacher_act_counts, teacher_act_means, teacher_act_stds = eval_student_teacher_likelihood(
        student=agent,
        teachers=configs["teachers"],
        env=eval_env,
        n_eval_episodes=configs["misc"]["n_eval_episodes"],
        deterministic=deterministic,
    )
    eval_results = {
        "teacher_likelihood_scores": teacher_act_counts,
        "teacher_likelihood_means": teacher_act_means,
        "teacher_likelihood_stds": teacher_act_stds,
    }
    eval_env.close()

    # Save evaluation results
    base_path = os.path.dirname(os.path.abspath(__file__))
    if policy_path:
        policy_path_parts = policy_path.split(os.sep)
        model_path = os.path.join(*policy_path_parts[:2]) # Assumes relative path instead of absolute path
    else:
        model_path = os.path.join(configs["folders"]["parent_dir"], configs["folders"]["model_name"])
    results_save_path = os.path.join(
        base_path,
        model_path,
        "evaluations",
        f"seed_{configs['misc']['seed']}",
        dir_name,
    )
    os.makedirs(results_save_path, exist_ok=True)
    with open(os.path.join(results_save_path, "results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to settings config")
    parser.add_argument("--eval_dir_name", type=str, required=False, help="Name of evaluations directory", default="evaluation_results")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Evaluate deterministic or stochastic policy", default=True)
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    opt = parser.parse_args()

    main(
        cfg=opt.cfg,
        deterministic=opt.deterministic,
        dir_name=opt.eval_dir_name,
        policy_path=opt.policy_path,
    )
