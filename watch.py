"""
watch.py: Observe a given policy's behaviour.
"""
import os
import argparse
import diambra.arena
import custom_wrappers as cw

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from diambra.arena import SpaceTypes
from utils import load_agent
from settings import load_settings

# diambra run -g python watch.py --cfg _ --policy_path _ --deterministic

def main(cfg: str, policy_path: str, deterministic: bool,):
    configs: dict = load_settings(cfg)
    # step_ratio > 1 will make gameplay too fast to observe normally
    settings = configs["train_settings"]
    settings.step_ratio = 1
    wrappers = configs["train_wrappers"]
    if deterministic:
        wrappers.wrappers.append([cw.NoOpWrapper, {
            "action_space_type": "discrete" if settings.action_space == SpaceTypes.DISCRETE else "multi_discrete",
            "no_attack": 0,
        }])

    env = diambra.arena.make(
        game_id=settings.game_id,
        env_settings=settings,
        wrappers_settings=wrappers,
    )
    # Transpose the env's images so that they have shape (C,H,W) instead of (H,W,C) (stable_baselines3 requires channel first observations)
    env = VecTransposeImage(DummyVecEnv([lambda: env]))
    
    model_checkpoint = configs["misc"]["model_checkpoint"]
    load_path = os.path.join(configs["folders"]["model_folder"], f"seed_{configs['misc']['seed']}")
    checkpoint_path = os.path.join(load_path, model_checkpoint) if not policy_path else policy_path
    agent = load_agent(settings_config=configs, env=env, policy_path=checkpoint_path, force_load=True)

    obs = env.reset()
    while True:
        # dist = agent.policy.get_distribution(th.tensor(obs).float().to(agent.device))
        # move_logits = dist.distribution[0].logits
        # act_logits = dist.distribution[1].logits
        # print("Move Logits:", move_logits)
        # print("Act Logits:", act_logits)
        actions, _ = agent.predict(obs, deterministic=deterministic)
        # print(f"Actions: {actions}")
        obs, rew, done, info = env.step(actions)
        # print(f"Observation: {obs}")
        # print(f"Reward: {rew}")
        # print(f"Info: {info}")
        
        if done:
            break

    env.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to settings config")
    parser.add_argument("--policy_path", type=str, required=False, help="Path to load pre-trained policy", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, required=False, help="Whether to follow a deterministic or stochastic policy", default=True)
    opt = parser.parse_args()

    main(cfg=opt.cfg, policy_path=opt.policy_path, deterministic=opt.deterministic)
