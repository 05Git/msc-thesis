import random
import torch
import numpy as np
import os
import time

import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings, RecordingSettings
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def set_global_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def make_sb3_envs(
    game_id: str,
    num_train_envs: int,
    num_eval_envs: int,
    train_env_settings: EnvironmentSettings=EnvironmentSettings(),
    eval_env_settings: EnvironmentSettings=EnvironmentSettings(),
    wrappers_settings: WrappersSettings=WrappersSettings(),
    episode_recording_settings: RecordingSettings=RecordingSettings(),
    render_mode: str="rgb_array", 
    seed: int=None, 
    start_index: int=0,
    allow_early_resets: bool=True, 
    start_method: str=None, 
    no_vec: bool=False,
    use_subprocess: bool=True, 
    log_dir_base: str="/tmp/DIAMBRALog/"
  ):
  def _make_sb3_env(rank, seed, env_settings):
        # Seed management
        env_settings.seed = int(time.time()) if seed is None else seed
        env_settings.seed += rank

        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                     episode_recording_settings, render_mode, rank=rank)

            # Create log dir
            log_dir = os.path.join(log_dir_base, str(rank))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            return env
        set_global_seed(env_settings.seed)
        return _init
  
  # If not wanting vectorized envs
  if no_vec and num_train_envs == 1 and num_eval_envs == 1:
      train_env = _make_sb3_env(0, seed)()
      eval_env = _make_sb3_env(1, seed)()
  else:
      # When using one environment, no need to start subprocesses
      if (num_train_envs == 1 and num_eval_envs == 1) or not use_subprocess:
          train_env = DummyVecEnv([_make_sb3_env(i + start_index, seed, train_env_settings) for i in range(num_train_envs)])
          start_index = num_train_envs
          eval_env = DummyVecEnv([_make_sb3_env(i + start_index, seed, eval_env_settings) for i in range(num_eval_envs)])
      else:
          train_env = SubprocVecEnv([_make_sb3_env(i + start_index, seed, train_env_settings) for i in range(num_train_envs)], start_method=start_method)
          start_index = num_train_envs
          eval_env = SubprocVecEnv([_make_sb3_env(i + start_index, seed, eval_env_settings) for i in range(num_eval_envs)], start_method=start_method)

  return train_env, eval_env