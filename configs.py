"""
configs.py: Adjust the settings and hyperparams for training and evaluating policies
Design follows example set by ___
"""
import os
import torch as th
import custom_wrappers as cw

from stable_baselines3 import PPO
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, SpaceTypes, load_settings_flat_dict
from filter_keys import get_filter_keys
from custom_networks import  CustomCNN


action_space = SpaceTypes.MULTI_DISCRETE    # Can be either DISCRETE or MULTI_DISCRETE spaces
frame_shape = (4, 84, 84)                   # SB3 expects channel first
env_settings = {                            # Adjustable environment settings
    "1_player": {
        "shared": {
            "step_ratio": 6,
            "frame_shape": (frame_shape[1], frame_shape[2], 1), # First two elements: H and W. Third element: 1 for greyscale, 0 for RGB.
            "continue_game": 0.,                                # Probability between 1 and 0 of an agent continuing an arcade run after losing, instead of starting a new episode
            "action_space": action_space,
            "outfits": 1,
            "splash_screen": False,
        },
        "sfiii3n": {
            "train": {
                "game_id": "sfiii3n",
                "characters": "Ryu",
                "difficulty": 6,
                "super_art": 1,
            },
            "eval": {
                "game_id": "sfiii3n",
                "characters": "Ryu",
                "difficulty": 6,
                "super_art": 1,
            },
        },
    },
    "2_player": {
        "shared": {
            "step_ratio": 6,
            "frame_shape": (frame_shape[1], frame_shape[2], 1),
            "continue_game": 0.,
            "action_space": (action_space, action_space),
            "outfits": (1, 1),
            "splash_screen": False,
        },
        "sfiii3n": {
            "train": {
                "characters": ("Ryu", None),
                "difficulty": 6,
                "super_art": (1, 1),
            },
            "eval": {
                "characters": [
                    ("Ryu", "Alex"),
                    ("Ryu", "Gouki"),
                    ("Ryu", "Hugo"),
                    ("Ryu", "Ibuki"),
                    ("Ryu", "Chun-Li"),
                    ("Ryu", "Makoto"),
                    ("Ryu", "Q"),
                ],
                "difficulty": 6,
                "super_art": (1, 1),
            },
        },
    },
    "seed": 0,
    "num_train_envs": 8,
    "num_eval_envs": 4,
}

wrappers_settings = {
    "normalize_reward": True,
    "normalization_factor": 1.0,                # See __ for reward and normalized reward functions
    "no_attack_buttons_combinations": False,    # Set to False to allow full range of attacks
    "stack_frames": 4,
    "dilation": 1,
    "add_last_action": False,
    "stack_actions": 1,
    "repeat_action": 1,
    "scale": False,
    "exclude_image_scaling": True,
    "role_relative": False,
    "filter_keys": [],
    "flatten": True,
    "wrappers": [],
}

# Choose whether to use specific custom wrappers or not
wrappers_options = {
    "add_last_actions": False,
    "use_teachers": False,
}

# Paths to teach policies for distillation
teacher_paths = [
    "experts/ryu_vs_alex/attack_expert_rand/model/seed_0/500000.zip",
    "experts/ryu_vs_alex/def_expert_rand/model/seed_0/500000.zip",
    "experts/ryu_vs_alex/anti_air_expert/model/seed_0/500000.zip",
]

def load_teachers():
    teachers = []
    for path in teacher_paths:
        teacher = PPO.load(path=path, device=ppo_settings["device"])
        teachers.append(teacher)
    
    return teachers

# List of custom wrappers for 1 and 2 player envs
wrappers_1p = [
    # [cw.ActionMaskWrapper, {
    #     "action_space": "discrete" if env_settings["1_player"]["shared"]["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
    #     "no_op": 0,
    #     "num_players": 1,
    #     "max_actions": [9,11],
    # }],
    [cw.PixelObsWrapper, {}],
]

wrappers_2p = [
    [cw.OpponentController, {
        "opp_type": "no_op"
    }],
    # [cw.ActionMaskWrapper, {
    #     "action_space": "discrete" if env_settings["2_player"]["shared"]["action_space"][0] == SpaceTypes.DISCRETE else "multi_discrete",
    #     "no_op": 0,
    #     "num_players": 2,
    #     "max_actions": [9,11],
    # }],
    [cw.AttTrainWrapper, {}],
    [cw.PixelObsWrapper, {}],
    [cw.BCTrainerWrapper, {}],
]

# Director path the policy, tensorboard and other items will be saved to
folders = {
    "parent_dir": "experts/bc_policies",
    "model_name": "antiair_simple",
}

# See __ for how to set custom architecture with policy_kwargs
policy_kwargs = {
    # "net_arch": {"pi": [64, 64], "vf": [32, 32]},
    # "features_extractor_class": CustomCNN,
    # "features_extractor_kwargs": {"features_dim": 1024},
}

n_steps = 128
nminibatches = 8
batch_lambda = 8
batch_size = ((n_steps * env_settings["num_train_envs"]) // nminibatches) * batch_lambda
assert (n_steps * env_settings["num_train_envs"]) % nminibatches == 0

ppo_settings = {
    "policy": "CnnPolicy",
    "model_checkpoint": "0",
    "time_steps": 2_000_000,
    "device": th.device("cuda" if th.cuda.is_available else "cpu"),
    "gamma": 0.99,
    "train_lr": (2.5e-5, 2.5e-6),
    "finetune_lr": (5.0e-5, 2.5e-6),
    "train_cr": (0.15, 0.025),
    "finetune_cr": (0.075, 0.025),
    "batch_size": batch_size,
    "n_epochs": 10,
    "n_steps": n_steps,
    "gae_lambda": 0.95,
    "ent_coef": 0.,
    "vf_coef": 0.5,
    "target_kl": None,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "normalize_advantage": True,
    "stats_window_size": 100,
    "use_rnd": False,        # Set to True to use Random Network Distillation (enhances policy exploration)
    "rnd_int_beta": 1e-3,   # Some float less than 0, to reduce the scale of the intrinsic reward
    "rnd_model_args": {     # RND model arguments
        "image_shape": frame_shape,
        "action_size": 2 if action_space == SpaceTypes.MULTI_DISCRETE else 1,
        "vec_fc_size": 128,
        "feature_size": 128, # Best behaviour so far: 128
        "rnd_type": "state",
        "optim_args": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.,
        }
    },
}

# Setting for which callbacks to use and how during training
callbacks_settings = {
    "autosave_freq": 500_000,
    "n_eval_episodes": 100,
    "eval_freq": 250_000,
    "evaluate_during_training": False,
    "stop_training_if_no_improvement": False,
    "measure_action_similarity": False,
}

# Behavioural cloning and GAIL settings
imitation_settings = {
    "type": "imitate", # imitate / adv
    "bc": {
        "n_epochs": 50
    },
    "gail": {
        "n_steps": 1_000_000
    },
}

def load_1p_settings(game_id: str):
    """
    Load settings for a 1 player env.

    :param game_id: ID of which game to load an env for. 
    """
    # Load the settings shared between the train and eval envs into a dictionary,
    # then create specific dictionaries for the train and eval envs
    general_settings = env_settings["1_player"]["shared"]
    game_settings = env_settings["1_player"][game_id]
    train_settings = game_settings["train"]
    eval_settings = game_settings["eval"]

    wrappers_settings["filter_keys"] = get_filter_keys(game_id, num_players=1) # Set the observation filter keys to use
    train_wrappers = wrappers_settings.copy()
    train_wrappers["wrappers"] = wrappers_1p.copy() # Set list of custom wrappers to use
    if type(train_settings["characters"]) == list:  # Add InterleavingWrapper if a list of characters is given as argument
        train_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": train_settings["characters"],
            "one_p_env": True,
        }])
    
    # Same procedure for eval env
    eval_wrappers = wrappers_settings.copy()
    eval_wrappers["wrappers"] = wrappers_1p.copy()
    if type(eval_settings["characters"]) == list:
        eval_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": eval_settings["characters"],
            "one_p_env": True,
        }])
    
    # EnvironmentSettings expects a str or tuple of str, not a list. If a list of characters was passed because
    # of interleaving, then set the initial character to be used as the first character in the list
    if type(train_settings["characters"]) == list:
        train_settings["characters"] = train_settings["characters"][0]
    
    if type(eval_settings["characters"]) == list:
        eval_settings["characters"] = eval_settings["characters"][0]

    train_settings.update(general_settings.copy())
    eval_settings.update(general_settings.copy())
    train_settings = load_settings_flat_dict(EnvironmentSettings, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)
    train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)
    eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)
    
    # Load teachers for policy distillation
    # This is done after loading the env settings classes to circumvent stable_baselines3 not allowing nested observation spaces
    if wrappers_options["use_teachers"]:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": ppo_settings["time_steps"],
            "deterministic": True,
            "teacher_action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
            "use_teacher_actions": False,
            "initial_epsilon": 1.,
        }]
        train_wrappers.wrappers.append(teacher_wrapper)
        eval_wrappers.wrappers.append(teacher_wrapper)
    
    # Also done after loading env settings classess to circumvent nested observation space check
    if wrappers_options["add_last_actions"]:
        add_last_actions_wrapper = [cw.AddLastActions, {
            "action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
            "action_history_len": 8,
            "use_similarity_penalty": False,
            "similarity_penalty_alpha": 1e-3,
        }]
        train_wrappers.wrappers.append(add_last_actions_wrapper)
        eval_wrappers.wrappers.append(add_last_actions_wrapper)
    
    return train_settings, eval_settings, train_wrappers, eval_wrappers

def load_2p_settings(game_id: str):
    """
    Load settings for a 2 player env.

    :param game_id: ID of which game to load an env for. 
    """
    # Load the settings shared between the train and eval envs into a dictionary,
    # then create specific dictionaries for the train and eval envs
    general_settings = env_settings["2_player"]["shared"]
    game_settings = env_settings["2_player"][game_id]
    train_settings = game_settings["train"]
    eval_settings = game_settings["eval"]

    wrappers_settings["filter_keys"] = get_filter_keys(game_id, num_players=2) # Set the observation filter keys to use
    train_wrappers = wrappers_settings.copy()
    train_wrappers["wrappers"] = wrappers_2p.copy()# Set list of custom wrappers to use
    if type(train_settings["characters"]) == list:  # Add InterleavingWrapper if a list of characters is given as argument
        train_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": train_settings["characters"],
            "one_p_env": False,
        }])
    
    # Same procedure for eval env
    eval_wrappers = wrappers_settings.copy()
    eval_wrappers["wrappers"] = wrappers_2p.copy()
    if type(eval_settings["characters"]) == list:
        eval_wrappers["wrappers"].append([cw.InterleavingWrapper, {
            "character_list": eval_settings["characters"],
            "one_p_env": False,
        }])

    # EnvironmentSettings expects a str or tuple of str, not a list. If a list of characters was passed because
    # of interleaving, then set the initial character to be used as the first character in the list
    if type(train_settings["characters"]) == list:
        train_settings["characters"] = train_settings["characters"][0]
        
    if type(eval_settings["characters"]) == list:
        eval_settings["characters"] = eval_settings["characters"][0]

    train_settings.update(general_settings.copy())
    eval_settings.update(general_settings.copy())
    train_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, eval_settings)
    train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)
    eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)

    # Load teachers for policy distillation
    # This is done after loading the env settings classes to circumvent stable_baselines3 not allowing nested observation spaces
    if wrappers_options["use_teachers"]:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": ppo_settings["time_steps"],
            "deterministic": True,
            "teacher_action_space": "discrete" if general_settings["action_space"][0] == SpaceTypes.DISCRETE else "multi_discrete",
            "use_teacher_actions": False,
            "initial_epsilon": 1.,
        }]
        train_wrappers.wrappers.append(teacher_wrapper)
        eval_wrappers.wrappers.append(teacher_wrapper)
    
    # Also done after loading env settings classess to circumvent nested observation space check
    if wrappers_options["add_last_actions"]:
        add_last_actions_wrapper = [cw.AddLastActions, {
            "action_space": "discrete" if general_settings["action_space"] == SpaceTypes.DISCRETE else "multi_discrete",
            "action_history_len": 6,
            "use_similarity_penalty": True,
            "similarity_penalty_alpha": 1e-3,
        }]
        train_wrappers.wrappers.append(add_last_actions_wrapper)
        eval_wrappers.wrappers.append(add_last_actions_wrapper)
    
    return train_settings, eval_settings, train_wrappers, eval_wrappers

# Set the path that the model and tensorboards will be saved to
base_path = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(
    base_path,
    folders["parent_dir"],
    folders["model_name"],
    "model"
)
tensor_board_folder = os.path.join(
    base_path,
    folders["parent_dir"],
    folders["model_name"],
    "tb"
)
# Make sure these paths exist
os.makedirs(model_folder, exist_ok=True)
os.makedirs(tensor_board_folder, exist_ok=True)

# IDs of games currently implemented
game_ids = [
    "sfiii3n",
    # "samsh5sp",
    # "kof98umh",
    # "umk3",
]
