"""
configs.py: _.
"""
import os
import yaml
import torch as th
import custom_wrappers as cw

from stable_baselines3 import PPO
from RND import RNDPPO
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule

# IDs of games currently implemented
game_ids = [
    "sfiii3n",
    # "samsh5sp",
    # "kof98umh",
    # "umk3",
]

# Device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

config_path = "configs/test_config.yaml"
config_file = open(config_path)
params = yaml.load(config_file, Loader=yaml.FullLoader)
config_file.close()

param_keys: dict.keys = params.keys()
misc: dict = params["misc"]
assert misc["num_players"] in [1, 2]

env_settings: dict = params["env_settings"]
general_settings: dict = env_settings["shared"]
assert general_settings["game_id"] in game_ids
general_settings["action_space"] = SpaceTypes.DISCRETE if general_settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
if misc["num_players"] == 2:
    general_settings["action_space"] = (general_settings["action_space"], general_settings["action_space"])

# Set the path that the model and tensorboards will be saved to
base_path = os.path.dirname(os.path.abspath(__file__))
folders: dict = params["folders"]
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

if "teachers" in param_keys:
    teachers: dict = params["teachers"] # Dictionaries containing teacher IDs and paths
    for id, path in teachers.items():
        teacher = PPO.load(path=path, device=device)
        teachers[id] = teacher
    teacher_probabilities: list[float] = params["teacher_probabilities"]
else:
    teachers = None
    teacher_probabilities = None

train_settings: dict = env_settings["train"]
train_settings.update(general_settings.copy())
eval_settings: dict = env_settings["eval"]
eval_settings.update(general_settings.copy())

wrappers_settings: dict = params["wrappers_settings"]
wrapper_aliases = {
    "pixelobs": cw.PixelObsWrapper,
    "teacher_input": cw.TeacherInputWrapper,
    "add_last_action": cw.AddLastActions,
    "action_mask": cw.ActionMaskWrapper,
    "opp_controller": cw.OpponentController,
    "interleave": cw.InterleavingWrapper,
    "def_train": cw.DefTrainWrapper,
    "att_train": cw.AttTrainWrapper,
    "2ptrain": cw.TwoPTrainWrapper,
}
for wrapper in wrappers_settings["wrappers"]:
    assert wrapper[0] in wrapper_aliases.keys()
    if wrapper[0] == "teacher_input":
        wrapper[1]["teachers"] = teachers
    wrapper[0] = wrapper_aliases[wrapper[0]]

train_wrappers: dict = wrappers_settings.copy()
train_wrappers["wrappers"] = wrappers_settings["wrappers"].copy()
if type(train_settings["characters"]) == list:
    train_wrappers["wrappers"].append([cw.InterleavingWrapper, {
        "character_list": train_settings["characters"],
        "one_p_env": misc["num_players"] == 1,
    }])
    train_settings["characters"] = train_settings["characters"][0]

eval_wrappers: dict = wrappers_settings.copy()
eval_wrappers["wrappers"] = wrappers_settings["wrappers"].copy()
if type(eval_settings["characters"]) == list:
    eval_wrappers["wrappers"].append([cw.InterleavingWrapper, {
        "character_list": eval_settings["characters"],
        "one_p_env": misc["num_players"] == 1,
    }])
    eval_settings["characters"] = eval_settings["characters"][0]

if misc["num_players"] == 1:
    train_settings = load_settings_flat_dict(EnvironmentSettings, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettings, eval_settings)
else:
    train_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, train_settings)
    eval_settings = load_settings_flat_dict(EnvironmentSettingsMultiAgent, eval_settings)
train_wrappers = load_settings_flat_dict(WrappersSettings, train_wrappers)
eval_wrappers = load_settings_flat_dict(WrappersSettings, eval_wrappers)

policy_settings: dict = params["policy_settings"]
assert policy_settings["batch_size"] % misc["num_train_envs"] == 0
policy_settings.update({"device": device})
policy_settings.update({"seed": misc["seed"]})
policy_settings["learning_rate"] = linear_schedule(
    policy_settings["learning_rate"][0],
    policy_settings["learning_rate"][1]
)
policy_settings["clip_range"] = linear_schedule(
    policy_settings["clip_range"][0],
    policy_settings["clip_range"][1]
)
policy_settings["clip_range_vf"] = policy_settings["clip_range"]
policy_settings["tensorboard_log"] = tensor_board_folder
agent_type = PPO

callbacks_settings: dict = params["callbacks_settings"]

if "rnd_settings" in param_keys:
    rnd_settings: dict = params["rnd_settings"]
    policy_settings.update(rnd_settings)
    agent_type = RNDPPO

if "imitation_settings" in param_keys:
    imitation_settings: dict = params["imitation_settings"]

if "policy_kwargs" in param_keys:
    policy_kwargs: dict = params["policy_kwargs"]
    policy_settings.update({"policy_kwargs": policy_kwargs})

"""
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
    "add_teacher_obs": False,
}

# Paths to teach policies for distillation
teacher_paths = [
    # "experts/bc_policies/antiair_simple/model/seed_0/trainer/trainer_policy.zip",
    # "experts/bc_policies/jump_ins_simple/model/seed_0/trainer/trainer_policy.zip",
    # "experts/bc_policies/defence_simple/model/seed_0/trainer/trainer_policy.zip",
    "experts/from_scratch/antiair/model/seed_0/500000.zip",
    "experts/from_scratch/attack_rand/model/seed_0/500000.zip",
    "experts/from_scratch/defence/model/seed_0/500000.zip",
    "final_policies/base_agents/ryu_vanilla/model/seed_0/2000000.zip",
]
teacher_probabilities = []

def load_teachers():
    teachers = []
    for path in teacher_paths:
        teacher = PPO.load(path=path, device=policy_settings["device"])
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
        "opp_type": "rand"
    }],
    # [cw.ActionMaskWrapper, {
    #     "action_space": "discrete" if env_settings["2_player"]["shared"]["action_space"][0] == SpaceTypes.DISCRETE else "multi_discrete",
    #     "no_op": 0,
    #     "num_players": 2,
    #     "max_actions": [9,11],
    # }],
    [cw.AttTrainWrapper, {}],
    [cw.PixelObsWrapper, {}],
    [cw.TwoPTrainWrapper, {}],
]

# Director path the policy, tensorboard and other items will be saved to
folders = {
    "parent_dir": "distil_policies",
    "model_name": "finetuned_ryu_vanilla",
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

policy_settings = {
    "policy_type": "student-distil",   # ppo / rnd / student-distil
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
    
    Load settings for a 1 player env.

    :param game_id: ID of which game to load an env for. 
    
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
    if wrappers_options["add_teacher_obs"]:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": policy_settings["time_steps"],
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
    
    Load settings for a 2 player env.

    :param game_id: ID of which game to load an env for. 
    
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
    if wrappers_options["add_teacher_obs"]:
        teacher_wrapper = [cw.TeacherInputWrapper, {
            "teachers": load_teachers(),
            "timesteps": policy_settings["time_steps"],
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
"""