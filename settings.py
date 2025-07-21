"""
settings.py: Load training and experiment settings from yaml configs.
"""
import os
import yaml
import torch as th
import custom_wrappers as cw

from stable_baselines3 import PPO
from RND import RNDPPO
from fusion_policy import MultiExpertFusionPolicy
from diambra.arena import EnvironmentSettings, EnvironmentSettingsMultiAgent, WrappersSettings, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule
from stable_baselines3.common.torch_layers import CombinedExtractor, NatureCNN

# IDs of games currently implemented
game_ids = [
    "sfiii3n",
    # "samsh5sp",
    # "kof98umh",
    # "umk3",
]

# Device
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_settings(cfg: str) -> dict:
    """
    Load settings from a yaml config.

    :param cfg: (str) Path to a yaml file.
    """
    print("Loading settings config...")
    # Dictionary of settings configs
    configs = {}

    config_file = open(cfg)
    params = yaml.load(config_file, Loader=yaml.FullLoader)
    config_file.close()

    param_keys: dict.keys = params.keys()
    misc: dict = params["misc"]
    assert misc["num_players"] in [1, 2]

    configs.update({"misc": misc})

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

    configs.update({"folders": {
        "parent_dir": folders["parent_dir"],
        "model_name": folders["model_name"],
        "model_folder": model_folder,
        "tensor_board_folder": tensor_board_folder
    }})

    if "teachers" in param_keys:
        teachers: dict = params["teachers"] # Dictionaries containing teacher IDs and paths
        for id, path in teachers.items():
            teacher = PPO.load(path=path, device=device)
            teachers[id] = teacher
    else:
        teachers = None
    
    configs.update({"teachers": teachers})

    env_settings: dict = params["env_settings"]
    general_settings: dict = env_settings["shared"]
    assert general_settings["game_id"] in game_ids
    general_settings["action_space"] = SpaceTypes.DISCRETE if general_settings["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    if misc["num_players"] == 2:
        general_settings["action_space"] = (general_settings["action_space"], general_settings["action_space"])

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
        "jump_bonus": cw.JumpBonus,
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

    configs.update({
        "train_settings": train_settings,
        "eval_settings": eval_settings,
        "train_wrappers": train_wrappers,
        "eval_wrappers": eval_wrappers,
    })

    callbacks_settings: dict = params["callbacks_settings"]
    configs.update({"callbacks_settings": callbacks_settings})

    policy_settings: dict = params["policy_settings"]
    assert policy_settings["batch_size"] % misc["num_train_envs"] == 0
    policy_settings["device"] = device
    policy_settings["seed"] = misc["seed"]
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

    if "rnd_settings" in param_keys:
        rnd_settings: dict = params["rnd_settings"]
        policy_settings.update(rnd_settings)
        agent_type = RNDPPO

    if "policy_kwargs" in param_keys:
        policy_kwargs: dict = params["policy_kwargs"]
        policy_settings.update({"policy_kwargs": policy_kwargs})

    if "fusion_settings" in param_keys:
        assert teachers is not None, "Must have a set of expert policies to give to the fusion policy."
        if "policy_kwargs" not in policy_settings.keys():
            policy_settings["policy_kwargs"] = dict()
        if policy_settings["policy"] == "CnnPolicy":
            # Image obs only
            policy_settings["policy_kwargs"]["features_extractor_class"] = NatureCNN
        elif policy_settings["policy"] == "MultiInputPolicy":
            # Image and vector obs
            policy_settings["policy_kwargs"]["features_extractor_class"] = CombinedExtractor
        else:
            raise ValueError(f"Policy type must be 'CnnPolicy' or 'MultiInputPolicy'.")
        policy_settings["policy"] = MultiExpertFusionPolicy
        configs.update({"fusion_settings": {
            "experts": teachers,
            "expert_params": params["fusion_settings"],
        }})

    if "distil_settings" in param_keys:
        distil_settings = params["distil_settings"]
        distil_settings["policy_settings"]["learning_rate"] = linear_schedule(
            initial_value=distil_settings["policy_settings"]["learning_rate"][0],
            final_value=distil_settings["policy_settings"]["learning_rate"][1]
        )
        distil_settings["tensorboard_log"] = tensor_board_folder
        distil_settings["policy_settings"]["device"] = device
        distil_settings["policy_settings"]["seed"] = misc["seed"]
        configs.update({"distil_settings": distil_settings})
    
    configs.update({"agent_type": agent_type, "policy_settings": policy_settings})

    if "imitation_settings" in param_keys:
        imitation_settings: dict = params["imitation_settings"]
        configs.update({"imitation_settings": imitation_settings})
    
    return configs
