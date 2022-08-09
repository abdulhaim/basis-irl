import os
import gym
import wandb
import torch
import numpy as np
import yaml

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def merge_configs(update, default):
    if isinstance(update,dict) and isinstance(default,dict):
        for k,v in default.items():
            if k not in update:
                update[k] = v
            else:
                update[k] = merge_configs(update[k],v)
    return update

def make_env(config):
    if config.domain == "highway":
        from envs.highway_domain import create_envs
        env = create_envs(config)
        return env
    elif config.domain == "fruitgrid":
        from envs.fruitgrid.pick import create_envs
        env = create_envs(config)

    elif config.domain == "roundabout":
        from envs.roundabout_domain import create_envs
        env = create_envs(config)
    else:
        raise NotImplementedError
    return env


def to_onehot(value, dim):
    """Convert batch of numbers to onehot
    Args:
        value (numpy.ndarray): Batch of numbers to convert to onehot. Shape: (batch,)
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray): Converted onehot. Shape: (batch, dim)
    """
    one_hot = torch.zeros(value.shape[0], dim)
    one_hot[torch.arange(value.shape[0]), value.long()] = 1
    return one_hot


def process_state(state, observation_shape):
    if len(observation_shape) == 3:
        state = torch.tensor(state)
        state = state.transpose(0, 2).transpose(1, 2)
        state = state.float().unsqueeze(0)  # swapped RGB dimension to come first
    return state

def generate_parameters(mode, domain):
    # set device
    os.environ["WANDB_API_KEY"] = 'INSERT_KEY' # insert key
    os.environ["WANDB_MODE"] = "online"

    # config parameters
    config_default = yaml.safe_load(open("config/default.yaml", "r"))
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))
    config_mode = yaml.safe_load(open("config/mode/" + mode + ".yaml", "r"))

    config_with_domain = merge_configs(config_domain, config_default)
    config = dotdict(merge_configs(config_mode, config_with_domain))

    wandb.init(project='experiments_basis_final', config=config)
    
    path_configs = {'model_name': config.mode + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'expert_model_path': "expert_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'load_model_path': config.load_model_start_path + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version}
    wandb.config.update(path_configs)

    print("CONFIG")
    print(wandb.config)

    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")

    # set all other train/ metrics to use this step
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not os.path.exists("models_irl/"):
        os.makedirs("models_irl/")

    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    wandb.run.name = config.model_name

    return wandb.config
