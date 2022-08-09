import random
import torch
import numpy as np
from misc.utils import make_env, generate_parameters

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    mode = ['expert', 'multitask-forward', 'multitask-irl', 'play-expert', 'play-multitask-forward',  'play-multitask-irl', 'irl']
    domain = ['highway', 'roundabout', 'fruitgrid']
    config = generate_parameters(mode=mode[0], domain=domain[0])

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config)

    # Train Model 
    if config.mode == "multitask-irl" or config.mode == 'play-multitask-irl':
        from agents.pretraining_irl import PretrainedAgent
        agent = PretrainedAgent(config, env, device)
        agent.inverse_train()

    elif not config.mode == 'irl':
        from agents.pretraining_forward import PretrainedAgent
        agent = PretrainedAgent(config, env, device)
        agent.train(env)
    else:
        from agents.irl import IRLAgent
        agent = IRLAgent(config, env, device)
        agent.inverse_train()
        agent.test_agent(env)


