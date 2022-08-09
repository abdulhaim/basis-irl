import math
import random
import wandb
import torch
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count

from misc.memory import ReplayMemory
from misc.utils import process_state

class PretrainedAgent():
    def __init__(self, config, env, device):
        super(PretrainedAgent, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        # Create Networks
        if len(env[0].observation_shape) == 2:
            from networks.sequential_discrete import Pretraining
            self.observation_shape = env[0].observation_shape
        else:
            from networks.pixel_discrete import Pretraining
            self.observation_shape = (env[0].observation_shape[2], env[0].observation_shape[0], env[0].observation_shape[1])

        self.memory = ReplayMemory(self.observation_shape, self.config.n_actions, size=self.config.memory_size)

        self.policy_net = Pretraining(obs=self.observation_shape,
                                      n_actions=self.config.n_actions,
                                      cumulants=self.config.cumulants,
                                      tasks=self.config.num_tasks,
                                      hidden_size=self.config.hidden_size,
                                      mode=self.config.mode).to(self.device)

        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        if config.mode == "play-expert" or config.mode == "play-multitask-forward":
            self.policy_net.eval()
            self.target_net.eval()

        if self.config.load_model:
            policy_state_dict = torch.load("models/model_" + self.config.load_model_path + ".pth")
            target_state_dict = torch.load("models/target_" + self.config.load_model_path + ".pth")
            optimizer_state_dict = torch.load("models/optimizer_" + self.config.load_model_path + ".pth")

            self.policy_net.load_state_dict(policy_state_dict)
            self.target_net.load_state_dict(target_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.total_steps = 0

    def select_action(self, state, task_id):
        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
                        math.exp(-1. * self.total_steps / self.config.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                q, _ = self.policy_net(torch.tensor(state).to(self.device), task_id)
                action = q.max(1)[1]
        else:
            action = torch.tensor(random.randrange(self.config.n_actions), device=self.device, dtype=torch.long)

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/action_selected": action,
            "step/epsilon": eps_threshold
        }
        wandb.log(log_dict)
        return action

    def save_trajectories(self):
        traj_name = "traj/" + self.config.mode + "_domain_" + self.config.domain + "_expert_demons_" + str(self.config.demonstrations) + "_seed_" + str(self.config.seed)

        if self.config.mode == "play-multitask-forward":
            traj_name +=  str(self.config.seed) + "_task_" + self.config.traj_task_id

        np.save(traj_name + "_state.npy", arr=self.memory.state_buf[self.config.initial_memory:])
        np.save(traj_name + "_action.npy", arr=self.memory.action_buf[self.config.initial_memory:])
        np.save(traj_name + "_next_state.npy", arr=self.memory.next_state_buf[self.config.initial_memory:])
        np.save(traj_name + "_reward.npy", arr=self.memory.reward_buf[self.config.initial_memory:])
        np.save(traj_name + "_next_action.npy", arr=self.memory.next_action_buf[self.config.initial_memory:])
        np.save(traj_name + "_task_id.npy", arr=self.memory.done_buf[self.config.initial_memory:])
        np.save(traj_name + "_next_task_id.npy", arr=self.memory.done_buf[self.config.initial_memory:])
        np.save(traj_name + "_done.npy", arr=self.memory.done_buf[self.config.initial_memory:])
        exit()

    def optimize_model(self, data):
        state, action, next_state, reward, \
        next_action, task_id, next_task_id, done = \
            data['state'], data['action'], data['next_state'], data['reward'], \
            data['next_action'], data['task_id'], data['next_task_id'], data['done']

        action = action.to(torch.int64)
        reward = reward.unsqueeze(-1)
        next_action = next_action.to(torch.int64)
        task_id = task_id.long()
        next_task_id = next_task_id.long()
        done = done.unsqueeze(-1)

        assert state.shape == (self.config.batch_size, *self.observation_shape)
        assert action.shape == (self.config.batch_size, 1)
        assert next_state.shape == (self.config.batch_size, *self.observation_shape)
        assert reward.shape == (self.config.batch_size, 1)
        assert next_action.shape == (self.config.batch_size, 1)
        assert next_action.shape == (self.config.batch_size, 1)
        assert done.shape == (self.config.batch_size, 1)

        self.optimizer.zero_grad()
        q, rewards = self.policy_net(state, task_id)
        state_action_values = torch.gather(q, 1, action)

        target_q, target_rewards = self.target_net(next_state, next_task_id)
        next_state_values = torch.max(target_q, axis=-1)[0].unsqueeze(-1)
        assert next_state_values.shape == (self.config.batch_size, 1)
       
        expected_state_action_values = torch.logical_not(done) * (next_state_values * self.config.gamma) + reward
        assert state_action_values.shape == (self.config.batch_size, 1)
        assert expected_state_action_values.shape == (self.config.batch_size, 1)
        dqn_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        computed_reward_action = torch.gather(rewards, 1, action)
        assert reward.shape == (self.config.batch_size, 1)
        assert computed_reward_action.shape == (self.config.batch_size, 1)
        reward_loss = F.smooth_l1_loss(reward, computed_reward_action)

        phi_action = torch.gather(rewards, 1, action)
        next_psi_values = torch.gather(target_q, 1, next_action)
        expected_psi_values = torch.logical_not(done) * (next_psi_values * self.config.gamma) + phi_action.detach()
        assert expected_psi_values.shape == (self.config.batch_size, 1)
        itd_loss = F.smooth_l1_loss(state_action_values, expected_psi_values)

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/dqn_loss": dqn_loss,
            "step/reward_loss": reward_loss,
            "step/itd_loss": itd_loss

        }
        wandb.log(log_dict)
        loss = dqn_loss + reward_loss + itd_loss
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self, env):
        all_envs = env

        for episode in range(self.config.n_episodes):
            if self.config.multiple_tasks:
                if self.config.mode == "play-multitask-forward":
                    task_id = self.config.traj_task_id
                else:
                    task_id = random.randrange(self.config.num_tasks)
            else:
                task_id = 0
                
            env = all_envs[task_id]
            state = env.reset()
            state = process_state(state, self.observation_shape)

            action = self.select_action(state, task_id)
            total_reward = 0.0

            for t in count():
                self.total_steps += 1
                next_state, true_reward, done, info = env.step(action.item())
                next_state = process_state(next_state, self.observation_shape)

                if len(self.observation_shape) == 2:
                    reward = true_reward[0]
                else:
                    reward = true_reward

                total_reward += reward
                next_action = self.select_action(next_state, task_id)
                self.memory.store(state, action.detach(), next_state,
                                  reward, next_action.detach(), task_id, task_id, done)

                state = next_state
                action = next_action

                if self.total_steps > self.config.initial_memory:
                    if self.total_steps % self.config.update_every == 0:
                        data = self.memory.sample_batch(self.config.batch_size)
                        self.optimize_model(data)

                    if self.total_steps % self.config.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if done:
                    break


                if self.config.mode.find('play')!=-1 and (self.total_steps > self.config.demonstrations + self.config.initial_memory):
                    self.save_trajectories()

            if episode % self.config.record_episode == 0:
                log_dict = {
                    "episode/x_axis": episode,
                    "episode/episodic_reward_episode": total_reward,
                    "episode/length": t,
                    "step/x_axis": self.total_steps,
                    "step/episodic_reward_steps": total_reward
                }
                wandb.log(log_dict)


                if self.config.domain == "highway" or self.config.domain == "roundabout":
                    driving_dict =  {"step/reward_min": true_reward[1],
                                    "step/reward_max": true_reward[2],
                                    "step/collision_reward": true_reward[3],
                                    "step/lane_pref_reward": true_reward[4],
                                    "step/speed_reward": true_reward[5],
                                    "step/distance_reward": true_reward[6]}

                    wandb.log(driving_dict)
     
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.total_steps, episode, t, total_reward))
                if episode % self.config.save_model_episode == 0:
                    torch.save(self.policy_net.state_dict(), "models/model_" + self.config.model_name + ".pth")
                    torch.save(self.target_net.state_dict(), "models/target_" + self.config.model_name + ".pth")
                    torch.save(self.optimizer.state_dict(), "models/optimizer_" +  self.config.model_name + ".pth")

                    wandb.save("models/model_" + self.config.model_name + ".pth")
                    wandb.save("models/target_" + self.config.model_name + ".pth")
                    wandb.save("models/optimizer_" + self.config.model_name + ".pth")

        env.close()
        return
