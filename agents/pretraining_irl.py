import math
import random
import wandb
import torch
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from misc.memory import ReplayMemory
from misc.utils import process_state
from itertools import count

from misc.utils import to_onehot
import matplotlib.pyplot as plt
from PIL import Image


class PhaseIAgent():
    def __init__(self, config, env, device):
        super(PhaseIAgent, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.memory_total_buff = []

        # Create Networks
        if len(env[0].observation_shape) == 2:
            from networks.sequential_discrete import Pretraining
            self.observation_shape = env[0].observation_shape
        else:
            from networks.pixel_discrete import Pretraining
            self.observation_shape = (env[0].observation_shape[2], env[0].observation_shape[0], env[0].observation_shape[1])

        for i in range(self.config.num_tasks):
            buffer = ReplayMemory(self.observation_shape, self.config.n_actions, size=self.config.memory_size)
            
            traj_name = "traj/play-expert_domain_" + self.config.domain + "_expert_demons_" + str(self.config.demonstrations) + "_seed_" + str(self.config.seed) + "_task_" + str(i)
            buffer.state_buf = np.load(traj_name + "_state.npy")
            buffer.action_buf = np.load(traj_name+ "_action.npy")
            buffer.next_state_buf = np.load(traj_name + "_next_state.npy")
            buffer.next_action_buf = np.load(traj_name + "_next_action.npy")
            buffer.task_id_buf = np.load(traj_name + "_task_id.npy")
            buffer.next_task_id_buf = np.load(traj_name + "_next_task_id.npy")
            buffer.done_buf = np.load(traj_name + "_done.npy")
            buffer.size = len(self.memory.done_buf)
            self.memory_total_buff.append(buffer)

        # Create Networks
        self.policy_net = Pretraining(obs=self.observation_shape,
                                      n_actions=self.config.n_actions,
                                      cumulants=self.config.cumulants,
                                      tasks=self.config.num_tasks,
                                      hidden_size=self.config.hidden_size,
                                      mode=self.config.mode).to(self.device)

        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        if config.mode == "play-multitask-irl":
            self.policy_net.eval()
            self.target_net.eval()

        if self.config.load_model:
            policy_state_dict = torch.load("models/model_" + self.load_model_path)
            target_state_dict = torch.load("models/target_" + self.load_model_path)
            optimizer_state_dict = torch.load("models/optimizer_" + self.load_model_path)

            self.policy_net.load_state_dict(policy_state_dict)
            self.target_net.load_state_dict(target_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.total_steps = 0

    def select_action(self, state, random_color):
        with torch.no_grad():
            q, rewards = self.policy_net(torch.tensor(state).to(self.device), random_color)
            action = q.max(1)[1]

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/action_selected": action
        }
        wandb.log(log_dict)
        return action

    def optimize_model_phase_II(self, data):
        state, action, \
        next_state, next_action, \
        task_id, next_task_id, done = data['state'], data['action'], \
                                      data['next_state'], data['next_action'],\
                                      data['task_id'], data['next_task_id'], data['done']

        action = action.to(torch.int64)
        next_action = next_action.to(torch.int64)
        task_id = task_id.long()
        next_task_id = next_task_id.long()
        done = done.unsqueeze(-1)

        assert state.shape == (self.config.batch_size, self.observation_shape[0], self.observation_shape[1])
        assert action.shape == (self.config.batch_size, 1)
        assert next_state.shape == (self.config.batch_size, self.observation_shape[0], self.observation_shape[1])
        assert next_action.shape == (self.config.batch_size, 1)
        assert done.shape == (self.config.batch_size, 1)

        self.optimizer.zero_grad()
        q, rewards = self.policy_net(state, task_id)
        state_action_values = torch.gather(q, 1, action)
        assert state_action_values.shape == (self.config.batch_size, 1)

        phi_action = torch.gather(rewards, 1, action)
        target_q, _ = self.target_net(next_state, task_id)
        next_psi_values = torch.gather(target_q, 1, next_action)
        expected_psi_values = torch.logical_not(done) * (next_psi_values * self.config.gamma) + phi_action.detach()
        assert expected_psi_values.shape == (self.config.batch_size, 1)
        itd_loss = F.smooth_l1_loss(state_action_values, expected_psi_values)

        assert q.shape == (self.config.batch_size, self.config.n_actions)
        bc_loss = self.nll_loss_fn(action, q)

        loss = bc_loss + itd_loss
        loss.backward()

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/bc_loss": bc_loss,
            "step/itd_loss": itd_loss,

        }
        wandb.log(log_dict)

        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def nll_loss_fn(self, action, q_next):
        softmax = nn.Softmax(dim=-1)
        q_action = softmax(q_next)
        action = to_onehot(action.squeeze(-1), 5)
        assert q_action.shape == action.shape
        loss = nn.CrossEntropyLoss()
        return loss(q_action, action)

    def inverse_train(self):
        for step in range(self.config.n_episodes):
            task_id = random.randrange(self.config.num_tasks)
            self.total_steps = step
            data = self.memory_total_buff[task_id].sample_batch()
            self.optimize_model_phase_II(data)

            print('Total steps: {}'.format(self.total_steps))

            if step % self.config.save_model_episode == 0:
                self.evaluate(task_id)

            if self.total_steps % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self, task_id):
        all_envs = self.env

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
            state = next_state
            action = next_action

            if done:
                break

        if episode % self.config.record_episode == 0:
            log_dict = {
                "episode/x_axis": episode,
                "episode/episodic_reward_episode": total_reward,
                "episode/length": t,
                "step/x_axis": self.total_steps,
                "step/episodic_reward_steps": total_reward
            }
            wandb.log(log_dict)

            if self.config.domain == "highway" or self.config.domain == "roundabout" :
                driving_dict =  {"step/reward_min": true_reward[1],
                                "step/reward_max": true_reward[2],
                                "step/collision_reward": true_reward[3],
                                "step/lane_pref_reward": true_reward[4],
                                "step/speed_reward": true_reward[5],
                                "step/distance_reward": true_reward[6]}

                wandb.log(driving_dict)

            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.total_steps, self.total_steps, t,
                                                                                 total_reward))
            if episode % self.config.save_model_episode == 0:
                torch.save(self.policy_net.state_dict(), "models/model_" + self.config.model_name + ".pth")
                torch.save(self.target_net.state_dict(), "models/target_" + self.config.model_name + ".pth")
                torch.save(self.optimizer.state_dict(), "models/optimizer_" +  self.config.model_name + ".pth")

                wandb.save("models/model_" + self.config.model_name + ".pth")
                wandb.save("models/target_" + self.config.model_name + ".pth")
                wandb.save("models/optimizer_" + self.config.model_name + ".pth")

        env.close()
        return
