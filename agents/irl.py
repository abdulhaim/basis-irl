import math
import random
import wandb
import torch
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from misc.memory import ReplayMemory
from misc.utils import process_state
from itertools import count

from misc.utils import to_onehot
import matplotlib.pyplot as plt
from PIL import Image

class IRLAgent():
    def __init__(self, config, env, device):
        super(IRLAgent, self).__init__()
        self.config = config
        self.env = env
        self.device = device

        # Create Networks
        if len(self.env[0].observation_shape) == 2:
            from networks.sequential_discrete import IRL
            self.observation_shape = self.env[0].observation_shape
        else:
            from networks.pixel_discrete import IRL
            self.observation_shape = (self.env[0].observation_shape[2], self.env[0].observation_shape[0], self.env[0].observation_shape[1])

        traj_name = "traj/play-expert_domain_" + self.config.domain + "_expert_demons_" + str(self.config.demonstrations) + "_seed_" + str(self.config.seed)
        size = np.load(traj_name + "_action.npy").shape
        self.memory = ReplayMemory(self.observation_shape, self.config.n_actions, size=size[0])
        self.memory.state_buf = np.load(traj_name + "_state.npy")
        self.memory.action_buf = np.load(traj_name+ "_action.npy")
        self.memory.next_state_buf = np.load(traj_name + "_next_state.npy")
        self.memory.next_action_buf = np.load(traj_name + "_next_action.npy")
        self.memory.done_buf = np.load(traj_name + "_done.npy")
        self.memory.size = len(self.memory.done_buf)

        # Create Networks
        self.policy_net = IRL(obs=self.observation_shape,
                              n_actions=self.config.n_actions,
                              cumulants=self.config.cumulants,
                              tasks=self.config.num_tasks,
                              hidden_size=self.config.hidden_size,
                              mode=self.config.mode).to(self.device)


        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        self.load_models()
        self.total_steps = 0

    def load_models(self):
        if len(self.env[0].observation_shape) == 2:
            from networks.sequential_discrete import Pretraining
        else:
            from networks.pixel_discrete import Pretraining

        policy_state_dict = torch.load("models/model_" + self.config.load_model_path + ".pth")
        target_state_dict = torch.load("models/target_" + self.config.load_model_path + ".pth")
        optimizer_state_dict = torch.load("models/optimizer_" + self.config.load_model_path + ".pth")

        # Load Optimizer
        self.optimizer.load_state_dict(optimizer_state_dict)
        old_policy_net = Pretraining(obs=self.env[0].observation_shape,
                                     n_actions=self.config.n_actions,
                                     cumulants=self.config.cumulants,
                                     tasks=self.config.pretraining_tasks,
                                     hidden_size=self.config.hidden_size,
                                     mode=self.config.mode).to(self.device)

        old_policy_net.load_state_dict(policy_state_dict)

        old_target_net = Pretraining(obs=self.env[0].observation_shape,
                                     n_actions=self.config.n_actions,
                                     cumulants=self.config.cumulants,
                                     tasks=self.config.pretraining_tasks, 
                                     hidden_size=self.config.hidden_size,
                                     mode=self.config.mode).to(self.device)

        old_target_net.load_state_dict(target_state_dict)

        # Policy: Load Psi, Phi, and average of w
        self.policy_net.features = old_policy_net.features
        self.policy_net.psi = old_policy_net.psi
        self.policy_net.phi = old_policy_net.phi
        self.policy_net.w = nn.Parameter(torch.mean(old_policy_net.w, axis=0))

        # Target: Load Psi, Phi, and average of w
        self.target_net.features = old_target_net.features
        self.target_net.psi = old_target_net.psi
        self.target_net.phi = old_target_net.phi
        self.target_net.w = nn.Parameter(torch.mean(old_target_net.w, axis=0))

        # Load Expert Model
        self.expert_policy = Pretraining(obs=self.env[0].observation_shape,
                                    n_actions=self.config.n_actions,
                                    cumulants=self.config.cumulants,
                                    tasks=self.config.env_num_tasks,
                                    hidden_size=self.config.hidden_size,
                                    mode=self.config.mode).to(self.device)

        
        expert_state_dict = torch.load("models/model_" + self.config.expert_model_path + ".pth")
        self.expert_policy.load_state_dict(expert_state_dict)


    def select_action(self, state, task_id):
        with torch.no_grad():
            q, rewards = self.policy_net(torch.tensor(state).to(self.device), self.config.pretraining_tasks*[task_id])
            action = q.max(1)[1]

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/action_selected": action
        }
        wandb.log(log_dict)
        return action, rewards

    def get_expert_action(self, state, task_id):
        with torch.no_grad():
            if len(state.shape) == 2:
                task_id = torch.tensor(task_id).unsqueeze(0)
                state = torch.tensor(state).unsqueeze(0)

            features = self.expert_policy.features(state)
            features = features.reshape(state.shape[0], -1)

            task_id_encoding = to_onehot(task_id, self.config.env_num_tasks)
            features = torch.cat([features, task_id_encoding.to(self.device)], dim=-1)

            output_psi = self.expert_policy.psi(features)
            output_psi = output_psi.view(state.shape[0], self.expert_policy.num_cumulants, self.expert_policy.n_actions)
            q = torch.einsum("bca, tc  -> bta", output_psi, self.expert_policy.w)
            q = q[torch.arange(state.shape[0]), task_id.squeeze(), :]

            action = q.max(1)[1]
            return action

    def optimize_model(self, data):
        state, action, next_state, next_action, done = \
            data['state'], data['action'], data['next_state'], data['next_action'], data['done']

        action = action.to(torch.int64)
        next_action = next_action.to(torch.int64)
        task_index = torch.cat(self.config.batch_size * [torch.tensor(self.config.pretraining_tasks*[0]).unsqueeze(0)])
        done = done.unsqueeze(-1)

        assert state.shape == (self.config.batch_size, *self.observation_shape)
        assert action.shape == (self.config.batch_size, 1)
        assert next_state.shape == (self.config.batch_size, *self.observation_shape)
        assert next_action.shape == (self.config.batch_size, 1)
        assert done.shape == (self.config.batch_size, 1)
        assert task_index.shape == (self.config.batch_size, self.config.pretraining_tasks)

        self.optimizer.zero_grad()
        q, rewards = self.policy_net(state, task_index)
        state_action_values = torch.gather(q, 1, action)
        assert state_action_values.shape == (self.config.batch_size, 1)

        phi_action = torch.gather(rewards, 1, action)
        target_q, _ = self.target_net(next_state, task_index)
        next_psi_values = torch.gather(target_q, 1, next_action)
        expected_psi_values = torch.logical_not(done) * (next_psi_values * self.config.gamma) + phi_action.detach()
        assert expected_psi_values.shape == (self.config.batch_size, 1)

        itd_loss = F.smooth_l1_loss(state_action_values, expected_psi_values)

        assert action.shape == (self.config.batch_size, 1)
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
        action = to_onehot(action.squeeze(-1), self.config.n_actions)
        assert q_action.shape == action.shape
        loss = nn.CrossEntropyLoss()
        return loss(q_action, action)

    def inverse_train(self):
        for step in range(self.config.irl_steps):
            self.total_steps = step
            data = self.memory.sample_batch(self.config.batch_size)
            # data = self.memory.sample_batch_index(step)
            self.optimize_model(data)
            print('Total steps: {}'.format(self.total_steps))
            if step % self.config.save_model_episode == 0:
                torch.save(self.policy_net.state_dict(), "models/" + self.config.model_name + ".pth")
                wandb.save("models/" + self.config.model_name + ".pth")

    def env_setup(self, env):
        if self.config.domain == "highway" or self.config.domain == "roundabout":
            print("Expert to Infer")
            print(env.config["lane_preference"])
            print(env.config["target_speed"])
            print(env.config["desired_distance"])

            self.lane_preference_dict = {'right': 0, 'center': 0, 'left': 0}

    def env_update(self, env, reward):
        if self.config.domain == "highway" or self.config.domain == "roundabout":
            if reward[4] == 0.5:
                self.lane_preference_dict['center'] += 1
            elif reward[4] == 1:
                self.lane_preference_dict[env.config["lane_preference"]] += 1
            else:
                if env.config["lane_preference"] == "right":
                    key = "left"
                else:
                    key = "right"
                self.lane_preference_dict[key] += 1

    def env_compute_reward_loss(self, predicted_reward, expert_reward, actions):
            total_predicted_reward = torch.stack(predicted_reward)
            total_predicted_reward = torch.gather(total_predicted_reward, 1, torch.tensor(actions).unsqueeze(-1))
            reward_loss = F.smooth_l1_loss(torch.tensor(expert_reward).unsqueeze(-1), total_predicted_reward)
            return reward_loss


    def test_agent(self, env):
        self.total_steps = 0
        self.policy_net.eval()
        self.target_net.eval()

        env = env[0]
        self.env_setup(env)
        for episode in range(self.config.irl_test_episodes):
            state = env.reset()
            state = process_state(state, self.observation_shape)
            task_id = 0 

            action, predicted_reward = self.select_action(state, task_id)
            expert_action = self.get_expert_action(state, task_id)
            total_reward = 0.0

            action_list = []
            total_predicted_reward = []
            total_expert_reward = []
            for t in count():
                self.total_steps += 1
                expert_reward = env.get_expert_reward(expert_action.item())
                next_state, true_reward, done, info = env.step(action.item())
                next_state = process_state(next_state, self.observation_shape)

                self.env_update(env, true_reward)


                if len(self.observation_shape) == 2:
                    reward = true_reward[0]
                else:
                    reward = true_reward

                total_reward += reward

                # For Reward MSE Loss
                action_list.append(action.item())
                total_predicted_reward.append(predicted_reward.squeeze(0))
                total_expert_reward.append(expert_reward)

                next_action, next_predicted_reward = self.select_action(next_state, task_id)
                action = next_action
                predicted_reward = next_predicted_reward

                if done:
                    break

            reward_loss = self.env_compute_reward_loss(total_predicted_reward, total_expert_reward, action_list)

            if episode % self.config.record_episode == 0:
                log_dict = {
                    "episode/x_axis": episode,
                    "episode/episodic_reward_episode": total_reward,
                    "episode/length": t,
                    "step/x_axis": self.total_steps,
                    "step/reward_loss": reward_loss,
                    "step/episodic_reward_steps": total_reward,
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
                    a = {k: v / self.total_steps for k, v in self.lane_preference_dict.items()}
                    print("Lane Preferences", a)


                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.total_steps, episode, t, total_reward))
            if episode % self.config.save_model_episode == 0:
                torch.save(self.policy_net.state_dict(), "models/phaseII_model_" + self.config.model_name + "_demons_" + str(self.config.demonstrations) + ".pth")
                torch.save(self.optimizer.state_dict(), "models/phaseII_optimizer_" + self.config.model_name + "_demons_" + str(self.config.demonstrations)  + ".pth")
                torch.save(self.target_net.state_dict(), "models/phaseII_target_model_" + self.config.model_name + "_demons_" + str(self.config.demonstrations)  + ".pth")

                wandb.save("models/phaseII_model_" + self.config.model_name + ".pth")
                wandb.save("models/phaseII_optimizer_" + self.config.model_name + ".pth")
                wandb.save("models/phaseII_target_model_" + self.config.model_name + ".pth")

        env.close()
        return
