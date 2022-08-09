import torch.nn as nn
import torch
from misc.utils import to_onehot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pretraining(nn.Module):
    def __init__(self, obs, n_actions, cumulants, tasks, hidden_size, mode):
        """
        Initialize Pretraining Network
        """
        super(Pretraining, self).__init__()
        self.obs_shape = obs[0]
        self.n_actions = n_actions
        self.num_cumulants = cumulants
        self.num_tasks = tasks
        self.hidden_size = hidden_size
        self.mode = mode
        self.features = nn.Sequential(
            nn.Linear(self.obs_shape, hidden_size),
            nn.ReLU())

        self.psi = nn.Sequential(
            nn.Linear(self.obs_shape*self.hidden_size+ self.num_tasks, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_cumulants * self.n_actions),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.obs_shape*self.hidden_size + self.num_tasks, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_cumulants * self.n_actions),
        )

        self.w = nn.Parameter(torch.randn(self.num_tasks, self.num_cumulants))

    def forward(self, x, task_id):
        if len(x.shape) == 2:
            task_id = torch.tensor(task_id).unsqueeze(-1)
            x = x.unsqueeze(0)

        features = self.features(x)
        features = features.reshape(x.shape[0], -1)

        task_id_encoding = to_onehot(task_id, self.num_tasks)
        features = torch.cat([features, task_id_encoding.to(device)], dim=-1)

        output_psi = self.psi(features)
        output_psi = output_psi.view(x.shape[0], self.num_cumulants, self.n_actions)
        q = torch.einsum("bca, tc  -> bta", output_psi, self.w)
        q = q[torch.arange(x.shape[0]), task_id.squeeze(), :]

        output_phi = self.phi(features)
        output_phi = output_phi.view(x.shape[0], self.num_cumulants, self.n_actions)
        rewards = torch.einsum("bca, tc  -> bta", output_phi, self.w)
        rewards = rewards[torch.arange(x.shape[0]), task_id.squeeze(), :]

        return q, rewards

class IRL(nn.Module):
    def __init__(self, obs, n_actions, cumulants, tasks, hidden_size, mode):
        """
        Initialize IRL Network
        """
        super(IRL, self).__init__()
        self.obs_shape = obs[0]
        self.n_actions = n_actions
        self.num_cumulants = cumulants
        self.num_tasks = tasks
        self.hidden_size = hidden_size
        self.mode = mode

        self.features = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.ReLU())

        self.psi = nn.Sequential(
            nn.Linear(self.obs_shape*self.hidden_size + self.num_tasks, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_cumulants * self.n_actions),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.obs_shape*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_cumulants * self.n_actions),
        )

        self.w = nn.Parameter(torch.randn(self.num_cumulants))

    def forward(self, x, task_encoding):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            task_encoding = torch.tensor(task_encoding).unsqueeze(0)

        features = self.features(x)
        features = features.reshape(x.shape[0], -1)

        features = torch.cat([features, task_encoding.to(device)], dim=-1)
        output_psi = self.psi(features)
        output_psi = output_psi.view(x.shape[0], self.num_cumulants, self.n_actions)
        q = torch.einsum("bca, c  -> ba", output_psi, self.w)

        output_phi = self.phi(features)
        output_phi = output_phi.view(x.shape[0], self.num_cumulants, self.n_actions)
        rewards = torch.einsum("bca, c  -> ba", output_phi, self.w)

        return q, rewards


