import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayMemory(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.next_state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.next_action_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.task_id_buf = np.zeros(size, dtype=np.float32)
        self.next_task_id_index_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, next_state, reward, next_action, task_id, next_task_id, done):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.next_state_buf[self.ptr] = next_state
        self.reward_buf[self.ptr] = reward
        self.next_action_buf[self.ptr] = next_action
        self.task_id_buf[self.ptr] = task_id
        self.next_task_id_index_buf[self.ptr] = next_task_id
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            next_state=self.next_state_buf[idxs],
            reward=self.reward_buf[idxs],
            next_action=self.next_action_buf[idxs],
            task_id=self.task_id_buf[idxs],
            next_task_id=self.next_task_id_index_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def sample_batch_index(self, index):
        idxs = [index]
        batch = dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            next_state=self.next_state_buf[idxs],
            reward=self.reward_buf[idxs],
            next_action=self.next_action_buf[idxs],
            task_id=self.task_id_buf[idxs],
            next_task_id=self.next_task_id_index_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

