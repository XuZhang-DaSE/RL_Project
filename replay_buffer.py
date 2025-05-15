import numpy as np
import random
from collections import namedtuple

# -------------- SumTree for Prioritized Replay --------------
class SumTree:
    """
    This SumTree data structure stores priorities and allows sampling
    based on the priority distribution.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

# -------------- PER Buffer Using SumTree --------------
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Stores transitions and samples them according to prioritized weights.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # how much prioritization is used (0 = uniform, 1 = full)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state', 'done'))

    def _get_beta(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, state, action, reward, next_state, done):
        """Add new experience with maximum priority so that it is sampled at least once."""
        transition = self.Transition(state, action, reward, next_state, done)
        p = self.max_priority ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size):
        """Sample a batch of experiences and compute their importance-sampling weights."""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # compute IS weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        beta = self._get_beta()
        self.frame += 1
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        weights /= weights.max()

        # stack into arrays
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])

        return (states, actions, rewards, next_states, dones), idxs, weights

    def update_priorities(self, idxs, td_errors, epsilon=1e-6):
        """Update priorities of sampled transitions based on new TD errors."""
        for idx, td_error in zip(idxs, td_errors):
            p = (abs(td_error) + epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries
