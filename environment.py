import numpy as np
import random
from datetime import timedelta

class CrowdEnv:
    def __init__(self, project_info, worker_quality, entry_info):
        self.project_info = project_info
        self.entry_info = entry_info
        self.worker_quality = worker_quality
        self.project_ids = list(project_info.keys())
        self.reset()

    def reset(self):
        self.current_worker = random.choice(list(self.worker_quality.keys()))
        self.available_projects = random.sample(self.project_ids, k=min(5, len(self.project_ids)))
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # 示例状态：当前worker质量 + 所有可选项目简单特征
        state = [self.worker_quality[self.current_worker]]
        for pid in self.available_projects:
            p = self.project_info[pid]
            state += [p["category"], p["sub_category"], p["industry"]]
        return np.array(state, dtype=np.float32)

    def step(self, action_index):
        chosen_project = self.available_projects[action_index]
        worker_q = self.worker_quality[self.current_worker]
        p = self.project_info[chosen_project]

        reward = worker_q * 10 - 0.1 * p["entry_count"]
        done = True  # 每轮只推荐一个任务
        return self._get_state(), reward, done, {}
