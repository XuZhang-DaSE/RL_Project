from data_loader import load_worker_quality, load_project_info
from environment import CrowdEnv
from agent import Agent
import os

def main():
    project_path = "data/project/"
    entry_path = "data/entry/"
    project_list_path = "data/project_list.csv"
    worker_quality_path = "data/worker_quality.csv"

    project_info, entry_info = load_project_info(project_list_path, project_path, entry_path)
    worker_quality = load_worker_quality(worker_quality_path)

    env = CrowdEnv(project_info, worker_quality, entry_info)
    agent = Agent(state_dim=1 + 3 * 5, action_dim=5)

    for episode in range(500):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.update()
        agent.soft_update()
        if agent.eps > agent.eps_min:
            agent.eps *= agent.eps_decay
        if episode % 50 == 0:
            print(f"Episode {episode}, reward={reward:.2f}, eps={agent.eps:.2f}")

if __name__ == "__main__":
    main()
