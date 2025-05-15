import torch
import numpy as np
from data_loader import load_worker_quality, load_project_info
from environment import CrowdEnv
from model import DuelingDQN

def test_agent(model_path, num_episodes=20):
    # 1. 加载数据
    project_info, entry_info = load_project_info(
        "data/project_list.csv", "data/project/", "data/entry/")
    worker_quality = load_worker_quality("data/worker_quality.csv")

    # 2. 构建环境
    env = CrowdEnv(project_info, worker_quality, entry_info)
    state_dim = env._get_state().shape[0]
    action_dim = len(env.available_projects)

    # 3. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingDQN(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. 运行若干轮并统计平均奖励
    total_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()

        next_state, reward, done, _ = env.step(action)
        total_rewards.append(reward)
        print(f"Episode {ep+1}: reward = {reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test trained Dueling DQN Agent on CrowdEnv")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the saved checkpoint (e.g., checkpoints/ddqn_per.pth)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of test episodes")
    args = parser.parse_args()

    test_agent(args.model_path, args.episodes)
