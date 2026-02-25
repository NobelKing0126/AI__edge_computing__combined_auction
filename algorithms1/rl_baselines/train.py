"""
DDPG/TD3 训练脚本

用于训练和评估强化学习基线算法
"""

import numpy as np
import torch
import copy
import time
import os
import sys
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.rl_baselines.ddpg import DDPGAgent, DDPGConfig, DDPGBaseline
from algorithms.rl_baselines.td3 import TD3Agent, TD3Config, TD3Baseline
from experiments.baselines import BaselineRunner
from config.system_config import SystemConfig


class TaskEnvironment:
    """
    任务调度环境
    
    用于RL训练的环境封装
    """
    
    def __init__(self, n_users: int = 50, n_uavs: int = 5, seed: int = 42):
        """
        初始化环境
        
        Args:
            n_users: 用户数量
            n_uavs: UAV数量
            seed: 随机种子
        """
        self.n_users = n_users
        self.n_uavs = n_uavs
        self.seed = seed
        self.config = SystemConfig()
        
        self.reset()
    
    def reset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        重置环境
        
        Returns:
            tasks: 新的任务列表
            uav_resources: 重置的UAV资源
        """
        np.random.seed(self.seed + np.random.randint(10000))
        
        # 生成任务
        self.tasks = []
        for i in range(self.n_users):
            self.tasks.append({
                'task_id': i,
                'user_id': i,
                'user_pos': (np.random.uniform(0, 2000), np.random.uniform(0, 2000)),
                'data_size': np.random.uniform(1e6, 5e6),
                'compute_size': np.random.uniform(5e9, 20e9),
                'deadline': np.random.uniform(2.0, 5.0),
                'priority': np.random.uniform(0.3, 0.9),
                'user_level': np.random.randint(1, 6)
            })
        
        # 初始化UAV资源
        self.uav_resources = []
        for i in range(self.n_uavs):
            self.uav_resources.append({
                'uav_id': i,
                'position': (400 + i * 300, 1000),
                'f_max': self.config.uav.f_max,
                'E_max': self.config.uav.E_max,
                'E_remain': self.config.uav.E_max
            })
        
        self.current_task_idx = 0
        
        return self.tasks, self.uav_resources
    
    def get_current_task(self) -> Optional[Dict]:
        """获取当前任务"""
        if self.current_task_idx < len(self.tasks):
            return self.tasks[self.current_task_idx]
        return None
    
    def step(self, result: Dict) -> bool:
        """
        执行一步
        
        Args:
            result: 执行结果
            
        Returns:
            done: 是否结束
        """
        # 更新UAV资源
        if result.get('success') and 'uav_id' in result:
            uav_id = result['uav_id']
            self.uav_resources[uav_id]['E_remain'] -= result.get('energy', 0)
        
        self.current_task_idx += 1
        
        return self.current_task_idx >= len(self.tasks)
    
    def get_uav_resources(self) -> List[Dict]:
        """获取当前UAV资源"""
        return self.uav_resources


def train_ddpg(n_episodes: int = 200, 
               n_users: int = 50, 
               n_uavs: int = 5,
               save_path: str = "models/ddpg_model.pt",
               verbose: bool = True) -> DDPGAgent:
    """
    训练DDPG Agent
    
    Args:
        n_episodes: 训练回合数
        n_users: 每回合用户数
        n_uavs: UAV数量
        save_path: 模型保存路径
        verbose: 是否打印训练信息
        
    Returns:
        DDPGAgent: 训练好的Agent
    """
    config = DDPGConfig(n_uavs=n_uavs)
    agent = DDPGAgent(config)
    env = TaskEnvironment(n_users=n_users, n_uavs=n_uavs)
    
    best_reward = -float('inf')
    episode_rewards = []
    episode_success_rates = []
    
    if verbose:
        print("=" * 60)
        print("训练DDPG")
        print("=" * 60)
        print(f"  回合数: {n_episodes}")
        print(f"  每回合任务数: {n_users}")
        print(f"  UAV数量: {n_uavs}")
        print(f"  设备: {agent.device}")
        print()
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        tasks, uav_resources = env.reset()
        agent.noise.reset()
        
        episode_reward = 0
        success_count = 0
        
        for task in tasks:
            # 获取状态
            state = agent.get_state(task, env.get_uav_resources())
            
            # 选择动作
            action = agent.select_action(state, add_noise=True)
            
            # 解码动作
            uav_id, split_ratio = agent.decode_action(action)
            uav_id = min(uav_id, n_uavs - 1)
            
            # 执行动作
            result = agent.execute_action(task, uav_id, split_ratio, 
                                          env.get_uav_resources())
            
            # 计算奖励
            reward = agent.compute_reward(task, result)
            
            # 获取下一状态
            done = env.step(result)
            next_task = env.get_current_task()
            
            if next_task is not None:
                next_state = agent.get_state(next_task, env.get_uav_resources())
            else:
                next_state = state  # 使用当前状态作为终止状态
            
            # 存入缓冲区
            agent.buffer.add(state, action, reward, next_state, done)
            
            # 更新网络
            if len(agent.buffer) >= config.batch_size:
                agent.update()
            
            episode_reward += reward
            if result['success']:
                success_count += 1
        
        # 衰减噪声
        agent.decay_noise()
        
        episode_rewards.append(episode_reward)
        success_rate = success_count / n_users
        episode_success_rates.append(success_rate)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
        
        if verbose and (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_success = np.mean(episode_success_rates[-20:])
            print(f"  Episode {episode+1:4d}: Reward={avg_reward:.2f}, "
                  f"Success={avg_success*100:.1f}%, Noise={agent.noise_scale:.3f}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print(f"训练完成! 耗时: {elapsed:.1f}s")
        print(f"最终成功率: {np.mean(episode_success_rates[-20:])*100:.1f}%")
        print(f"模型已保存: {save_path}")
    
    # 加载最佳模型
    agent.load(save_path)
    
    return agent


def train_td3(n_episodes: int = 200,
              n_users: int = 50,
              n_uavs: int = 5,
              save_path: str = "models/td3_model.pt",
              verbose: bool = True) -> TD3Agent:
    """
    训练TD3 Agent
    
    Args:
        n_episodes: 训练回合数
        n_users: 每回合用户数
        n_uavs: UAV数量
        save_path: 模型保存路径
        verbose: 是否打印训练信息
        
    Returns:
        TD3Agent: 训练好的Agent
    """
    config = TD3Config(n_uavs=n_uavs)
    agent = TD3Agent(config)
    env = TaskEnvironment(n_users=n_users, n_uavs=n_uavs)
    
    best_reward = -float('inf')
    episode_rewards = []
    episode_success_rates = []
    
    if verbose:
        print("=" * 60)
        print("训练TD3")
        print("=" * 60)
        print(f"  回合数: {n_episodes}")
        print(f"  每回合任务数: {n_users}")
        print(f"  UAV数量: {n_uavs}")
        print(f"  策略延迟: {config.policy_delay}")
        print(f"  设备: {agent.device}")
        print()
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        tasks, uav_resources = env.reset()
        agent.noise.reset()
        
        episode_reward = 0
        success_count = 0
        
        for task in tasks:
            state = agent.get_state(task, env.get_uav_resources())
            action = agent.select_action(state, add_noise=True)
            uav_id, split_ratio = agent.decode_action(action)
            uav_id = min(uav_id, n_uavs - 1)
            
            result = agent.execute_action(task, uav_id, split_ratio,
                                          env.get_uav_resources())
            reward = agent.compute_reward(task, result)
            
            done = env.step(result)
            next_task = env.get_current_task()
            
            if next_task is not None:
                next_state = agent.get_state(next_task, env.get_uav_resources())
            else:
                next_state = state
            
            agent.buffer.add(state, action, reward, next_state, done)
            
            if len(agent.buffer) >= config.batch_size:
                agent.update()
            
            episode_reward += reward
            if result['success']:
                success_count += 1
        
        agent.decay_noise()
        
        episode_rewards.append(episode_reward)
        success_rate = success_count / n_users
        episode_success_rates.append(success_rate)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
        
        if verbose and (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_success = np.mean(episode_success_rates[-20:])
            print(f"  Episode {episode+1:4d}: Reward={avg_reward:.2f}, "
                  f"Success={avg_success*100:.1f}%, Noise={agent.noise_scale:.3f}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print(f"训练完成! 耗时: {elapsed:.1f}s")
        print(f"最终成功率: {np.mean(episode_success_rates[-20:])*100:.1f}%")
        print(f"模型已保存: {save_path}")
    
    agent.load(save_path)
    
    return agent


def evaluate_rl_baselines(ddpg_agent: DDPGAgent = None,
                          td3_agent: TD3Agent = None,
                          n_users: int = 50,
                          n_trials: int = 5) -> Dict:
    """
    评估RL基线算法
    
    Args:
        ddpg_agent: DDPG Agent
        td3_agent: TD3 Agent
        n_users: 用户数量
        n_trials: 评估次数
        
    Returns:
        Dict: 评估结果
    """
    print("=" * 70)
    print("评估RL基线算法")
    print("=" * 70)
    
    results = {}
    
    # 生成测试数据
    np.random.seed(42)
    test_tasks = []
    for i in range(n_users):
        test_tasks.append({
            'task_id': i,
            'user_id': i,
            'user_pos': (np.random.uniform(0, 2000), np.random.uniform(0, 2000)),
            'data_size': np.random.uniform(1e6, 5e6),
            'compute_size': np.random.uniform(5e9, 20e9),
            'deadline': np.random.uniform(2.0, 5.0),
            'priority': np.random.uniform(0.3, 0.9)
        })
    
    uav_resources = [
        {'uav_id': i, 'f_max': 15e9, 'E_max': 500e3}
        for i in range(5)
    ]
    
    cloud_resources = {'f_cloud': 500e9}
    
    # 评估DDPG
    if ddpg_agent is not None:
        ddpg_baseline = DDPGBaseline(agent=ddpg_agent)
        ddpg_results = []
        
        for _ in range(n_trials):
            result = ddpg_baseline.run(test_tasks, uav_resources, cloud_resources)
            ddpg_results.append(result)
        
        avg_success = np.mean([r.success_rate for r in ddpg_results])
        avg_delay = np.mean([r.avg_delay for r in ddpg_results])
        avg_energy = np.mean([r.total_energy for r in ddpg_results])
        
        results['DDPG'] = {
            'success_rate': avg_success,
            'avg_delay': avg_delay,
            'total_energy': avg_energy
        }
        
        print(f"\nDDPG:")
        print(f"  成功率: {avg_success*100:.1f}%")
        print(f"  平均时延: {avg_delay*1000:.1f}ms")
        print(f"  总能耗: {avg_energy:.1f}J")
    
    # 评估TD3
    if td3_agent is not None:
        td3_baseline = TD3Baseline(agent=td3_agent)
        td3_results = []
        
        for _ in range(n_trials):
            result = td3_baseline.run(test_tasks, uav_resources, cloud_resources)
            td3_results.append(result)
        
        avg_success = np.mean([r.success_rate for r in td3_results])
        avg_delay = np.mean([r.avg_delay for r in td3_results])
        avg_energy = np.mean([r.total_energy for r in td3_results])
        
        results['TD3'] = {
            'success_rate': avg_success,
            'avg_delay': avg_delay,
            'total_energy': avg_energy
        }
        
        print(f"\nTD3:")
        print(f"  成功率: {avg_success*100:.1f}%")
        print(f"  平均时延: {avg_delay*1000:.1f}ms")
        print(f"  总能耗: {avg_energy:.1f}J")
    
    # 与其他基线对比
    print("\n" + "=" * 70)
    print("与传统基线对比")
    print("=" * 70)
    
    baseline_runner = BaselineRunner()
    baseline_results = baseline_runner.run_all(test_tasks, uav_resources, cloud_resources)
    
    print(f"\n{'算法':<20} {'成功率':>10} {'平均时延':>15} {'能耗':>12}")
    print("-" * 60)
    
    if 'DDPG' in results:
        r = results['DDPG']
        print(f"{'DDPG':<20} {r['success_rate']*100:>9.1f}% {r['avg_delay']*1000:>13.1f}ms {r['total_energy']:>10.1f}J")
    
    if 'TD3' in results:
        r = results['TD3']
        print(f"{'TD3':<20} {r['success_rate']*100:>9.1f}% {r['avg_delay']*1000:>13.1f}ms {r['total_energy']:>10.1f}J")
    
    for name, result in baseline_results.items():
        if name in ['Edge-Only', 'Cloud-Only', 'Greedy', 'Random-Auction']:
            print(f"{name:<20} {result.success_rate*100:>9.1f}% "
                  f"{result.avg_delay*1000:>13.1f}ms {result.total_energy:>10.1f}J")
    
    return results


def main():
    """主函数"""
    print("=" * 70)
    print("DDPG/TD3 强化学习基线训练与评估")
    print("=" * 70)
    
    # 创建模型目录
    os.makedirs("models", exist_ok=True)
    
    # 训练DDPG
    print("\n[1/3] 训练DDPG...")
    ddpg_agent = train_ddpg(
        n_episodes=100,  # 减少回合数以加快演示
        n_users=50,
        n_uavs=5,
        save_path="models/ddpg_model.pt"
    )
    
    # 训练TD3
    print("\n[2/3] 训练TD3...")
    td3_agent = train_td3(
        n_episodes=100,
        n_users=50,
        n_uavs=5,
        save_path="models/td3_model.pt"
    )
    
    # 评估
    print("\n[3/3] 评估...")
    results = evaluate_rl_baselines(ddpg_agent, td3_agent)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
