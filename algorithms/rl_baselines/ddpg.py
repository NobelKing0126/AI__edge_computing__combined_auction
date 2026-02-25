"""
DDPG (Deep Deterministic Policy Gradient) 算法实现

用于UAV任务卸载调度问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .networks import ActorNetwork, CriticNetwork, OUNoise
from .replay_buffer import ReplayBuffer
from experiments.baselines import BaselineResult, BaselineAlgorithm
from config.system_config import SystemConfig


@dataclass
class DDPGConfig:
    """DDPG配置"""
    state_dim: int = 26
    action_dim: int = 6
    n_uavs: int = 5
    hidden_dim: int = 256
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01


class DDPGAgent:
    """
    DDPG Agent
    
    用于学习UAV任务卸载策略
    """
    
    def __init__(self, config: Optional[DDPGConfig] = None):
        """
        初始化DDPG Agent
        
        Args:
            config: DDPG配置
        """
        self.config = config or DDPGConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 计算状态维度: 6 (任务) + n_uavs * 4 (UAV状态)
        self.state_dim = 6 + self.config.n_uavs * 4
        self.action_dim = self.config.n_uavs + 1  # UAV选择 + 切分比例
        
        # 创建网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, 
                                   self.config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = CriticNetwork(self.state_dim, self.action_dim,
                                     self.config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                          lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.config.lr_critic)
        
        # 经验回放
        self.buffer = ReplayBuffer(self.config.buffer_size)
        
        # 探索噪声
        self.noise = OUNoise(self.action_dim, sigma=self.config.exploration_noise)
        self.noise_scale = 1.0
        
        # 系统配置
        self.sys_config = SystemConfig()
        
        # 训练统计
        self.total_steps = 0
        self.episode_rewards = []
    
    def get_state(self, task: Dict, uav_resources: List[Dict]) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            task: 任务信息
            uav_resources: UAV资源列表
            
        Returns:
            state: 归一化状态向量
        """
        # 任务特征 (6维)
        task_features = np.array([
            task['data_size'] / 5e6,           # 归一化数据量
            task['compute_size'] / 20e9,        # 归一化计算量
            task['deadline'] / 5.0,             # 归一化截止时间
            task['priority'],                   # 优先级
            task['user_pos'][0] / 2000,         # 归一化X坐标
            task['user_pos'][1] / 2000,         # 归一化Y坐标
        ])
        
        # UAV状态 (每UAV 4维)
        uav_features = []
        for uav in uav_resources:
            # 计算到用户的距离
            dist = np.sqrt(
                (task['user_pos'][0] - uav['position'][0])**2 +
                (task['user_pos'][1] - uav['position'][1])**2 +
                self.sys_config.uav.H**2
            )
            
            uav_features.extend([
                uav['E_remain'] / uav['E_max'],     # 剩余能量比例
                0.5,                                  # 负载比例 (简化)
                min(dist / 1000, 1.0),               # 归一化距离
                1.0 if uav['E_remain'] > 1000 else 0.0  # 是否可用
            ])
        
        state = np.concatenate([task_features, np.array(uav_features)])
        return state.astype(np.float32)
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态向量
            add_noise: 是否添加探索噪声
            
        Returns:
            action: 动作向量
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample() * self.noise_scale
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def decode_action(self, action: np.ndarray) -> Tuple[int, float]:
        """
        解码动作
        
        Args:
            action: 动作向量 [-1, 1]
            
        Returns:
            uav_id: 选择的UAV ID
            split_ratio: 切分比例 [0, 1]
        """
        # UAV选择: 取最大值对应的索引
        uav_logits = action[:-1]
        uav_id = int(np.argmax(uav_logits))
        
        # 切分比例: 映射到 [0.1, 0.9]
        split_ratio = (action[-1] + 1) / 2  # [-1,1] -> [0,1]
        split_ratio = 0.1 + split_ratio * 0.8  # [0,1] -> [0.1, 0.9]
        
        return uav_id, split_ratio
    
    def _get_feature_size_at_layer(self, split_layer: int, model_spec, n_layers: int) -> float:
        """获取精确的中间特征大小"""
        if split_layer == 0:
            return model_spec.input_size_bytes if hasattr(model_spec, 'input_size_bytes') else 150000
        elif split_layer >= n_layers:
            return 1000
        
        if model_spec and hasattr(model_spec, 'get_output_size_at_layer'):
            return model_spec.get_output_size_at_layer(split_layer - 1)
        
        if hasattr(model_spec, 'typical_feature_sizes') and model_spec.typical_feature_sizes:
            idx = int(split_layer / n_layers * len(model_spec.typical_feature_sizes))
            idx = min(idx, len(model_spec.typical_feature_sizes) - 1)
            return model_spec.typical_feature_sizes[idx]
        return 150000 * (1 - split_layer / n_layers) * 0.5
    
    def _get_cumulative_flops_at_layer(self, split_layer: int, model_spec, n_layers: int, C_total: float) -> float:
        """获取精确的累计计算量"""
        if split_layer <= 0:
            return 0.0
        if split_layer >= n_layers:
            return C_total
        
        if model_spec and hasattr(model_spec, 'get_flops_ratio_at_layer'):
            return C_total * model_spec.get_flops_ratio_at_layer(split_layer)
        
        return C_total * (split_layer / n_layers)
    
    def execute_action(self, task: Dict, uav_id: int, split_ratio: float,
                       uav_resources: List[Dict], n_concurrent: int = 10) -> Dict:
        """
        执行动作并返回结果
        
        时延模型（考虑云端资源竞争和传播延迟，使用精确的每层数据）
        
        Args:
            task: 任务信息
            uav_id: 选择的UAV
            split_ratio: 切分比例
            uav_resources: UAV资源
            n_concurrent: 并发任务数（用于云端资源竞争）
            
        Returns:
            result: 执行结果
        """
        uav = uav_resources[uav_id]
        
        # 检查UAV是否可用
        if uav['E_remain'] < 1000:
            return {
                'success': False,
                'delay': 999.0,
                'energy': 0.0,
                'met_deadline': False
            }
        
        # 计算距离
        dist = np.sqrt(
            (task['user_pos'][0] - uav['position'][0])**2 +
            (task['user_pos'][1] - uav['position'][1])**2 +
            self.sys_config.uav.H**2
        )
        
        # 信道增益和传输速率
        h = self.sys_config.channel.beta_0 / (dist ** 2)
        P_tx = self.sys_config.channel.P_tx_user
        noise = self.sys_config.channel.N_0 * self.sys_config.channel.W
        snr = P_tx * h / noise
        rate = self.sys_config.channel.W * np.log2(1 + snr)
        
        # 上传时延
        T_upload = task['data_size'] / rate
        
        # 获取模型信息
        model_spec = task.get('model_spec')
        n_layers = model_spec.layers if model_spec and hasattr(model_spec, 'layers') else 10
        split_layer = int(split_ratio * n_layers)
        
        # 使用精确的计算量分配
        C_total = task['compute_size']
        C_edge = self._get_cumulative_flops_at_layer(split_layer, model_spec, n_layers, C_total)
        C_cloud = C_total - C_edge
        
        T_edge = C_edge / uav['f_max'] if C_edge > 0 else 0
        
        # 使用精确的特征大小
        feature_size = self._get_feature_size_at_layer(split_layer, model_spec, n_layers)
        T_trans = feature_size / self.sys_config.channel.R_backhaul if split_layer < n_layers else 0
        
        # 云端计算时延（考虑资源竞争和单任务算力上限）
        max_concurrent = self.sys_config.cloud.max_concurrent_tasks
        effective_concurrent = max(1, min(n_concurrent, max_concurrent))
        f_cloud_per_task = min(
            self.sys_config.cloud.F_c / effective_concurrent,
            self.sys_config.cloud.F_per_task_max
        )
        T_cloud = C_cloud / f_cloud_per_task if C_cloud > 0 and split_layer < n_layers else 0
        
        # 传播延迟（往返）
        T_propagation = 2 * self.sys_config.cloud.T_propagation if split_layer < n_layers else 0
        
        T_total = T_upload + T_edge + T_trans + T_propagation + T_cloud
        
        # 能耗（边缘计算 + 通信）
        kappa = self.sys_config.energy.kappa_edge
        P_rx = self.sys_config.uav.P_rx
        P_tx_uav = self.sys_config.uav.P_tx
        E_compute = kappa * (uav['f_max'] ** 2) * C_edge
        E_comm = P_rx * T_upload + P_tx_uav * T_trans
        energy = E_compute + E_comm
        
        # 判断成功
        success = T_total <= task['deadline']
        
        return {
            'success': success,
            'delay': T_total,
            'energy': energy,
            'met_deadline': success,
            'uav_id': uav_id,
            'split_ratio': split_ratio
        }
    
    def compute_reward(self, task: Dict, result: Dict) -> float:
        """
        计算奖励
        
        Args:
            task: 任务信息
            result: 执行结果
            
        Returns:
            reward: 奖励值
        """
        # 基础奖励
        if result['success']:
            base_reward = 1.0
        else:
            base_reward = -1.0
        
        # 时延奖励
        delay = result['delay']
        deadline = task['deadline']
        time_reward = (deadline - delay) / deadline
        time_reward = np.clip(time_reward, -1, 1)
        
        # 能耗惩罚
        energy = result['energy']
        energy_penalty = -energy / 1000
        energy_penalty = np.clip(energy_penalty, -0.5, 0)
        
        # 高优先级奖励
        priority_bonus = task['priority'] * 0.5 if result['success'] else 0
        
        # 综合奖励
        reward = (0.4 * base_reward + 
                  0.3 * time_reward + 
                  0.2 * energy_penalty + 
                  0.1 * priority_bonus)
        
        return reward
    
    def update(self) -> Tuple[float, float]:
        """
        更新网络
        
        Returns:
            critic_loss: Critic损失
            actor_loss: Actor损失
        """
        if len(self.buffer) < self.config.batch_size:
            return 0.0, 0.0
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + 
                (1 - self.config.tau) * target_param.data
            )
    
    def decay_noise(self):
        """衰减探索噪声"""
        self.noise_scale = max(self.config.min_noise, 
                               self.noise_scale * self.config.noise_decay)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


class DDPGBaseline(BaselineAlgorithm):
    """
    DDPG基线算法
    
    使用训练好的DDPG策略进行任务调度
    """
    
    def __init__(self, agent: Optional[DDPGAgent] = None, 
                 model_path: Optional[str] = None):
        """
        初始化DDPG基线
        
        Args:
            agent: 预训练的DDPG Agent
            model_path: 模型路径
        """
        super().__init__("DDPG")
        
        if agent is not None:
            self.agent = agent
        else:
            self.agent = DDPGAgent()
            if model_path and os.path.exists(model_path):
                self.agent.load(model_path)
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        """
        运行DDPG调度 (完整指标输出)
        
        Args:
            tasks: 任务列表
            uav_resources: UAV资源
            cloud_resources: 云端资源
            
        Returns:
            BaselineResult: 调度结果 (包含所有4.1-4.4指标)
        """
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        self._reset_tracking(n_uavs)
        
        uav_state = copy.deepcopy(uav_resources)
        
        # 确保UAV有position属性
        for i, uav in enumerate(uav_state):
            if 'position' not in uav:
                uav['position'] = (400 + i * 300, 1000)
            if 'E_remain' not in uav:
                uav['E_remain'] = uav.get('E_max', 500e3)
        
        for task_idx, task in enumerate(tasks):
            # 获取状态
            state = self.agent.get_state(task, uav_state)
            
            # 选择动作（无噪声）
            action = self.agent.select_action(state, add_noise=False)
            
            # 解码动作
            uav_id, split_ratio = self.agent.decode_action(action)
            
            # 确保uav_id有效
            uav_id = min(uav_id, len(uav_state) - 1)
            
            # 执行动作
            result = self.agent.execute_action(task, uav_id, split_ratio, uav_state)
            
            # 计算边缘和云端计算量
            C_total = task.get('compute_size', 10e9)
            C_edge = C_total * split_ratio
            C_cloud = C_total * (1 - split_ratio)
            
            # 更新UAV资源
            if result['success']:
                uav_state[uav_id]['E_remain'] -= result['energy']
            
            priority = task.get('priority', 0.5)
            utility = 1.0 if result['success'] else 0.0
            
            result_dict = {
                'task_id': task_idx,
                'success': result['success'],
                'delay': result['delay'],
                'energy': result['energy'],
                'met_deadline': result['met_deadline'],
                'priority': priority,
                'uav_id': uav_id,
                'utility': utility
            }
            results.append(result_dict)
            
            # 跟踪资源使用
            self._track_task_result(result_dict, uav_id, 
                                    compute_edge=C_edge, compute_cloud=C_cloud)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


if __name__ == "__main__":
    print("测试DDPG模块")
    print("=" * 60)
    
    # 创建Agent
    config = DDPGConfig(n_uavs=5)
    agent = DDPGAgent(config)
    
    print(f"状态维度: {agent.state_dim}")
    print(f"动作维度: {agent.action_dim}")
    print(f"设备: {agent.device}")
    
    # 创建测试任务
    task = {
        'task_id': 0,
        'user_id': 0,
        'user_pos': (500, 500),
        'data_size': 2e6,
        'compute_size': 10e9,
        'deadline': 3.0,
        'priority': 0.7
    }
    
    uav_resources = [
        {'uav_id': i, 'position': (400 + i * 300, 1000), 
         'f_max': 15e9, 'E_max': 500e3, 'E_remain': 500e3}
        for i in range(5)
    ]
    
    # 测试状态获取
    state = agent.get_state(task, uav_resources)
    print(f"\n状态向量形状: {state.shape}")
    print(f"状态向量前6维(任务): {state[:6]}")
    
    # 测试动作选择
    action = agent.select_action(state, add_noise=True)
    print(f"\n动作向量: {action}")
    
    # 测试动作解码
    uav_id, split_ratio = agent.decode_action(action)
    print(f"选择UAV: {uav_id}, 切分比例: {split_ratio:.2f}")
    
    # 测试执行动作
    result = agent.execute_action(task, uav_id, split_ratio, uav_resources)
    print(f"\n执行结果: {result}")
    
    # 测试奖励计算
    reward = agent.compute_reward(task, result)
    print(f"奖励: {reward:.4f}")
    
    print("\n✓ DDPG模块测试通过!")
