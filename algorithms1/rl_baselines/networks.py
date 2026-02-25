"""
神经网络模块

定义Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor网络 (策略网络)
    
    输入: 状态向量
    输出: 动作向量 (UAV选择logits + 切分比例)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # 输出层使用更小的初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch, state_dim]
            
        Returns:
            action: 动作张量 [batch, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # 输出范围 [-1, 1]
        return action


class CriticNetwork(nn.Module):
    """
    Critic网络 (Q值网络)
    
    输入: 状态向量 + 动作向量
    输出: Q值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch, state_dim]
            action: 动作张量 [batch, action_dim]
            
        Returns:
            q_value: Q值 [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class OUNoise:
    """
    Ornstein-Uhlenbeck噪声
    
    用于DDPG的探索噪声
    """
    
    def __init__(self, action_dim: int, mu: float = 0.0, 
                 theta: float = 0.15, sigma: float = 0.2):
        """
        初始化OU噪声
        
        Args:
            action_dim: 动作维度
            mu: 均值
            theta: 回归速度
            sigma: 噪声强度
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """采样噪声"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class GaussianNoise:
    """
    高斯噪声
    
    用于TD3的探索噪声
    """
    
    def __init__(self, action_dim: int, sigma: float = 0.1):
        """
        初始化高斯噪声
        
        Args:
            action_dim: 动作维度
            sigma: 标准差
        """
        self.action_dim = action_dim
        self.sigma = sigma
    
    def sample(self) -> np.ndarray:
        """采样噪声"""
        return np.random.normal(0, self.sigma, self.action_dim)
    
    def reset(self):
        """重置（高斯噪声无状态）"""
        pass


if __name__ == "__main__":
    # 测试网络
    print("测试神经网络模块")
    print("=" * 50)
    
    state_dim = 26
    action_dim = 6
    batch_size = 32
    
    # 创建网络
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim, action_dim)
    
    # 测试前向传播
    state = torch.randn(batch_size, state_dim)
    action = actor(state)
    q_value = critic(state, action)
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"Actor输出形状: {action.shape}")
    print(f"Critic输出形状: {q_value.shape}")
    print(f"动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
    
    # 测试噪声
    ou_noise = OUNoise(action_dim)
    gaussian_noise = GaussianNoise(action_dim)
    
    ou_sample = ou_noise.sample()
    gaussian_sample = gaussian_noise.sample()
    
    print(f"\nOU噪声样本: {ou_sample[:3]}...")
    print(f"高斯噪声样本: {gaussian_sample[:3]}...")
    
    print("\n✓ 网络测试通过!")
