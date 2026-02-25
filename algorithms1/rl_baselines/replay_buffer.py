"""
经验回放缓冲区

用于存储和采样训练数据
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 元组
    """
    
    def __init__(self, capacity: int = 100000):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
        """
        添加经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区 (可选)
    
    基于TD误差对经验进行优先级采样
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """优先级采样"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    # 测试缓冲区
    print("测试经验回放缓冲区")
    print("=" * 50)
    
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加测试数据
    state_dim = 26
    action_dim = 6
    
    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i == 99
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 采样测试
    states, actions, rewards, next_states, dones = buffer.sample(32)
    
    print(f"采样批次:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next States: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
    
    print("\n✓ 缓冲区测试通过!")
