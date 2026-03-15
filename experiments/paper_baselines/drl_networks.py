"""
DRL神经网络模块

包含:
1. MAPPO (Multi-Agent PPO with Attention) 用于 MAPPO-Attention

注: SAC (用于 Lyapunov-DRL) 已于 2026-03-14 移除

基于论文参数和SystemConfig配置
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random

from config.system_config import SystemConfig


# ============ 配置类 ============

@dataclass
class SACConfig:
    """SAC算法配置"""
    # 网络参数
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # 训练参数
    learning_rate: float = 3e-4
    gamma: float = 0.99           # 折扣因子
    tau: float = 0.005            # 软更新系数
    alpha: float = 0.2            # 熵系数
    auto_entropy: bool = True     # 自动调整熵系数

    # 经验回放
    buffer_size: int = 100000
    batch_size: int = 256
    warmup_steps: int = 1000      # 预热步数

    # 训练频率
    update_every: int = 1
    target_update_every: int = 1


@dataclass
class MAPPOConfig:
    """MAPPO算法配置"""
    # 网络参数
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    attention_dim: int = 128
    n_attention_heads: int = 4

    # PPO参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练参数
    n_epochs: int = 10            # 每次更新的epoch数
    batch_size: int = 64
    buffer_size: int = 10000

    # 探索
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995


# ============ 经验回放缓冲区 ============

class ReplayBuffer:
    """SAC经验回放缓冲区"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样"""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }

    def __len__(self):
        return self.size


class PPORolloutBuffer:
    """PPO轨迹缓冲区"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def push(self, state, action, reward, value, log_prob, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float):
        """计算GAE优势估计"""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0
        last_return = last_value

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

            last_return = self.rewards[t] + gamma * next_non_terminal * last_return
            self.returns[t] = last_return

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取批次数据"""
        indices = np.random.permutation(len(self.states))[:batch_size]

        return {
            'states': torch.FloatTensor(np.array(self.states)[indices]),
            'actions': torch.FloatTensor(np.array(self.actions)[indices]),
            'old_log_probs': torch.FloatTensor(np.array(self.log_probs)[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.states)


# ============ SAC网络 ============

class SACActorNetwork(nn.Module):
    """SAC Actor网络 (策略网络)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # 输出均值和标准差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 动作范围限制
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = torch.zeros(state.size(0), 1, device=state.device)
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化采样
            action = torch.tanh(x_t)

            # 计算log_prob
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        # 缩放到动作空间
        action = action * self.action_scale + self.action_bias

        return action, log_prob


class SACCriticNetwork(nn.Module):
    """SAC Critic网络 (Q网络)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q


class SACAgent:
    """SAC智能体"""

    def __init__(self, state_dim: int, action_dim: int, config: SACConfig = None, device: str = 'cpu'):
        self.config = config if config else SACConfig()
        self.device = device

        # 网络初始化
        self.actor = SACActorNetwork(state_dim, action_dim, self.config.actor_hidden_dim).to(device)
        self.critic1 = SACCriticNetwork(state_dim, action_dim, self.config.critic_hidden_dim).to(device)
        self.critic2 = SACCriticNetwork(state_dim, action_dim, self.config.critic_hidden_dim).to(device)

        # 目标网络
        self.target_critic1 = SACCriticNetwork(state_dim, action_dim, self.config.critic_hidden_dim).to(device)
        self.target_critic2 = SACCriticNetwork(state_dim, action_dim, self.config.critic_hidden_dim).to(device)

        # 复制参数
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.learning_rate)

        # 自动熵调整
        if self.config.auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config.alpha

        # 经验回放
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, state_dim, action_dim)

        # 训练计数
        self.total_steps = 0
        self.update_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.config.warmup_steps:
            return {}

        if self.total_steps % self.config.update_every != 0:
            return {}

        losses = {'actor_loss': 0, 'critic_loss': 0, 'alpha_loss': 0}
        n_updates = 0

        # 采样批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        losses['critic_loss'] = critic_loss.item()

        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses['actor_loss'] = actor_loss.item()

        # 更新Alpha
        if self.config.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            losses['alpha_loss'] = alpha_loss.item()

        # 软更新目标网络
        if self.update_count % self.config.target_update_every == 0:
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

        self.update_count += 1
        n_updates += 1

        return losses

    def _soft_update(self, target, source):
        """软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha if self.config.auto_entropy else None,
            'total_steps': self.total_steps
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        if self.config.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
        self.total_steps = checkpoint.get('total_steps', 0)


# ============ MAPPO网络 ============

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        attn_output, _ = self.attention(query, key, value)
        return self.norm(query + attn_output)


class MAPPOActorNetwork(nn.Module):
    """MAPPO Actor网络 (带注意力机制)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 attention_dim: int = 128, n_heads: int = 4):
        super().__init__()

        # 状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim)
        )

        # 注意力层
        self.attention = MultiHeadAttention(attention_dim, n_heads)

        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 动作输出 (离散动作)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, other_states: List[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 编码本地状态
        local_embed = self.state_encoder(state)

        # 如果有其他智能体状态，使用注意力
        if other_states is not None and len(other_states) > 0:
            # 编码其他智能体状态
            other_embeds = [self.state_encoder(s) for s in other_states]

            # 拼接为序列
            query = local_embed.unsqueeze(1)
            keys = torch.stack([local_embed] + other_embeds, dim=1)
            values = keys

            # 注意力
            attended = self.attention(query, keys, values).squeeze(1)
            combined = local_embed + 0.3 * attended
        else:
            combined = local_embed

        # 策略输出
        features = self.policy_net(combined)
        action_logits = self.action_head(features)

        return action_logits

    def get_action(self, state: torch.Tensor, other_states: List[torch.Tensor] = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        logits = self.forward(state, other_states)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)

        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor,
                        other_states: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        logits = self.forward(state, other_states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy


class MAPPOCriticNetwork(nn.Module):
    """MAPPO Critic网络 (价值网络)"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(state)


class MAPPOAgent:
    """MAPPO智能体"""

    def __init__(self, state_dim: int, action_dim: int, config: MAPPOConfig = None, device: str = 'cpu'):
        self.config = config if config else MAPPOConfig()
        self.device = device

        # 网络初始化
        self.actor = MAPPOActorNetwork(
            state_dim, action_dim,
            self.config.actor_hidden_dim,
            self.config.attention_dim,
            self.config.n_attention_heads
        ).to(device)
        self.critic = MAPPOCriticNetwork(state_dim, self.config.critic_hidden_dim).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

        # 轨迹缓冲区
        self.rollout_buffer = PPORolloutBuffer(self.config.buffer_size)

        # 探索率
        self.epsilon = self.config.epsilon_start

        # 训练计数
        self.total_steps = 0
        self.update_count = 0

    def select_action(self, state: np.ndarray, other_states: List[np.ndarray] = None,
                     deterministic: bool = False) -> Tuple[int, float]:
        """选择动作"""
        # Epsilon-greedy探索
        if not deterministic and random.random() < self.epsilon:
            # 随机动作
            action = random.randint(0, 1)  # 假设动作空间为2
            log_prob = np.log(0.5)
            return action, log_prob

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            other_tensors = [torch.FloatTensor(s).unsqueeze(0).to(self.device)
                           for s in other_states] if other_states else None

            action, log_prob = self.actor.get_action(state_tensor, other_tensors, deterministic)

        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储转移"""
        self.rollout_buffer.push(state, action, reward, value, log_prob, done)
        self.total_steps += 1

        # 更新探索率
        self.epsilon = max(self.config.epsilon_end,
                          self.epsilon * self.config.epsilon_decay)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """更新网络"""
        if len(self.rollout_buffer) < self.config.batch_size:
            return {}

        # 计算GAE
        self.rollout_buffer.compute_gae(
            self.config.gamma,
            self.config.gae_lambda,
            last_value
        )

        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

        # 多轮更新
        for epoch in range(self.config.n_epochs):
            batch = self.rollout_buffer.get_batch(self.config.batch_size)

            states = batch['states'].to(self.device)
            actions = batch['actions'].long().to(self.device)
            old_log_probs = batch['old_log_probs'].to(self.device)
            advantages = batch['advantages'].to(self.device)
            returns = batch['returns'].to(self.device)

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 更新Critic
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            # 更新Actor
            new_log_probs, entropy = self.actor.evaluate_actions(states, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                               1 + self.config.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.config.entropy_coef * entropy.mean()

            total_actor_loss = actor_loss + entropy_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            losses['actor_loss'] += actor_loss.item()
            losses['critic_loss'] += critic_loss.item()
            losses['entropy'] += entropy.mean().item()

        # 平均损失
        for key in losses:
            losses[key] /= self.config.n_epochs

        # 清空缓冲区
        self.rollout_buffer.clear()
        self.update_count += 1

        return losses

    def get_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
        return value.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.total_steps = checkpoint.get('total_steps', 0)


# ============ 状态/动作编码器 ============

class StateEncoder:
    """状态编码器 - 将环境状态转换为网络输入"""

    def __init__(self, n_uavs: int, system_config: SystemConfig):
        self.n_uavs = n_uavs
        self.config = system_config

        # 状态维度:
        # - 任务特征: data_size, total_flops, deadline, priority, user_pos(2) = 6
        # - UAV特征 * n_uavs: x, y, E_current/E_max, utilization, price = 5 * n_uavs
        self.state_dim = 6 + 5 * n_uavs

    def encode(self, task: Dict, uav_resources: List[Dict]) -> np.ndarray:
        """编码状态"""
        # 任务特征
        task_features = [
            task.get('data_size', task.get('D', 1e6)) / 1e7,  # 归一化
            task.get('total_flops', task.get('C_total', 10e9)) / 1e11,
            task.get('deadline', 2.0) / 10.0,
            task.get('priority', 0.5),
            task.get('user_pos', (100, 100))[0] / 500.0,
            task.get('user_pos', (100, 100))[1] / 500.0
        ]

        # UAV特征
        uav_features = []
        for uav in uav_resources:
            uav_features.extend([
                uav.get('x', 100) / 500.0,
                uav.get('y', 100) / 500.0,
                uav.get('E_current', uav.get('E_max', self.config.uav.E_max)) /
                    max(uav.get('E_max', self.config.uav.E_max), 1),
                uav.get('utilization', 0.0),
                uav.get('price', 1.0) / 2.0
            ])

        return np.array(task_features + uav_features, dtype=np.float32)


class ActionDecoder:
    """动作解码器 - 将网络输出转换为环境动作"""

    def __init__(self, n_uavs: int, n_split_options: int = 5):
        self.n_uavs = n_uavs
        self.n_split_options = n_split_options  # 切分选项数
        self.action_dim = n_uavs + 1  # UAV选择 + 切分选择

    def decode(self, action: int) -> Tuple[int, int]:
        """
        解码动作

        Returns:
            (uav_id, split_option)
            - uav_id: -1表示云端, 0到n_uavs-1表示UAV
            - split_option: 0-4 对应 0%, 25%, 50%, 75%, 100% 边缘计算
        """
        if action < self.n_split_options:
            # 云端执行
            return -1, action
        else:
            # UAV执行
            remaining = action - self.n_split_options
            uav_id = remaining // self.n_split_options
            split_option = remaining % self.n_split_options
            return uav_id, split_option

    def encode(self, uav_id: int, split_option: int) -> int:
        """编码动作为网络输出"""
        if uav_id == -1:
            return split_option
        else:
            return self.n_split_options + uav_id * self.n_split_options + split_option

    @property
    def total_actions(self) -> int:
        """总动作数"""
        return self.n_split_options + self.n_uavs * self.n_split_options
