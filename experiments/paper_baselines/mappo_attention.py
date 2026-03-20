"""
MAPPO-Attention 基线算法

论文: Joint Trajectory and Resource Optimization of MEC-Assisted UAVs in Sub-THz Networks:
      A Resources-Based Multi-Agent Proximal Policy Optimization DRL With Attention Mechanism

核心思想:
1. 多智能体近端策略优化 (MAPPO)
2. 注意力机制编码多智能体观测
3. 差异化奖励函数 (能耗+延迟+公平性)
4. Sub-THz信道模型

兼容接口:
- 继承自 experiments.baselines.BaselineAlgorithm
- 返回 BaselineResult 格式结果
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.baselines import BaselineAlgorithm, BaselineResult
from config.system_config import SystemConfig


# ============ 配置类 ============

@dataclass
class SubTHzChannelConfig:
    """
    Sub-THz信道配置

    基于论文参数:
    - 频率范围: 298-313 GHz (2x2 MIMO LOS)
    - 适用于6G网络场景
    """
    # 频率参数
    frequency_min: float = 298e9      # 最低频率 298 GHz
    frequency_max: float = 313e9      # 最高频率 313 GHz
    center_frequency: float = 305.5e9 # 中心频率

    # 信道配置
    mimo_config: Tuple[int, int] = (2, 2)  # 2x2 MIMO
    los_mode: str = "dominant"  # 主导LOS路径

    # 带宽配置 (每UAV)
    bandwidth_uplink: float = 100e6    # 上行带宽 100 MHz
    bandwidth_downlink: float = 100e6  # 下行带宽 100 MHz
    num_subchannels: int = 10          # 子信道数

    # 功率配置
    tx_power_uav: float = 1.0    # UAV发射功率 (W)
    tx_power_user: float = 0.1   # 用户发射功率 (W)

    # THz特有参数
    molecular_absorption: bool = True   # 分子吸收
    rain_attenuation: bool = True       # 雨衰
    coverage_range: float = 100.0       # 覆盖范围 (m)


@dataclass
class MAPPOConfig:
    """
    MAPPO算法配置

    基于论文的Multi-Agent PPO with Attention机制
    """
    # PPO核心参数
    learning_rate: float = 3e-4
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95      # GAE参数
    clip_epsilon: float = 0.2     # PPO裁剪参数
    entropy_coef: float = 0.01    # 熵系数
    value_loss_coef: float = 0.5  # 价值损失系数

    # 网络参数 (简化版不使用神经网络)
    hidden_dim: int = 256
    attention_dim: int = 128
    n_attention_heads: int = 4

    # 奖励权重 (差异化奖励)
    beta_energy: float = 0.4      # 能量权重
    beta_delay: float = 0.4       # 延迟权重
    beta_fairness: float = 0.2    # 公平性权重

    # 探索参数
    exploration_rate: float = 0.1


# ============ 注意力编码器 (简化版) ============

class SimplifiedAttentionEncoder:
    """
    简化的注意力编码器 (不依赖PyTorch)

    用于编码多智能体观测
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def encode(
        self,
        local_obs: np.ndarray,
        other_obs: List[np.ndarray]
    ) -> np.ndarray:
        """
        使用注意力机制编码观测

        Args:
            local_obs: 本地观测
            other_obs: 其他智能体观测列表

        Returns:
            编码后的观测
        """
        if not other_obs:
            return local_obs

        # 简化的注意力权重计算
        weights = []
        for obs in other_obs:
            # 基于相似度的权重
            similarity = np.dot(local_obs.flatten(), obs.flatten())
            similarity /= (np.linalg.norm(local_obs) * np.linalg.norm(obs) + 1e-8)
            weights.append(np.exp(similarity))

        # 归一化
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        # 加权聚合
        other_obs_array = np.array([obs.flatten() for obs in other_obs])
        context = np.sum(weights[:, np.newaxis] * other_obs_array, axis=0)

        # 残差连接
        encoded = local_obs.flatten() + 0.3 * context

        return encoded


# ============ 差异化奖励计算器 ============

class DifferentiatedRewardCalculator:
    """
    差异化奖励计算器

    基于论文: 网络协调器为每个agent提供差异化奖励
    r = β₁ * U_energy + β₂ * U_delay + β₃ * U_fairness
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config

    def compute(
        self,
        energy_consumption: float,
        task_delay: float,
        fairness_index: float,
        alpha: float = 0.5
    ) -> float:
        """
        计算差异化奖励

        Args:
            energy_consumption: 能耗
            task_delay: 任务延迟
            fairness_index: 公平性指数
            alpha: 能耗-延迟权衡参数

        Returns:
            综合奖励值
        """
        # 能耗效用 (负值，越低越好)
        u_energy = -alpha * energy_consumption

        # 延迟效用 (负值，越低越好)
        u_delay = -(1 - alpha) * task_delay

        # 公平性效用 (正值，越高越好)
        u_fairness = fairness_index

        # 加权综合
        reward = (
            self.config.beta_energy * u_energy +
            self.config.beta_delay * u_delay +
            self.config.beta_fairness * u_fairness
        )

        return reward

    def compute_fairness_index(self, utilizations: List[float]) -> float:
        """
        计算公平性指数 (Jain's Fairness Index)

        Args:
            utilizations: 各UAV利用率列表

        Returns:
            公平性指数 [0, 1]
        """
        if not utilizations:
            return 1.0

        u = np.array(utilizations)
        sum_u = np.sum(u)
        sum_sq = np.sum(u ** 2)

        if sum_sq == 0:
            return 1.0

        jfi = (sum_u ** 2) / (len(u) * sum_sq)
        return jfi


# ============ MAPPO-Attention基线算法 ============

class MAPPOAttentionBaseline(BaselineAlgorithm):
    """
    MAPPO-Attention 基线算法

    实现论文核心思想:
    1. 多智能体协同决策 (通过注意力机制)
    2. 差异化奖励 (能耗+延迟+公平性)
    3. Sub-THz信道模型
    4. 简化版不使用神经网络，使用启发式策略

    兼容接口:
    - run(tasks, uav_resources, cloud_resources) -> BaselineResult
    """

    def __init__(
        self,
        mappo_config: MAPPOConfig = None,
        channel_config: SubTHzChannelConfig = None
    ):
        """
        初始化MAPPO-Attention基线

        Args:
            mappo_config: MAPPO配置（可选）
            channel_config: Sub-THz信道配置（可选）
        """
        super().__init__("MAPPO-Attention")

        self.mappo_config = mappo_config if mappo_config else MAPPOConfig()
        self.channel_config = channel_config if channel_config else SubTHzChannelConfig()
        self.system_config = SystemConfig()

        # 注意力编码器
        self.attention_encoder = SimplifiedAttentionEncoder(
            embed_dim=self.mappo_config.attention_dim,
            n_heads=self.mappo_config.n_attention_heads
        )

        # 奖励计算器
        self.reward_calculator = DifferentiatedRewardCalculator(self.mappo_config)

        # 历史记录
        self.reward_history: List[float] = []
        self.fairness_history: List[float] = []

        # 云端任务计数（用于资源竞争计算）
        self.cloud_task_count = 0

    def run(
        self,
        tasks: List[Dict],
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> BaselineResult:
        """
        运行MAPPO-Attention算法

        Args:
            tasks: 任务列表
            uav_resources: UAV资源列表
            cloud_resources: 云端资源

        Returns:
            BaselineResult
        """
        n_tasks = len(tasks)
        n_uavs = len(uav_resources)

        if n_tasks == 0:
            return self._create_empty_result()

        # 重置资源跟踪
        self._reset_tracking(n_uavs)

        # 结果收集
        success_tasks = []
        delays = []
        energies = []
        social_welfare = 0.0
        high_priority_count = 0
        high_priority_success = 0

        # 按优先级排序任务
        sorted_tasks = sorted(
            enumerate(tasks),
            key=lambda x: x[1].get('priority', 0.5),
            reverse=True
        )

        # 构建UAV观测向量
        uav_observations = self._build_uav_observations(uav_resources)

        # 处理每个任务
        for task_idx, (original_idx, task) in enumerate(sorted_tasks):
            # 获取任务参数
            data_size = task.get('data_size', task.get('D', 1e6))
            total_flops = task.get('total_flops', task.get('C_total', 10e9))
            deadline = task.get('deadline', 2.0)
            priority = task.get('priority', 0.5)
            user_pos = task.get('user_pos', (100, 100))
            if isinstance(user_pos, dict):
                user_pos = (user_pos.get('x', 100), user_pos.get('y', 100))

            # 高优先级统计
            if priority >= 0.7:
                high_priority_count += 1

            # 使用注意力机制选择UAV
            best_uav, best_split, attention_weights = self._select_uav_with_attention(
                task, uav_resources, uav_observations, user_pos
            )

            if best_uav is None:
                continue

            # 计算时延和能耗
            delay, energy, success = self._compute_task_metrics_subthz(
                task, best_uav, best_split, uav_resources, cloud_resources
            )

            # 时延约束 (与Proposed对齐，使用标准deadline)
            if success and delay <= deadline:
                success_tasks.append(original_idx)
                delays.append(delay)
                energies.append(energy)

                # 计算社会福利 (使用差异化奖励)
                utilizations = [
                    self.uav_compute_used.get(i, 0) / uav_resources[i].get('f_max', self.system_config.uav.f_max)
                    for i in range(n_uavs)
                ]
                fairness = self.reward_calculator.compute_fairness_index(utilizations)

                reward = self.reward_calculator.compute(
                    energy_consumption=energy,
                    task_delay=delay,
                    fairness_index=fairness
                )
                social_welfare += priority * (1 - delay / deadline) + reward * 0.1

                if priority >= 0.7:
                    high_priority_success += 1

                # 更新资源使用
                self._update_resource_usage(best_uav, best_split, task, uav_resources)

                # 记录历史
                self.fairness_history.append(fairness)
                self.reward_history.append(reward)

                # 更新UAV观测
                uav_observations = self._build_uav_observations(uav_resources)

        # 计算结果
        success_rate = len(success_tasks) / n_tasks if n_tasks > 0 else 0.0
        avg_delay = np.mean(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0
        total_energy = sum(energies)
        avg_energy = np.mean(energies) if energies else 0.0

        # 计算资源利用率
        uav_utils, avg_uav_util = self._compute_uav_utilization(uav_resources)
        jfi = self.reward_calculator.compute_fairness_index(uav_utils)

        return BaselineResult(
            name=self.name,
            total_tasks=n_tasks,
            success_count=len(success_tasks),
            success_rate=success_rate,
            avg_delay=avg_delay,
            max_delay=max_delay,
            deadline_meet_rate=success_rate,
            total_energy=total_energy,
            avg_energy=avg_energy,
            high_priority_rate=high_priority_success / high_priority_count if high_priority_count > 0 else 1.0,
            social_welfare=social_welfare,
            energy_efficiency=len(success_tasks) / total_energy if total_energy > 0 else 0.0,
            avg_uav_utilization=avg_uav_util,
            jfi_load_balance=jfi,
            cloud_utilization=0.0,
            channel_utilization=0.0,
            uav_utilizations=uav_utils,
            uav_loads=[self.uav_task_count.get(i, 0) for i in range(n_uavs)]
        )

    def _reset_tracking(self, n_uavs: int = 5):
        """重置资源跟踪（覆盖父类方法）"""
        # 调用父类方法
        super()._reset_tracking(n_uavs)
        # 重置云端任务计数
        self.cloud_task_count = 0

    def _build_uav_observations(
        self,
        uav_resources: List[Dict]
    ) -> List[np.ndarray]:
        """
        构建UAV观测向量

        Args:
            uav_resources: UAV资源列表

        Returns:
            观测向量列表
        """
        observations = []

        for uav in uav_resources:
            # 观测: 位置 + 能量 + 利用率
            obs = np.array([
                uav.get('x', 100) / 500.0,  # 归一化位置
                uav.get('y', 100) / 500.0,
                uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) / uav.get('E_max', self.system_config.uav.E_max),
                uav.get('utilization', 0.0),
                uav.get('price', 1.0) / 2.0  # 归一化价格
            ])
            observations.append(obs)

        return observations

    def _select_uav_with_attention(
        self,
        task: Dict,
        uav_resources: List[Dict],
        uav_observations: List[np.ndarray],
        user_pos: Tuple[float, float]
    ) -> Tuple[Optional[int], int, np.ndarray]:
        """
        使用注意力机制选择UAV

        Args:
            task: 任务信息
            uav_resources: UAV资源列表
            uav_observations: UAV观测列表
            user_pos: 用户位置

        Returns:
            (best_uav_id, best_split_layer, attention_weights)
        """
        best_uav = None
        best_split = 0
        best_score = float('-inf')
        attention_weights = np.zeros(len(uav_resources))

        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        deadline = task.get('deadline', 2.0)

        # 计算每个UAV的注意力权重
        for uav_id, (uav, obs) in enumerate(zip(uav_resources, uav_observations)):
            # 检查覆盖
            uav_pos = (uav.get('x', 100), uav.get('y', 100))
            dist = np.sqrt(
                (user_pos[0] - uav_pos[0])**2 +
                (user_pos[1] - uav_pos[1])**2
            )

            # 覆盖范围检查 (1.2倍覆盖半径)
            if dist > uav.get('R_cover', self.system_config.uav.R_cover) * 1.2:
                continue

            # 能量检查 (100J阈值)
            if uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) < 100:
                continue

            # 获取其他UAV的观测
            other_obs = [uav_observations[i] for i in range(len(uav_observations)) if i != uav_id]

            # 注意力编码
            encoded_obs = self.attention_encoder.encode(obs, other_obs)

            # 计算注意力权重
            att_weight = np.mean(encoded_obs)
            attention_weights[uav_id] = att_weight

            # 尝试不同切分点
            for split in [0, 0.25, 0.5, 0.75, 1.0]:
                # 估算时延和能耗
                edge_flops = total_flops * split
                cloud_flops = total_flops * (1 - split)

                # Sub-THz上传速率
                upload_rate = self._compute_subthz_upload_rate(user_pos, uav_pos)
                upload_delay = data_size / upload_rate if upload_rate > 0 else float('inf')

                # 边缘计算时延
                f_edge = uav.get('f_max', self.system_config.uav.f_max)
                edge_delay = edge_flops / f_edge if f_edge > 0 else 0

                # 云端计算时延
                cloud_delay = self._compute_cloud_delay(cloud_flops)
                backhaul_delay = self._compute_subthz_backhaul_delay(cloud_flops)

                total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

                # 严格时延约束 (必须满足deadline)
                if total_delay > deadline:
                    continue

                # 估算能耗
                energy = self._estimate_energy_subthz(
                    edge_flops, cloud_flops, upload_delay, edge_delay, dist
                )

                # 计算差异化奖励
                fairness = self.reward_calculator.compute_fairness_index(
                    [self.uav_compute_used.get(i, 0) / uav_resources[i].get('f_max', self.system_config.uav.f_max)
                     for i in range(len(uav_resources))]
                )

                reward = self.reward_calculator.compute(
                    energy_consumption=energy,
                    task_delay=total_delay,
                    fairness_index=fairness
                )

                # 综合评分
                score = reward + attention_weights[uav_id] * 0.1

                if score > best_score:
                    best_score = score
                    best_uav = uav_id
                    best_split = int(split * task.get('total_layers', 50))

        return best_uav, best_split, attention_weights

    def _compute_subthz_upload_rate(
        self,
        user_pos: Tuple[float, float],
        uav_pos: Tuple[float, float]
    ) -> float:
        """
        计算Sub-THz上行速率

        基于论文Sub-THz信道模型

        Args:
            user_pos: 用户位置
            uav_pos: UAV位置

        Returns:
            上行速率 (bps)
        """
        # 计算3D距离
        H = self.system_config.uav.H
        dist = np.sqrt(
            (user_pos[0] - uav_pos[0])**2 +
            (user_pos[1] - uav_pos[1])**2 +
            H**2
        )

        # Sub-THz路径损耗 (简化模型)
        # PL = FSPL + molecular absorption
        f_c = self.channel_config.center_frequency
        c = 3e8

        # 自由空间路径损耗
        fspl = 20 * np.log10(4 * np.pi * dist * f_c / c)

        # 分子吸收损耗 (水蒸气)
        molecular_loss = 0.1 * dist / 100  # dB/km

        total_loss = fspl + molecular_loss

        # 接收功率
        tx_power = self.channel_config.tx_power_user
        rx_power = tx_power * 10 ** (-total_loss / 10)

        # 噪声功率
        k_boltzmann = 1.38e-23
        T = 290  # 室温
        bandwidth = self.channel_config.bandwidth_uplink
        noise_power = k_boltzmann * T * bandwidth

        # SNR和速率 (Shannon)
        snr = rx_power / noise_power
        rate = bandwidth * np.log2(1 + snr)

        return max(rate, 1e6)  # 最小1 Mbps

    def _compute_subthz_backhaul_delay(self, cloud_flops: float) -> float:
        """
        计算Sub-THz回程延迟

        Args:
            cloud_flops: 云端计算量

        Returns:
            回程延迟 (s)
        """
        if cloud_flops <= 0:
            return 0.0

        # Sub-THz回程带宽
        backhaul_rate = self.channel_config.bandwidth_downlink * 10  # 假设10倍上行带宽

        # 传播延迟 (Sub-THz特性)
        propagation_delay = 0.01  # 10ms

        # 传输延迟
        tx_delay = cloud_flops / (backhaul_rate * 1e6)  # 简化

        return propagation_delay + tx_delay

    def _estimate_energy_subthz(
        self,
        edge_flops: float,
        cloud_flops: float,
        upload_delay: float,
        edge_delay: float,
        distance: float
    ) -> float:
        """
        估算Sub-THz任务能耗

        Args:
            edge_flops: 边缘计算量
            cloud_flops: 云端计算量
            upload_delay: 上传时延
            edge_delay: 边缘计算时延
            distance: 传输距离

        Returns:
            总能耗 (J)
        """
        # 边缘计算能耗
        kappa_edge = self.system_config.energy.kappa_edge
        f_edge = self.system_config.uav.f_max
        edge_energy = kappa_edge * (f_edge ** 2) * edge_flops if edge_flops > 0 else 0

        # Sub-THz传输能耗 (考虑分子吸收)
        tx_power = self.channel_config.tx_power_user
        molecular_factor = 1.0 + 0.1 * distance / 100  # 分子吸收增加能耗
        tx_energy = tx_power * upload_delay * molecular_factor

        # 悬停能耗
        hover_energy = self.system_config.uav.P_hover * edge_delay

        return edge_energy + tx_energy + hover_energy

    def _compute_task_metrics_subthz(
        self,
        task: Dict,
        uav_id: int,
        split_layer: int,
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> Tuple[float, float, bool]:
        """
        计算Sub-THz任务的时延和能耗

        Args:
            task: 任务信息
            uav_id: UAV ID
            split_layer: 切分层
            uav_resources: UAV资源
            cloud_resources: 云端资源

        Returns:
            (delay, energy, success)
        """
        uav = uav_resources[uav_id]
        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        total_layers = task.get('total_layers', 50)
        user_pos = task.get('user_pos', (100, 100))
        if isinstance(user_pos, dict):
            user_pos = (user_pos.get('x', 100), user_pos.get('y', 100))

        uav_pos = (uav.get('x', 100), uav.get('y', 100))
        dist = np.sqrt(
            (user_pos[0] - uav_pos[0])**2 +
            (user_pos[1] - uav_pos[1])**2
        )

        # 计算切分比例
        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5
        edge_flops = total_flops * split_ratio
        cloud_flops = total_flops * (1 - split_ratio)

        # 获取任务deadline
        deadline = task.get('deadline', 2.0)

        # Sub-THz上传时延
        upload_rate = self._compute_subthz_upload_rate(user_pos, uav_pos)
        if upload_rate <= 0:
            return float('inf'), 0, False

        upload_delay = data_size / upload_rate

        # 获取UAV当前状态
        queue_size = self.uav_task_count.get(uav_id, 0)
        uav_remaining_energy = uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max))

        # 考虑队列竞争的算力分配
        f_edge = uav.get('f_max', self.system_config.uav.f_max)
        effective_f_edge = f_edge / max(queue_size + 1, 1)

        # 边缘计算时延（考虑算力竞争）
        edge_delay = edge_flops / effective_f_edge if effective_f_edge > 0 and edge_flops > 0 else 0

        # 云端路径时延（考虑云端资源竞争）
        max_concurrent = cloud_resources.get('max_concurrent_tasks', 5)
        cloud_competition_factor = min(self.cloud_task_count / max_concurrent, 1.0) if max_concurrent > 0 else 0
        effective_cloud_flops = cloud_flops * (1 - cloud_competition_factor)

        cloud_delay = self._compute_cloud_delay(effective_cloud_flops) if cloud_flops > 0 else 0
        backhaul_delay = self._compute_subthz_backhaul_delay(cloud_flops)

        total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

        # 计算能量预算（与proposed算法一致）
        E_budget = min(uav_remaining_energy / (queue_size + 1), 0.3 * self.system_config.uav.E_max)

        # 计算实际能耗（使用有效算力）
        energy_edge = self.system_config.energy.kappa_edge * (effective_f_edge ** 2) * edge_flops
        energy_cloud = self._estimate_energy_subthz_cloud(cloud_flops, backhaul_delay)
        total_energy = energy_edge + energy_cloud

        # 严格检查约束（与proposed算法一致）
        T_comm = upload_delay + backhaul_delay
        T_budget = deadline - T_comm

        success = (edge_delay <= T_budget and
                   cloud_delay <= T_budget and
                   total_energy <= E_budget and
                   total_delay <= deadline)

        return total_delay, total_energy, success

    def _estimate_energy_subthz_cloud(self, cloud_flops: float, backhaul_delay: float) -> float:
        """
        估算云端计算能耗（新增方法）

        Args:
            cloud_flops: 云端计算量
            backhaul_delay: 回程时延

        Returns:
            云端能耗
        """
        if cloud_flops <= 0:
            return 0.0

        # 使用云端能耗系数（比边缘低）
        kappa_cloud = self.system_config.energy.kappa_cloud if hasattr(self.system_config.energy, 'kappa_cloud') else 1e-29

        # 云端计算能耗（使用总算力F_c）
        f_cloud = self.system_config.cloud.F_c if hasattr(self.system_config.cloud, 'F_c') else 4.0e9
        energy = kappa_cloud * (f_cloud ** 2) * cloud_flops

        return energy

    def _update_resource_usage(
        self,
        uav_id: int,
        split_layer: int,
        task: Dict,
        uav_resources: List[Dict]
    ):
        """
        更新资源使用状态

        Args:
            uav_id: UAV ID
            split_layer: 切分层
            task: 任务信息
            uav_resources: UAV资源
        """
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        total_layers = task.get('total_layers', 50)
        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5

        edge_flops = total_flops * split_ratio

        self.uav_compute_used[uav_id] = self.uav_compute_used.get(uav_id, 0) + edge_flops
        self.uav_task_count[uav_id] = self.uav_task_count.get(uav_id, 0) + 1

        # 如果有云端计算，增加云端任务计数
        if split_layer < total_layers:
            self.cloud_task_count += 1

        # 更新UAV能量
        energy_used = self.system_config.energy.kappa_edge * (self.system_config.uav.f_max ** 2) * edge_flops
        uav_resources[uav_id]['E_current'] = uav_resources[uav_id].get(
            'E_current', uav_resources[uav_id].get('E_max', self.system_config.uav.E_max)
        ) - energy_used

        # 更新利用率
        f_max = uav_resources[uav_id].get('f_max', self.system_config.uav.f_max)
        uav_resources[uav_id]['utilization'] = self.uav_compute_used[uav_id] / f_max

    def _compute_uav_utilization(
        self,
        uav_resources: List[Dict]
    ) -> Tuple[List[float], float]:
        """计算UAV利用率"""
        utils = []
        for i, uav in enumerate(uav_resources):
            f_max = uav.get('f_max', self.system_config.uav.f_max)
            used = self.uav_compute_used.get(i, 0)
            util = min(used / f_max, 1.0) if f_max > 0 else 0
            utils.append(util)

        avg_util = np.mean(utils) if utils else 0.0
        return utils, avg_util

    def _create_empty_result(self) -> BaselineResult:
        """创建空结果"""
        return BaselineResult(
            name=self.name,
            total_tasks=0,
            success_count=0,
            success_rate=0.0,
            avg_delay=0.0,
            max_delay=0.0,
            deadline_meet_rate=0.0,
            total_energy=0.0,
            avg_energy=0.0,
            high_priority_rate=1.0,
            social_welfare=0.0
        )
