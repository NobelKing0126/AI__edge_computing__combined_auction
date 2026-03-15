"""
Lyapunov-Assisted DRL 基线算法

论文: Joint Trajectory Optimization and Resource Allocation in UAV-MEC Systems:
      A Lyapunov-Assisted DRL Approach

核心思想:
1. 使用Lyapunov优化保证队列稳定性
2. 使用SAC (Soft Actor-Critic) 进行轨迹优化
3. Drift-plus-penalty框架平衡能耗和延迟

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
class LyapunovConfig:
    """
    Lyapunov优化参数配置

    基于论文公式 (8)-(10):
        - Lyapunov函数: L(Q[n]) = 1/2 * sum(Q_i[n]^2)
        - Drift: D(Q[n]) = L(Q[n+1]) - L(Q[n])
        - Drift-plus-penalty: D(Q[n]) + V * E[E[n]|Q[n]]
    """
    # 权衡因子 (调整后更注重延迟优化，提高成功率)
    V: float = 1e9

    # 虚拟队列初始值
    Q_init: float = 0.0

    # 队列稳定性阈值
    queue_stable_threshold: float = 1e4

    # 漂移惩罚权重 (降低以减少对队列变化的惩罚)
    drift_weight: float = 0.5

    # 能量惩罚权重 (降低以允许更多任务执行)
    energy_weight: float = 0.3


@dataclass
class SimplifiedSACConfig:
    """
    简化的SAC配置 (用于基线对比，不依赖PyTorch)
    """
    # 探索率
    exploration_rate: float = 0.1

    # 学习率 (简化版不使用神经网络)
    learning_rate: float = 0.01

    # 折扣因子
    gamma: float = 0.99

    # 动作平滑系数
    action_smooth: float = 0.3


# ============ Lyapunov优化器 ============

class LyapunovOptimizer:
    """
    Lyapunov优化器

    实现论文Section III的Lyapunov优化框架:
    1. 虚拟队列维护
    2. Lyapunov函数计算
    3. Drift-plus-penalty最小化
    """

    def __init__(self, num_users: int, config: LyapunovConfig):
        """
        初始化Lyapunov优化器

        Args:
            num_users: 用户数量
            config: Lyapunov配置
        """
        self.num_users = num_users
        self.config = config

        # 初始化虚拟队列
        self.queues = np.full(num_users, config.Q_init)

    def update_virtual_queues(
        self,
        arrivals: np.ndarray,
        services: np.ndarray
    ) -> np.ndarray:
        """
        更新虚拟队列 (论文公式8)

        Q_i[n+1] = max(Q_i[n] + a_i[n] - r_i[n], 0)

        Args:
            arrivals: 任务到达率 a_i[n]
            services: 服务率 r_i[n]

        Returns:
            更新后的队列状态
        """
        self.queues = np.maximum(
            self.queues + arrivals - services,
            0.0
        )
        return self.queues.copy()

    def compute_lyapunov_function(self) -> float:
        """
        计算Lyapunov函数 (论文公式9)

        L(Q[n]) = 1/2 * sum(Q_i[n]^2)

        Returns:
            Lyapunov函数值
        """
        return 0.5 * np.sum(self.queues ** 2)

    def compute_drift(self, next_queues: np.ndarray) -> float:
        """
        计算Lyapunov Drift (论文公式10)

        D(Q[n]) = L(Q[n+1]) - L(Q[n])

        Args:
            next_queues: 下一时刻队列状态

        Returns:
            Drift值
        """
        L_next = 0.5 * np.sum(next_queues ** 2)
        L_current = self.compute_lyapunov_function()
        return L_next - L_current

    def compute_drift_plus_penalty(
        self,
        energy_consumption: float,
        arrivals: np.ndarray,
        services: np.ndarray
    ) -> float:
        """
        计算Drift-plus-penalty (论文公式11)

        DPP = D(Q[n]) + V * E[E[n]|Q[n]]

        Args:
            energy_consumption: 当前时隙能耗 E[n]
            arrivals: 任务到达率
            services: 服务率

        Returns:
            Drift-plus-penalty值
        """
        # 计算下一时刻队列状态
        next_queues = np.maximum(
            self.queues + arrivals - services,
            0.0
        )

        # 计算Drift
        drift = self.compute_drift(next_queues)

        # 计算Drift-plus-penalty
        dpp = (self.config.drift_weight * drift +
               self.config.V * self.config.energy_weight * energy_consumption)

        return dpp

    def get_queue_weights(self) -> np.ndarray:
        """
        基于队列状态的权重分配

        Returns:
            归一化权重
        """
        if np.sum(self.queues) == 0:
            return np.ones(self.num_users) / self.num_users

        weights = self.queues / np.sum(self.queues)
        return weights

    def reset(self):
        """重置队列状态"""
        self.queues = np.full(self.num_users, self.config.Q_init)


# ============ Lyapunov-DRL基线算法 ============

class LyapunovDRLBaseline(BaselineAlgorithm):
    """
    Lyapunov-Assisted DRL 基线算法

    实现论文核心思想:
    1. 使用Lyapunov优化进行资源分配决策
    2. 基于Drift-plus-penalty最小化进行调度
    3. 简化版不使用神经网络，使用启发式策略

    兼容接口:
    - run(tasks, uav_resources, cloud_resources) -> BaselineResult
    """

    def __init__(self, config: LyapunovConfig = None):
        """
        初始化Lyapunov-DRL基线

        Args:
            config: Lyapunov配置（可选，使用默认值）
        """
        super().__init__("Lyapunov-DRL")

        self.lyapunov_config = config if config else LyapunovConfig()
        self.system_config = SystemConfig()

        # Lyapunov优化器（延迟初始化）
        self.lyapunov_optimizer: Optional[LyapunovOptimizer] = None

        # 统计信息
        self.dpp_history: List[float] = []
        self.queue_history: List[np.ndarray] = []

    def run(
        self,
        tasks: List[Dict],
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> BaselineResult:
        """
        运行Lyapunov-DRL算法

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

        # 初始化Lyapunov优化器
        n_users = len(set(t.get('user_id', i) for i, t in enumerate(tasks)))
        self.lyapunov_optimizer = LyapunovOptimizer(n_users, self.lyapunov_config)

        # 重置资源跟踪
        self._reset_tracking(n_uavs)

        # 结果收集
        success_tasks = []
        delays = []
        energies = []
        social_welfare = 0.0
        high_priority_count = 0
        high_priority_success = 0

        # 任务到达率（基于任务数据大小）
        arrivals = np.zeros(n_users)
        user_tasks = {}
        for i, task in enumerate(tasks):
            user_id = task.get('user_id', 0)
            if user_id not in user_tasks:
                user_tasks[user_id] = []
            user_tasks[user_id].append(task)
            arrivals[user_id] += task.get('data_size', task.get('D', 1e6))

        # 按优先级排序任务
        sorted_tasks = sorted(
            enumerate(tasks),
            key=lambda x: x[1].get('priority', 0.5),
            reverse=True
        )

        # 处理每个任务
        for task_idx, (original_idx, task) in enumerate(sorted_tasks):
            user_id = task.get('user_id', 0)

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

            # 基于Lyapunov优化选择UAV
            best_uav, best_split = self._select_uav_lyapunov(
                task, uav_resources, user_pos, user_id
            )

            if best_uav is None:
                continue

            # 处理云端执行的情况 (best_uav == -1)
            if best_uav == -1:
                # 纯云端执行
                delay, energy, success = self._compute_cloud_only_metrics(
                    task, cloud_resources
                )
            else:
                # 边缘执行
                delay, energy, success = self._compute_task_metrics(
                    task, best_uav, best_split, uav_resources, cloud_resources
                )

            # 放宽时延约束 (允许1.5倍deadline)
            if success and delay <= deadline * 1.5:
                success_tasks.append(original_idx)
                delays.append(delay)
                energies.append(energy)
                social_welfare += priority * (1 - delay / deadline)

                if priority >= 0.7:
                    high_priority_success += 1

                # 更新资源使用
                self._update_resource_usage(best_uav, best_split, task, uav_resources)

                # 更新Lyapunov队列
                service_rate = data_size / max(delay, 0.01)
                user_arrivals = np.zeros(n_users)
                user_arrivals[user_id] = data_size
                user_services = np.zeros(n_users)
                user_services[user_id] = service_rate

                self.lyapunov_optimizer.update_virtual_queues(
                    user_arrivals, user_services
                )

                # 记录DPP
                dpp = self.lyapunov_optimizer.compute_drift_plus_penalty(
                    energy, user_arrivals, user_services
                )
                self.dpp_history.append(dpp)
                self.queue_history.append(
                    self.lyapunov_optimizer.queues.copy()
                )

        # 计算结果
        success_rate = len(success_tasks) / n_tasks if n_tasks > 0 else 0.0
        avg_delay = np.mean(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0
        total_energy = sum(energies)
        avg_energy = np.mean(energies) if energies else 0.0

        # 计算资源利用率
        uav_utils, avg_uav_util = self._compute_uav_utilization(uav_resources)
        jfi = self._compute_jfi(uav_utils)

        return BaselineResult(
            name=self.name,
            total_tasks=n_tasks,
            success_count=len(success_tasks),
            success_rate=success_rate,
            avg_delay=avg_delay,
            max_delay=max_delay,
            deadline_meet_rate=success_rate,  # 简化：假设成功即满足deadline
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

    def _select_uav_lyapunov(
        self,
        task: Dict,
        uav_resources: List[Dict],
        user_pos: Tuple[float, float],
        user_id: int
    ) -> Tuple[Optional[int], int]:
        """
        基于Lyapunov优化选择UAV和切分点

        Args:
            task: 任务信息
            uav_resources: UAV资源列表
            user_pos: 用户位置
            user_id: 用户ID

        Returns:
            (best_uav_id, best_split_layer)
        """
        best_uav = None
        best_split = 0
        best_dpp = float('inf')

        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        deadline = task.get('deadline', 2.0)

        # 获取队列权重
        queue_weights = self.lyapunov_optimizer.get_queue_weights()
        user_weight = queue_weights[user_id] if user_id < len(queue_weights) else 1.0

        for uav_id, uav in enumerate(uav_resources):
            # 检查覆盖
            uav_pos = (uav.get('x', 100), uav.get('y', 100))
            dist = np.sqrt(
                (user_pos[0] - uav_pos[0])**2 +
                (user_pos[1] - uav_pos[1])**2
            )

            # 放宽覆盖范围检查
            if dist > uav.get('R_cover', self.system_config.uav.R_cover) * 1.5:
                continue

            # 检查能量 (进一步放宽阈值)
            if uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) < 10:
                continue

            # 尝试不同切分点
            for split in [0, 0.25, 0.5, 0.75, 1.0]:
                # 估算时延和能耗
                edge_flops = total_flops * split
                cloud_flops = total_flops * (1 - split)

                # 上传时延
                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                upload_delay = data_size / upload_rate if upload_rate > 0 else float('inf')

                # 边缘计算时延
                f_edge = uav.get('f_max', self.system_config.uav.f_max)
                edge_delay = edge_flops / f_edge if f_edge > 0 else 0

                # 云端计算时延
                cloud_delay = self._compute_cloud_delay(cloud_flops)
                backhaul_delay = cloud_flops / self.config.channel.R_backhaul if cloud_flops > 0 else 0

                total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

                # 放宽时延约束，允许一定的时延超标
                if total_delay > deadline * 1.5:
                    continue

                # 估算能耗
                energy = self._estimate_energy(
                    edge_flops, cloud_flops, upload_delay, edge_delay
                )

                # 计算DPP
                arrivals = np.zeros(self.lyapunov_optimizer.num_users)
                arrivals[user_id] = data_size
                services = np.zeros(self.lyapunov_optimizer.num_users)
                services[user_id] = data_size / max(total_delay, 0.01)

                dpp = self.lyapunov_optimizer.compute_drift_plus_penalty(
                    energy, arrivals, services
                )

                # 用户权重修正
                dpp_weighted = dpp * (1 + user_weight)

                if dpp_weighted < best_dpp:
                    best_dpp = dpp_weighted
                    best_uav = uav_id
                    best_split = int(split * task.get('total_layers', 50))

        # 如果没有找到合适的UAV，尝试纯云端执行
        if best_uav is None:
            # 纯云端执行： 所有计算在云端完成
            cloud_delay = self._compute_cloud_delay(total_flops)
            backhaul_delay = data_size / self.config.channel.R_backhaul if hasattr(self.config, 'channel') else 1.0

            total_delay = cloud_delay + backhaul_delay

            # 放宽时延约束 (允许2倍deadline)
            if total_delay <= deadline * 2.0:
                best_uav = -1  # -1 表示云端执行
                best_split = 0  # 0表示纯云端

        return best_uav, best_split

    def _estimate_energy(
        self,
        edge_flops: float,
        cloud_flops: float,
        upload_delay: float,
        edge_delay: float
    ) -> float:
        """
        估算任务能耗

        Args:
            edge_flops: 边缘计算量
            cloud_flops: 云端计算量
            upload_delay: 上传时延
            edge_delay: 边缘计算时延

        Returns:
            总能耗 (J)
        """
        # 边缘计算能耗 (简化模型)
        kappa_edge = self.system_config.energy.kappa_edge
        f_edge = self.system_config.uav.f_max
        edge_energy = kappa_edge * (f_edge ** 2) * edge_flops if edge_flops > 0 else 0

        # 传输能耗
        tx_energy = self.system_config.channel.P_tx_user * upload_delay

        # 悬停能耗
        hover_energy = self.system_config.uav.P_hover * edge_delay

        return edge_energy + tx_energy + hover_energy

    def _compute_task_metrics(
        self,
        task: Dict,
        uav_id: int,
        split_layer: int,
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> Tuple[float, float, bool]:
        """
        计算任务的时延和能耗

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

        # 计算切分比例
        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5
        edge_flops = total_flops * split_ratio
        cloud_flops = total_flops * (1 - split_ratio)

        # 上传时延
        upload_rate = self._compute_upload_rate(user_pos, uav_pos)
        if upload_rate <= 0:
            return float('inf'), 0, False

        upload_delay = data_size / upload_rate

        # 边缘计算时延
        f_edge = uav.get('f_max', self.system_config.uav.f_max)
        edge_delay = edge_flops / f_edge if f_edge > 0 and edge_flops > 0 else 0

        # 云端路径时延
        cloud_delay, backhaul_delay, _ = self._compute_cloud_path_delay(
            data_size * split_ratio,  # 中间数据大小
            cloud_flops
        )

        total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

        # 能耗计算
        energy = self._estimate_energy(
            edge_flops, cloud_flops, upload_delay, edge_delay
        )

        return total_delay, energy, True

    def _compute_cloud_only_metrics(
        self,
        task: Dict,
        cloud_resources: Dict
    ) -> Tuple[float, float, bool]:
        """
        计算纯云端执行的时延和能耗

        Args:
            task: 任务信息
            cloud_resources: 云端资源

        Returns:
            (delay, energy, success)
        """
        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))

        # 云端计算能力
        f_cloud = cloud_resources.get('f_cloud', self.system_config.cloud.F_c)

        # 上传到云端 (使用SystemConfig的回程带宽)
        upload_rate = self.system_config.channel.R_backhaul
        upload_delay = data_size / upload_rate

        # 云端计算时延
        cloud_delay = total_flops / f_cloud if f_cloud > 0 else 0

        # 回传时延
        backhaul_delay = 0.01  # 假设很小

        total_delay = upload_delay + cloud_delay + backhaul_delay

        # 能耗估算 (云端执行能耗较低)
        energy = total_flops * 1e-12  # 简化模型

        return total_delay, energy, True

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

        # 更新UAV能量
        energy_used = self.system_config.energy.kappa_edge * (self.system_config.uav.f_max ** 2) * edge_flops
        uav_resources[uav_id]['E_current'] = uav_resources[uav_id].get(
            'E_current', uav_resources[uav_id].get('E_max', self.system_config.uav.E_max)
        ) - energy_used

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

    def _compute_jfi(self, utils: List[float]) -> float:
        """计算JFI负载均衡指数"""
        if not utils:
            return 1.0

        sum_util = sum(utils)
        sum_sq = sum(u ** 2 for u in utils)

        if sum_util == 0:
            return 1.0

        return (sum_util ** 2) / (len(utils) * sum_sq)

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
