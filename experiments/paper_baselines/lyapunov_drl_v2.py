"""
Lyapunov-Assisted DRL 基线算法 V2

论文: Joint Trajectory Optimization and Resource Allocation in UAV-MEC Systems:
      A Lyapunov-Assisted DRL Approach

核心思想:
1. 使用Lyapunov优化保证队列稳定性
2. 使用SAC (Soft Actor-Critic) 进行决策优化
3. Drift-plus-penalty框架平衡能耗和延迟
4. 完整的训练过程支持
5. 集成凸优化解析解和统一价格模型

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

# 导入DRL网络模块
try:
    from experiments.paper_baselines.drl_networks import (
        SACAgent, SACConfig, StateEncoder, ActionDecoder
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 导入新模块
try:
    from algorithms.optimization.convex_solver import (
        ConvexSolver, ConvexSolverConfig, AllocationResult
    )
    from algorithms.pricing.unified_pricing import (
        UnifiedPricingModel, PricingConfig
    )
    from models.user_benefit import UserBenefitModel, UserBenefitConfig
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False


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
    # 权衡因子
    V: float = 1e9

    # 虚拟队列初始值
    Q_init: float = 0.0

    # 队列稳定性阈值
    queue_stable_threshold: float = 1e4

    # 漂移惩罚权重
    drift_weight: float = 0.5

    # 能量惩罚权重
    energy_weight: float = 0.3


@dataclass
class TrainingConfig:
    """训练配置"""
    # 是否使用训练模式
    enable_training: bool = True

    # 预训练模型路径
    pretrained_model_path: str = None

    # 保存模型路径
    save_model_path: str = "models/lyapunov_drl_model.pt"

    # 训练频率 (每处理多少任务训练一次)
    train_every: int = 10

    # 是否在线学习
    online_learning: bool = True

    # 奖励缩放
    reward_scale: float = 1.0

    # 延迟奖励权重
    delay_reward_weight: float = 0.4

    # 能耗奖励权重
    energy_reward_weight: float = 0.3

    # 成功奖励
    success_reward: float = 1.0

    # 失败惩罚
    fail_penalty: float = -1.0


# ============ Lyapunov优化器 ============

class LyapunovOptimizer:
    """
    Lyapunov优化器

    实现论文Section III的Lyapunov优化框架:
    1. 虚拟队列维护
    2. Lyapunov函数计算
    3. Drift-plus-penalty最小化

    公式参考 (论文公式8-11):
        - Lyapunov函数: L(Q[n]) = 1/2 * sum(Q_i[n]^2)
        - Drift: D(Q[n]) = L(Q[n+1]) - L(Q[n])
        - Drift-plus-penalty: D(Q[n]) + V * E[E[n]|Q[n]]
    """

    def __init__(self, num_users: int, config: LyapunovConfig):
        self.num_users = num_users
        self.config = config
        self.queues = np.full(num_users, config.Q_init)
        self.queue_history: List[np.ndarray] = []
        self.drift_history: List[float] = []

    def update_virtual_queues(
        self,
        arrivals: np.ndarray,
        services: np.ndarray
    ) -> np.ndarray:
        """
        更新虚拟队列 (论文公式8)

        Q[n+1] = max(Q[n] + A[n] - S[n], 0)

        Args:
            arrivals: 到达率向量
            services: 服务率向量

        Returns:
            np.ndarray: 更新后的队列状态
        """
        self.queues = np.maximum(
            self.queues + arrivals - services,
            0.0
        )
        self.queue_history.append(self.queues.copy())
        return self.queues.copy()

    def compute_lyapunov_function(self) -> float:
        """
        计算Lyapunov函数 (论文公式9)

        L(Q[n]) = 1/2 * sum(Q_i[n]^2)

        Returns:
            float: Lyapunov函数值
        """
        return 0.5 * np.sum(self.queues ** 2)

    def compute_drift(self, next_queues: np.ndarray) -> float:
        """
        计算Lyapunov Drift (论文公式10)

        D(Q[n]) = L(Q[n+1]) - L(Q[n])

        Args:
            next_queues: 下一时刻队列状态

        Returns:
            float: Drift值
        """
        L_next = 0.5 * np.sum(next_queues ** 2)
        L_current = self.compute_lyapunov_function()
        drift = L_next - L_current
        self.drift_history.append(drift)
        return drift

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
            energy_consumption: 能耗
            arrivals: 到达率向量
            services: 服务率向量

        Returns:
            float: Drift-plus-penalty值
        """
        next_queues = np.maximum(
            self.queues + arrivals - services,
            0.0
        )
        drift = self.compute_drift(next_queues)
        dpp = (self.config.drift_weight * drift +
               self.config.V * self.config.energy_weight * energy_consumption)
        return dpp

    def compute_dpp_detailed(
        self,
        energy_consumption: float,
        delay_penalty: float,
        arrivals: np.ndarray,
        services: np.ndarray
    ) -> Dict[str, float]:
        """
        计算详细的Drift-plus-penalty分解

        Args:
            energy_consumption: 能耗
            delay_penalty: 时延惩罚
            arrivals: 到达率向量
            services: 服务率向量

        Returns:
            Dict[str, float]: 分解的DPP分量
        """
        next_queues = np.maximum(
            self.queues + arrivals - services,
            0.0
        )
        drift = self.compute_drift(next_queues)

        energy_penalty = self.config.V * self.config.energy_weight * energy_consumption
        delay_component = self.config.V * (1 - self.config.energy_weight) * delay_penalty

        total_dpp = self.config.drift_weight * drift + energy_penalty + delay_component

        return {
            'drift': drift,
            'energy_penalty': energy_penalty,
            'delay_penalty': delay_component,
            'total_dpp': total_dpp,
            'queue_mean': np.mean(self.queues),
            'queue_max': np.max(self.queues)
        }

    def get_queue_weights(self) -> np.ndarray:
        """
        基于队列状态的权重分配

        权重与队列长度成正比，确保长队列优先服务

        Returns:
            np.ndarray: 权重向量
        """
        if np.sum(self.queues) == 0:
            return np.ones(self.num_users) / self.num_users
        weights = self.queues / np.sum(self.queues)
        return weights

    def is_stable(self) -> bool:
        """
        检查队列是否稳定

        Returns:
            bool: 队列是否稳定
        """
        return np.max(self.queues) < self.config.queue_stable_threshold

    def reset(self):
        """重置队列状态"""
        self.queues = np.full(self.num_users, self.config.Q_init)
        self.queue_history = []
        self.drift_history = []

    def get_statistics(self) -> Dict[str, float]:
        """
        获取队列统计信息

        Returns:
            Dict[str, float]: 统计信息
        """
        return {
            'queue_mean': np.mean(self.queues),
            'queue_std': np.std(self.queues),
            'queue_max': np.max(self.queues),
            'queue_min': np.min(self.queues),
            'lyapunov_function': self.compute_lyapunov_function(),
            'is_stable': self.is_stable()
        }


# ============ Lyapunov-DRL基线算法 V2 ============

class LyapunovDRLBaselineV2(BaselineAlgorithm):
    """
    Lyapunov-Assisted DRL 基线算法 V2

    实现论文核心思想:
    1. 使用Lyapunov优化进行资源分配决策
    2. 使用SAC进行在线学习和决策优化
    3. Drift-plus-penalty框架平衡能耗和延迟
    4. 集成凸优化解析解和统一价格模型

    兼容接口:
    - run(tasks, uav_resources, cloud_resources) -> BaselineResult
    """

    def __init__(
        self,
        lyapunov_config: LyapunovConfig = None,
        sac_config: 'SACConfig' = None,
        training_config: TrainingConfig = None,
        device: str = 'cpu'
    ):
        """
        初始化Lyapunov-DRL基线V2

        Args:
            lyapunov_config: Lyapunov配置
            sac_config: SAC配置
            training_config: 训练配置
            device: 计算设备 ('cpu' 或 'cuda')
        """
        super().__init__("Lyapunov-DRL-V2")

        self.lyapunov_config = lyapunov_config if lyapunov_config else LyapunovConfig()
        self.training_config = training_config if training_config else TrainingConfig()
        self.system_config = SystemConfig()
        self.device = device

        # Lyapunov优化器（延迟初始化）
        self.lyapunov_optimizer: Optional[LyapunovOptimizer] = None

        # SAC智能体（延迟初始化）
        self.sac_agent: Optional['SACAgent'] = None
        self.state_encoder: Optional['StateEncoder'] = None
        self.action_decoder: Optional['ActionDecoder'] = None

        # SAC配置
        if TORCH_AVAILABLE and sac_config is None:
            self.sac_config = SACConfig()
        else:
            self.sac_config = sac_config

        # 集成新模块
        if NEW_MODULES_AVAILABLE:
            self.convex_solver = ConvexSolver(
                ConvexSolverConfig(
                    kappa_edge=self.system_config.energy.kappa_edge,
                    kappa_cloud=self.system_config.energy.kappa_cloud
                ),
                self.system_config
            )
            self.pricing_model = UnifiedPricingModel(
                PricingConfig(
                    c_edge_base=self.system_config.pricing.c_edge_base,
                    c_cloud_base=self.system_config.pricing.c_cloud_base,
                    gamma_F=self.system_config.pricing.gamma_F,
                    F_threshold=self.system_config.pricing.F_threshold
                ),
                self.system_config
            )
        else:
            self.convex_solver = None
            self.pricing_model = None

        # 统计信息
        self.dpp_history: List[float] = []
        self.queue_history: List[np.ndarray] = []
        self.training_losses: List[Dict[str, float]] = []
        self.total_tasks_processed = 0

        # 资源跟踪
        self.uav_compute_used: Dict[int, float] = {}
        self.uav_task_count: Dict[int, int] = {}

        # 加载预训练模型
        if (self.training_config.pretrained_model_path and
            TORCH_AVAILABLE and self.sac_agent is not None):
            self._load_pretrained_model()

    def _initialize_agent(self, n_uavs: int):
        """初始化SAC智能体"""
        if not TORCH_AVAILABLE:
            return

        # 状态编码器
        self.state_encoder = StateEncoder(n_uavs, self.system_config)

        # 动作解码器
        self.action_decoder = ActionDecoder(n_uavs, n_split_options=5)

        # SAC智能体
        self.sac_agent = SACAgent(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.total_actions,
            config=self.sac_config,
            device=self.device
        )

        # 加载预训练模型
        if self.training_config.pretrained_model_path:
            self._load_pretrained_model()

    def _reset_tracking(self, n_uavs: int):
        """重置资源跟踪"""
        self.uav_compute_used = {i: 0.0 for i in range(n_uavs)}
        self.uav_task_count = {i: 0 for i in range(n_uavs)}

    def _load_pretrained_model(self):
        """加载预训练模型"""
        if self.sac_agent and os.path.exists(self.training_config.pretrained_model_path):
            try:
                self.sac_agent.load(self.training_config.pretrained_model_path)
                print(f"  [Lyapunov-DRL-V2] 加载预训练模型: {self.training_config.pretrained_model_path}")
            except Exception as e:
                print(f"  [Lyapunov-DRL-V2] 加载模型失败: {e}")

    def _save_model(self):
        """保存模型"""
        if self.sac_agent and self.training_config.save_model_path:
            try:
                os.makedirs(os.path.dirname(self.training_config.save_model_path), exist_ok=True)
                self.sac_agent.save(self.training_config.save_model_path)
            except Exception as e:
                print(f"  [Lyapunov-DRL-V2] 保存模型失败: {e}")

    def _compute_reward(
        self,
        task: Dict,
        delay: float,
        energy: float,
        success: bool,
        deadline: float
    ) -> float:
        """
        计算奖励

        结合Lyapunov DPP和任务执行结果
        """
        config = self.training_config

        if success and delay <= deadline:
            # 成功且满足deadline
            delay_reward = config.delay_reward_weight * (1 - delay / deadline)
            energy_reward = -config.energy_reward_weight * (energy / 1e6)  # 归一化
            success_bonus = config.success_reward

            reward = (delay_reward + energy_reward + success_bonus) * config.reward_scale
        else:
            # 失败或超时
            reward = config.fail_penalty * config.reward_scale

        return reward

    def _solve_optimal_allocation(
        self,
        C_edge: float,
        C_cloud: float,
        f_avail: float,
        E_remain: float,
        E_max: float
    ) -> Tuple[float, float, float]:
        """
        使用凸优化求解最优资源分配

        基于idea118.txt 2.6节的闭式解

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_avail: 可用边缘算力 (FLOPS)
            E_remain: 剩余能量 (J)
            E_max: 最大能量 (J)

        Returns:
            Tuple[float, float, float]: (f_edge*, f_cloud*, compute_delay)
        """
        if NEW_MODULES_AVAILABLE and self.convex_solver is not None:
            # 使用新的凸优化求解器
            E_budget = min(E_remain / 5, 0.3 * E_max)  # 单任务能量预算

            result = self.convex_solver.solve(
                C_edge=C_edge,
                C_cloud=C_cloud,
                f_avail=f_avail,
                f_cloud_max=self.system_config.cloud.F_per_task_max,
                E_budget=E_budget
            )

            return result.f_edge_star, result.f_cloud_star, result.compute_delay
        else:
            # 降级为简单启发式
            f_edge = min(f_avail, self.system_config.uav.f_max * 0.5)
            f_cloud = self.system_config.cloud.F_per_task_max

            t_edge = C_edge / f_edge if f_edge > 0 and C_edge > 0 else 0
            t_cloud = C_cloud / f_cloud if f_cloud > 0 and C_cloud > 0 else 0

            return f_edge, f_cloud, t_edge + t_cloud

    def _compute_drift_plus_penalty_enhanced(
        self,
        energy_consumption: float,
        delay: float,
        deadline: float,
        user_id: int,
        n_users: int
    ) -> Dict[str, float]:
        """
        计算增强的Drift-plus-penalty

        结合队列状态、能耗和时延

        Args:
            energy_consumption: 能耗
            delay: 时延
            deadline: 截止时间
            user_id: 用户ID
            n_users: 用户数量

        Returns:
            Dict[str, float]: DPP分解分量
        """
        # 构建到达率和服务率向量
        arrivals = np.zeros(n_users)
        services = np.zeros(n_users)

        # 到达率基于任务数据量
        arrivals[user_id] = 1.0
        services[user_id] = 1.0 / max(delay, 0.01)

        # 时延惩罚
        delay_penalty = max(0, delay - deadline) / deadline if deadline > 0 else 0

        return self.lyapunov_optimizer.compute_dpp_detailed(
            energy_consumption,
            delay_penalty,
            arrivals,
            services
        )

    def _select_action_drl(
        self,
        task: Dict,
        uav_resources: List[Dict],
        user_pos: Tuple[float, float],
        user_id: int,
        training: bool = True
    ) -> Tuple[int, int, float]:
        """
        使用DRL选择动作

        Returns:
            (uav_id, split_layer, log_prob)
        """
        if not TORCH_AVAILABLE or self.sac_agent is None:
            # 回退到启发式方法
            return self._select_action_heuristic(task, uav_resources, user_pos, user_id)

        # 编码状态
        state = self.state_encoder.encode(task, uav_resources)

        # 使用SAC选择动作
        deterministic = not training or (self.total_tasks_processed <
                                         self.sac_config.warmup_steps)
        action = self.sac_agent.select_action(state, deterministic)

        # 解码动作
        uav_id, split_option = self.action_decoder.decode(int(action))

        # 转换split_option为split_layer
        split_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        split_ratio = split_ratios[split_option]
        total_layers = task.get('total_layers', 50)
        split_layer = int(split_ratio * total_layers)

        return uav_id, split_layer, 0.0  # log_prob暂不使用

    def _select_action_heuristic(
        self,
        task: Dict,
        uav_resources: List[Dict],
        user_pos: Tuple[float, float],
        user_id: int
    ) -> Tuple[int, int, float]:
        """
        启发式动作选择（回退方法）
        """
        best_uav = None
        best_split = 0
        best_dpp = float('inf')

        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        deadline = task.get('deadline', 2.0)

        queue_weights = self.lyapunov_optimizer.get_queue_weights()
        user_weight = queue_weights[user_id] if user_id < len(queue_weights) else 1.0

        for uav_id, uav in enumerate(uav_resources):
            uav_pos = (uav.get('x', 100), uav.get('y', 100))
            dist = np.sqrt(
                (user_pos[0] - uav_pos[0])**2 +
                (user_pos[1] - uav_pos[1])**2
            )

            if dist > uav.get('R_cover', self.system_config.uav.R_cover) * 1.5:
                continue

            if uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) < 10:
                continue

            for split in [0, 0.25, 0.5, 0.75, 1.0]:
                edge_flops = total_flops * split
                cloud_flops = total_flops * (1 - split)

                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                upload_delay = data_size / upload_rate if upload_rate > 0 else float('inf')

                f_edge = uav.get('f_max', self.system_config.uav.f_max)
                edge_delay = edge_flops / f_edge if f_edge > 0 else 0

                cloud_delay = self._compute_cloud_delay(cloud_flops)
                backhaul_delay = cloud_flops / self.system_config.channel.R_backhaul if cloud_flops > 0 else 0

                total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

                if total_delay > deadline * 1.5:
                    continue

                energy = self._estimate_energy(edge_flops, cloud_flops, upload_delay, edge_delay)

                arrivals = np.zeros(self.lyapunov_optimizer.num_users)
                arrivals[user_id] = data_size
                services = np.zeros(self.lyapunov_optimizer.num_users)
                services[user_id] = data_size / max(total_delay, 0.01)

                dpp = self.lyapunov_optimizer.compute_drift_plus_penalty(
                    energy, arrivals, services
                )
                dpp_weighted = dpp * (1 + user_weight)

                if dpp_weighted < best_dpp:
                    best_dpp = dpp_weighted
                    best_uav = uav_id
                    best_split = int(split * task.get('total_layers', 50))

        if best_uav is None:
            cloud_delay = self._compute_cloud_delay(total_flops)
            backhaul_delay = data_size / self.system_config.channel.R_backhaul
            total_delay = cloud_delay + backhaul_delay
            if total_delay <= deadline * 2.0:
                best_uav = -1
                best_split = 0

        return best_uav, best_split, 0.0

    def run(
        self,
        tasks: List[Dict],
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> BaselineResult:
        """
        运行Lyapunov-DRL算法V2

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

        # 初始化SAC智能体
        if TORCH_AVAILABLE and self.sac_agent is None:
            self._initialize_agent(n_uavs)

        # 重置资源跟踪
        self._reset_tracking(n_uavs)

        # 结果收集
        success_tasks = []
        delays = []
        energies = []
        social_welfare = 0.0
        high_priority_count = 0
        high_priority_success = 0

        # 任务到达率
        arrivals = np.zeros(n_users)
        for i, task in enumerate(tasks):
            user_id = task.get('user_id', 0)
            arrivals[user_id] += task.get('data_size', task.get('D', 1e6))

        # 按优先级排序任务
        sorted_tasks = sorted(
            enumerate(tasks),
            key=lambda x: x[1].get('priority', 0.5),
            reverse=True
        )

        # 存储转移用于训练
        last_state = None
        last_action = None

        # 处理每个任务
        for task_idx, (original_idx, task) in enumerate(sorted_tasks):
            user_id = task.get('user_id', 0)

            data_size = task.get('data_size', task.get('D', 1e6))
            total_flops = task.get('total_flops', task.get('C_total', 10e9))
            deadline = task.get('deadline', 2.0)
            priority = task.get('priority', 0.5)
            user_pos = task.get('user_pos', (100, 100))
            if isinstance(user_pos, dict):
                user_pos = (user_pos.get('x', 100), user_pos.get('y', 100))

            if priority >= 0.7:
                high_priority_count += 1

            # 编码当前状态
            if TORCH_AVAILABLE and self.state_encoder:
                current_state = self.state_encoder.encode(task, uav_resources)
            else:
                current_state = None

            # 使用DRL或启发式选择动作
            use_drl = (TORCH_AVAILABLE and
                      self.sac_agent is not None and
                      self.training_config.enable_training)

            if use_drl:
                best_uav, best_split, _ = self._select_action_drl(
                    task, uav_resources, user_pos, user_id, training=True
                )
            else:
                best_uav, best_split, _ = self._select_action_heuristic(
                    task, uav_resources, user_pos, user_id
                )

            if best_uav is None:
                continue

            # 计算任务指标
            if best_uav == -1:
                delay, energy, success = self._compute_cloud_only_metrics(
                    task, cloud_resources
                )
            else:
                delay, energy, success = self._compute_task_metrics(
                    task, best_uav, best_split, uav_resources, cloud_resources
                )

            # 判断成功
            task_success = success and delay <= deadline * 1.5

            if task_success:
                success_tasks.append(original_idx)
                delays.append(delay)
                energies.append(energy)
                social_welfare += priority * (1 - delay / deadline)

                if priority >= 0.7:
                    high_priority_success += 1

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

                dpp = self.lyapunov_optimizer.compute_drift_plus_penalty(
                    energy, user_arrivals, user_services
                )
                self.dpp_history.append(dpp)
                self.queue_history.append(
                    self.lyapunov_optimizer.queues.copy()
                )

            # DRL训练
            if (TORCH_AVAILABLE and
                self.sac_agent is not None and
                self.training_config.enable_training and
                self.training_config.online_learning):

                # 计算奖励
                reward = self._compute_reward(task, delay, energy, task_success, deadline)

                # 编码下一个状态
                next_state = self.state_encoder.encode(task, uav_resources)

                # 存储转移
                if last_state is not None and last_action is not None:
                    done = (task_idx == len(sorted_tasks) - 1)
                    self.sac_agent.store_transition(
                        last_state, last_action, reward, next_state, done
                    )

                # 训练
                if (self.total_tasks_processed > self.sac_config.warmup_steps and
                    self.total_tasks_processed % self.training_config.train_every == 0):
                    losses = self.sac_agent.update()
                    if losses:
                        self.training_losses.append(losses)

                # 保存当前状态和动作
                last_state = current_state
                if current_state is not None:
                    # 编码动作
                    split_ratio = best_split / task.get('total_layers', 50)
                    split_options = [0, 0.25, 0.5, 0.75, 1.0]
                    split_option = min(range(len(split_options)),
                                      key=lambda i: abs(split_options[i] - split_ratio))
                    last_action = self.action_decoder.encode(best_uav, split_option)

            self.total_tasks_processed += 1

        # 保存模型
        if self.training_config.save_model_path and self.sac_agent:
            self._save_model()

        # 计算结果
        success_rate = len(success_tasks) / n_tasks if n_tasks > 0 else 0.0
        avg_delay = np.mean(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0
        total_energy = sum(energies)
        avg_energy = np.mean(energies) if energies else 0.0

        uav_utils, avg_uav_util = self._compute_uav_utilization(uav_resources)
        jfi = self._compute_jfi(uav_utils)

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

    def _compute_upload_rate(
        self,
        user_pos: Tuple[float, float],
        uav_pos: Tuple[float, float]
    ) -> float:
        """计算上传速率"""
        dist = np.sqrt(
            (user_pos[0] - uav_pos[0])**2 +
            (user_pos[1] - uav_pos[1])**2 +
            self.system_config.uav.H**2
        )

        W = self.system_config.channel.W
        beta_0 = self.system_config.channel.beta_0
        N_0 = self.system_config.channel.N_0
        P_tx = self.system_config.channel.P_tx_user

        h = beta_0 / (dist ** 2 + 1e-6)
        snr = P_tx * h / (N_0 * W)
        rate = W * np.log2(1 + snr)

        return max(rate, 1e6)

    def _compute_cloud_delay(self, cloud_flops: float) -> float:
        """计算云端计算延迟"""
        if cloud_flops <= 0:
            return 0.0
        f_cloud = self.system_config.cloud.F_c
        T_prop = self.system_config.cloud.T_propagation
        compute_delay = cloud_flops / f_cloud
        return compute_delay + T_prop

    def _estimate_energy(
        self,
        edge_flops: float,
        cloud_flops: float,
        upload_delay: float,
        edge_delay: float
    ) -> float:
        """估算能耗"""
        kappa_edge = self.system_config.energy.kappa_edge
        f_edge = self.system_config.uav.f_max
        edge_energy = kappa_edge * (f_edge ** 2) * edge_flops if edge_flops > 0 else 0

        tx_energy = self.system_config.channel.P_tx_user * upload_delay
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
        """计算任务指标"""
        uav = uav_resources[uav_id]
        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        total_layers = task.get('total_layers', 50)
        user_pos = task.get('user_pos', (100, 100))
        if isinstance(user_pos, dict):
            user_pos = (user_pos.get('x', 100), user_pos.get('y', 100))

        uav_pos = (uav.get('x', 100), uav.get('y', 100))

        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5
        edge_flops = total_flops * split_ratio
        cloud_flops = total_flops * (1 - split_ratio)

        upload_rate = self._compute_upload_rate(user_pos, uav_pos)
        if upload_rate <= 0:
            return float('inf'), 0, False

        upload_delay = data_size / upload_rate

        f_edge = uav.get('f_max', self.system_config.uav.f_max)
        edge_delay = edge_flops / f_edge if f_edge > 0 and edge_flops > 0 else 0

        cloud_delay = self._compute_cloud_delay(cloud_flops)
        backhaul_delay = cloud_flops / self.system_config.channel.R_backhaul if cloud_flops > 0 else 0

        # 添加DNN推理时延
        dnn_inference_delay = self.system_config.dnn.T_inference

        total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay + dnn_inference_delay

        energy = self._estimate_energy(edge_flops, cloud_flops, upload_delay, edge_delay)

        return total_delay, energy, True

    def _compute_cloud_only_metrics(
        self,
        task: Dict,
        cloud_resources: Dict
    ) -> Tuple[float, float, bool]:
        """计算纯云端执行指标"""
        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))

        f_cloud = cloud_resources.get('f_cloud', self.system_config.cloud.F_c)
        upload_rate = self.system_config.channel.R_backhaul
        upload_delay = data_size / upload_rate
        cloud_delay = total_flops / f_cloud if f_cloud > 0 else 0
        backhaul_delay = 0.01

        total_delay = upload_delay + cloud_delay + backhaul_delay
        energy = total_flops * 1e-12

        return total_delay, energy, True

    def _update_resource_usage(
        self,
        uav_id: int,
        split_layer: int,
        task: Dict,
        uav_resources: List[Dict]
    ):
        """更新资源使用"""
        if uav_id < 0:
            return

        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        total_layers = task.get('total_layers', 50)
        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5

        edge_flops = total_flops * split_ratio

        self.uav_compute_used[uav_id] = self.uav_compute_used.get(uav_id, 0) + edge_flops
        self.uav_task_count[uav_id] = self.uav_task_count.get(uav_id, 0) + 1

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
        """计算JFI"""
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


# 保留原始类名作为别名
LyapunovDRLBaseline = LyapunovDRLBaselineV2
