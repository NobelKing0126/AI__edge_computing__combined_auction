"""
MAPPO-Attention 基线算法 V2

论文: Joint Trajectory and Resource Optimization of MEC-Assisted UAVs in Sub-THz Networks:
      A Resources-Based Multi-Agent Proximal Policy Optimization DRL With Attention Mechanism

核心思想:
1. 多智能体近端策略优化 (MAPPO)
2. 注意力机制编码多智能体观测
3. 差异化奖励函数 (能耗+延迟+公平性)
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
        MAPPOAgent, MAPPOConfig, StateEncoder, ActionDecoder
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
class SubTHzChannelConfig:
    """Sub-THz信道配置"""
    frequency_min: float = 298e9
    frequency_max: float = 313e9
    center_frequency: float = 305.5e9

    mimo_config: Tuple[int, int] = (2, 2)
    los_mode: str = "dominant"

    bandwidth_uplink: float = 100e6
    bandwidth_downlink: float = 100e6
    num_subchannels: int = 10

    tx_power_uav: float = 1.0
    tx_power_user: float = 0.1

    molecular_absorption: bool = True
    rain_attenuation: bool = True
    coverage_range: float = 100.0


@dataclass
class TrainingConfig:
    """训练配置"""
    enable_training: bool = True
    pretrained_model_path: str = None
    save_model_path: str = "models/mappo_attention_model.pt"
    train_every: int = 32
    online_learning: bool = True

    reward_scale: float = 1.0
    delay_reward_weight: float = 0.4
    energy_reward_weight: float = 0.3
    fairness_reward_weight: float = 0.3
    success_reward: float = 1.0
    fail_penalty: float = -1.0


# ============ 差异化奖励计算器 ============

class DifferentiatedRewardCalculator:
    """
    差异化奖励计算器

    基于论文: 网络协调器为每个agent提供差异化奖励
    r = β₁ * U_energy + β₂ * U_delay + β₃ * U_fairness

    增强功能:
    - 支持用户收益集成
    - 支持价格风险溢价
    - 支持GAE优势估计
    """

    def __init__(self, training_config: TrainingConfig, system_config: SystemConfig = None):
        self.config = training_config
        self.system_config = system_config if system_config else SystemConfig()

        # 集成用户收益模型
        if NEW_MODULES_AVAILABLE:
            self.user_benefit_model = UserBenefitModel(
                UserBenefitConfig(
                    v0=self.system_config.user_benefit.v0,
                    beta_T=self.system_config.user_benefit.beta_T
                ),
                self.system_config
            )
        else:
            self.user_benefit_model = None

    def compute(
        self,
        energy_consumption: float,
        task_delay: float,
        fairness_index: float,
        success: bool,
        deadline: float
    ) -> float:
        """计算差异化奖励"""
        if success and task_delay <= deadline:
            # 成功且满足deadline
            u_energy = -self.config.energy_reward_weight * (energy_consumption / 1e6)
            u_delay = -self.config.delay_reward_weight * (task_delay / deadline)
            u_fairness = self.config.fairness_reward_weight * fairness_index
            success_bonus = self.config.success_reward

            reward = (u_energy + u_delay + u_fairness + success_bonus) * self.config.reward_scale
        else:
            # 失败或超时
            reward = self.config.fail_penalty * self.config.reward_scale

        return reward

    def compute_with_user_benefit(
        self,
        energy_consumption: float,
        task_delay: float,
        fairness_index: float,
        success: bool,
        deadline: float,
        priority: float,
        price: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算带用户收益的差异化奖励

        Args:
            energy_consumption: 能耗
            task_delay: 任务时延
            fairness_index: 公平性指数
            success: 是否成功
            deadline: 截止时间
            priority: 任务优先级
            price: 服务价格

        Returns:
            Tuple[float, Dict[str, float]]: (总奖励, 奖励分量)
        """
        # 基础奖励
        base_reward = self.compute(
            energy_consumption, task_delay, fairness_index, success, deadline
        )

        components = {
            'base_reward': base_reward,
            'energy_component': -self.config.energy_reward_weight * (energy_consumption / 1e6),
            'delay_component': -self.config.delay_reward_weight * (task_delay / deadline) if deadline > 0 else 0,
            'fairness_component': self.config.fairness_reward_weight * fairness_index
        }

        # 用户收益奖励（如果可用）
        if NEW_MODULES_AVAILABLE and self.user_benefit_model is not None:
            user_benefit = self.user_benefit_model.compute_user_benefit(
                priority, task_delay, price
            )
            # 将用户收益纳入奖励
            user_benefit_scaled = user_benefit / self.system_config.user_benefit.v0  # 归一化
            components['user_benefit'] = user_benefit_scaled

            total_reward = base_reward + 0.2 * user_benefit_scaled
        else:
            total_reward = base_reward
            components['user_benefit'] = 0.0

        components['total_reward'] = total_reward
        return total_reward, components

    def compute_fairness_index(self, utilizations: List[float]) -> float:
        """计算Jain's Fairness Index"""
        if not utilizations:
            return 1.0

        u = np.array(utilizations)
        sum_u = np.sum(u)
        sum_sq = np.sum(u ** 2)

        if sum_sq == 0:
            return 1.0

        jfi = (sum_u ** 2) / (len(u) * sum_sq)
        return jfi

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[List[float], List[float]]:
        """
        计算Generalized Advantage Estimation (GAE)

        基于论文: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Args:
            rewards: 奖励序列
            values: 价值估计序列
            next_value: 最后状态的下一价值
            gamma: 折扣因子
            gae_lambda: GAE平滑参数

        Returns:
            Tuple[List[float], List[float]]: (优势估计, 回报)
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0
        last_return = next_value

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD误差
            delta = rewards[t] + gamma * next_val - values[t]

            # GAE递推
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

            # 回报
            last_return = rewards[t] + gamma * last_return
            returns[t] = last_return

        return advantages.tolist(), returns.tolist()


# ============ MAPPO-Attention基线算法 V2 ============

class MAPPOAttentionBaselineV2(BaselineAlgorithm):
    """
    MAPPO-Attention 基线算法 V2

    实现论文核心思想:
    1. 多智能体协同决策 (通过注意力机制)
    2. 差异化奖励 (能耗+延迟+公平性)
    3. 完整的MAPPO训练过程

    兼容接口:
    - run(tasks, uav_resources, cloud_resources) -> BaselineResult
    """

    def __init__(
        self,
        mappo_config: 'MAPPOConfig' = None,
        channel_config: SubTHzChannelConfig = None,
        training_config: TrainingConfig = None,
        device: str = 'cpu'
    ):
        """
        初始化MAPPO-Attention基线V2

        Args:
            mappo_config: MAPPO配置
            channel_config: Sub-THz信道配置
            training_config: 训练配置
            device: 计算设备
        """
        super().__init__("MAPPO-Attention-V2")

        self.mappo_config = mappo_config if mappo_config else (
            MAPPOConfig() if TORCH_AVAILABLE else None
        )
        self.channel_config = channel_config if channel_config else SubTHzChannelConfig()
        self.training_config = training_config if training_config else TrainingConfig()
        self.system_config = SystemConfig()
        self.device = device

        # MAPPO智能体（延迟初始化）
        self.mappo_agent: Optional['MAPPOAgent'] = None
        self.state_encoder: Optional['StateEncoder'] = None
        self.action_decoder: Optional['ActionDecoder'] = None

        # 奖励计算器
        self.reward_calculator = DifferentiatedRewardCalculator(
            self.training_config, self.system_config
        )

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

        # 历史记录
        self.reward_history: List[float] = []
        self.fairness_history: List[float] = []
        self.training_losses: List[Dict[str, float]] = []
        self.total_tasks_processed = 0

        # 资源跟踪
        self.uav_compute_used: Dict[int, float] = {}
        self.uav_task_count: Dict[int, int] = {}

        # 加载预训练模型
        if (self.training_config.pretrained_model_path and
            TORCH_AVAILABLE and self.mappo_agent is not None):
            self._load_pretrained_model()

    def _initialize_agent(self, n_uavs: int):
        """初始化MAPPO智能体"""
        if not TORCH_AVAILABLE:
            return

        # 状态编码器
        self.state_encoder = StateEncoder(n_uavs, self.system_config)

        # 动作解码器
        self.action_decoder = ActionDecoder(n_uavs, n_split_options=5)

        # MAPPO智能体
        self.mappo_agent = MAPPOAgent(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.total_actions,
            config=self.mappo_config,
            device=self.device
        )

        # 初始化资源跟踪
        self._reset_tracking(n_uavs)

        # 加载预训练模型
        if self.training_config.pretrained_model_path:
            self._load_pretrained_model()

    def _reset_tracking(self, n_uavs: int):
        """重置资源跟踪"""
        self.uav_compute_used = {i: 0.0 for i in range(n_uavs)}
        self.uav_task_count = {i: 0 for i in range(n_uavs)}
        self.uav_compute_used = {}
        self.uav_task_count = {}

    def _load_pretrained_model(self):
        """加载预训练模型"""
        if self.mappo_agent and os.path.exists(self.training_config.pretrained_model_path):
            try:
                self.mappo_agent.load(self.training_config.pretrained_model_path)
                print(f"  [MAPPO-Attention-V2] 加载预训练模型: {self.training_config.pretrained_model_path}")
            except Exception as e:
                print(f"  [MAPPO-Attention-V2] 加载模型失败: {e}")

    def _save_model(self):
        """保存模型"""
        if self.mappo_agent and self.training_config.save_model_path:
            try:
                os.makedirs(os.path.dirname(self.training_config.save_model_path), exist_ok=True)
                self.mappo_agent.save(self.training_config.save_model_path)
            except Exception as e:
                print(f"  [MAPPO-Attention-V2] 保存模型失败: {e}")

    def _select_action_drl(
        self,
        task: Dict,
        uav_resources: List[Dict],
        uav_observations: List[np.ndarray],
        user_pos: Tuple[float, float],
        training: bool = True
    ) -> Tuple[int, int]:
        """使用DRL选择动作"""
        if not TORCH_AVAILABLE or self.mappo_agent is None:
            return self._select_action_heuristic(task, uav_resources, uav_observations, user_pos)

        # 编码状态
        state = self.state_encoder.encode(task, uav_resources)

        # 其他UAV的观测（用于注意力机制）
        other_states = [uav_observations[i] for i in range(len(uav_observations))]

        # 使用MAPPO选择动作
        deterministic = not training or (self.total_tasks_processed <
                                         self.mappo_config.batch_size)
        action, log_prob = self.mappo_agent.select_action(
            state, other_states, deterministic
        )

        # 解码动作
        uav_id, split_option = self.action_decoder.decode(action)

        # 转换split_option为split_layer
        split_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        split_ratio = split_ratios[split_option]
        total_layers = task.get('total_layers', 50)
        split_layer = int(split_ratio * total_layers)

        return uav_id, split_layer

    def _select_action_heuristic(
        self,
        task: Dict,
        uav_resources: List[Dict],
        uav_observations: List[np.ndarray],
        user_pos: Tuple[float, float]
    ) -> Tuple[int, int]:
        """启发式动作选择"""
        best_uav = None
        best_split = 0
        best_score = float('-inf')

        data_size = task.get('data_size', task.get('D', 1e6))
        total_flops = task.get('total_flops', task.get('C_total', 10e9))
        deadline = task.get('deadline', 2.0)

        for uav_id, (uav, obs) in enumerate(zip(uav_resources, uav_observations)):
            uav_pos = (uav.get('x', 100), uav.get('y', 100))
            dist = np.sqrt(
                (user_pos[0] - uav_pos[0])**2 +
                (user_pos[1] - uav_pos[1])**2
            )

            if dist > uav.get('R_cover', self.system_config.uav.R_cover) * 1.2:
                continue

            if uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) < 100:
                continue

            # 注意力权重
            att_weight = np.mean(obs) if len(obs) > 0 else 0.5

            for split in [0, 0.25, 0.5, 0.75, 1.0]:
                edge_flops = total_flops * split
                cloud_flops = total_flops * (1 - split)

                upload_rate = self._compute_subthz_upload_rate(user_pos, uav_pos)
                upload_delay = data_size / upload_rate if upload_rate > 0 else float('inf')

                f_edge = uav.get('f_max', self.system_config.uav.f_max)
                edge_delay = edge_flops / f_edge if f_edge > 0 else 0

                cloud_delay = self._compute_cloud_delay(cloud_flops)
                backhaul_delay = self._compute_subthz_backhaul_delay(cloud_flops)

                total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay

                if total_delay > deadline:
                    continue

                energy = self._estimate_energy_subthz(
                    edge_flops, cloud_flops, upload_delay, edge_delay, dist
                )

                # 计算公平性
                utilizations = [
                    self.uav_compute_used.get(i, 0) /
                    uav_resources[i].get('f_max', self.system_config.uav.f_max)
                    for i in range(len(uav_resources))
                ]
                fairness = self.reward_calculator.compute_fairness_index(utilizations)

                # 计算奖励
                reward = self.reward_calculator.compute(
                    energy, total_delay, fairness, True, deadline
                )

                score = reward + att_weight * 0.1

                if score > best_score:
                    best_score = score
                    best_uav = uav_id
                    best_split = int(split * task.get('total_layers', 50))

        return best_uav, best_split

    def run(
        self,
        tasks: List[Dict],
        uav_resources: List[Dict],
        cloud_resources: Dict
    ) -> BaselineResult:
        """
        运行MAPPO-Attention算法V2

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

        # 初始化MAPPO智能体
        if TORCH_AVAILABLE and self.mappo_agent is None:
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

        # 按优先级排序任务
        sorted_tasks = sorted(
            enumerate(tasks),
            key=lambda x: x[1].get('priority', 0.5),
            reverse=True
        )

        # 构建UAV观测向量
        uav_observations = self._build_uav_observations(uav_resources)

        # 存储转移用于训练
        last_state = None
        last_action = None
        last_value = None
        last_log_prob = None

        # 处理每个任务
        for task_idx, (original_idx, task) in enumerate(sorted_tasks):
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

            # 获取当前价值估计
            if TORCH_AVAILABLE and self.mappo_agent:
                current_value = self.mappo_agent.get_value(current_state)
            else:
                current_value = 0.0

            # 使用DRL或启发式选择动作
            use_drl = (TORCH_AVAILABLE and
                      self.mappo_agent is not None and
                      self.training_config.enable_training)

            if use_drl:
                best_uav, best_split = self._select_action_drl(
                    task, uav_resources, uav_observations, user_pos, training=True
                )
            else:
                best_uav, best_split = self._select_action_heuristic(
                    task, uav_resources, uav_observations, user_pos
                )

            if best_uav is None:
                continue

            # 计算任务指标
            delay, energy, success = self._compute_task_metrics_subthz(
                task, best_uav, best_split, uav_resources, cloud_resources
            )

            # 时延约束 (统一使用标准deadline判定，与其他baseline对齐)
            task_success = success and delay <= deadline

            if task_success:
                success_tasks.append(original_idx)
                delays.append(delay)
                energies.append(energy)

                # 计算公平性
                utilizations = [
                    self.uav_compute_used.get(i, 0) /
                    uav_resources[i].get('f_max', self.system_config.uav.f_max)
                    for i in range(n_uavs)
                ]
                fairness = self.reward_calculator.compute_fairness_index(utilizations)

                # 计算奖励
                reward = self.reward_calculator.compute(
                    energy, delay, fairness, True, deadline
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

            # MAPPO训练
            if (TORCH_AVAILABLE and
                self.mappo_agent is not None and
                self.training_config.enable_training and
                self.training_config.online_learning):

                # 计算奖励
                reward = self.reward_calculator.compute(
                    energy, delay, 1.0, task_success, deadline
                )

                # 获取下一个状态
                next_state = self.state_encoder.encode(task, uav_resources) if self.state_encoder else None

                # 存储转移
                if last_state is not None and last_action is not None:
                    done = (task_idx == len(sorted_tasks) - 1) or not task_success
                    self.mappo_agent.store_transition(
                        last_state, last_action, reward, last_value, last_log_prob, done
                    )

                # 训练
                if (len(self.mappo_agent.rollout_buffer) >= self.mappo_config.batch_size and
                    self.total_tasks_processed % self.training_config.train_every == 0):

                    last_value = self.mappo_agent.get_value(current_state) if current_state is not None else 0.0
                    losses = self.mappo_agent.update(last_value)
                    if losses:
                        self.training_losses.append(losses)

                # 保存当前状态和动作
                last_state = current_state
                if current_state is not None:
                    split_ratio = best_split / task.get('total_layers', 50)
                    split_options = [0, 0.25, 0.5, 0.75, 1.0]
                    split_option = min(range(len(split_options)),
                                      key=lambda i: abs(split_options[i] - split_ratio))
                    last_action = self.action_decoder.encode(best_uav, split_option)
                    last_value = current_value
                    # log_prob 暂不使用
                    last_log_prob = 0.0

            self.total_tasks_processed += 1

        # 保存模型
        if self.training_config.save_model_path and self.mappo_agent:
            self._save_model()

        # 计算结果
        success_rate = len(success_tasks) / n_tasks if n_tasks > 0 else 0.0
        avg_delay = np.mean(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0
        total_energy = sum(energies)
        avg_energy = np.mean(energies) if energies else 0.0

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

    def _build_uav_observations(self, uav_resources: List[Dict]) -> List[np.ndarray]:
        """构建UAV观测向量"""
        observations = []

        for uav in uav_resources:
            obs = np.array([
                uav.get('x', 100) / 500.0,
                uav.get('y', 100) / 500.0,
                uav.get('E_current', uav.get('E_max', self.system_config.uav.E_max)) /
                    max(uav.get('E_max', self.system_config.uav.E_max), 1),
                uav.get('utilization', 0.0),
                uav.get('price', 1.0) / 2.0
            ])
            observations.append(obs)

        return observations

    def _compute_subthz_upload_rate(
        self,
        user_pos: Tuple[float, float],
        uav_pos: Tuple[float, float]
    ) -> float:
        """计算Sub-THz上行速率"""
        H = self.system_config.uav.H
        dist = np.sqrt(
            (user_pos[0] - uav_pos[0])**2 +
            (user_pos[1] - uav_pos[1])**2 +
            H**2
        )

        f_c = self.channel_config.center_frequency
        c = 3e8

        fspl = 20 * np.log10(4 * np.pi * dist * f_c / c + 1e-10)
        molecular_loss = 0.1 * dist / 100

        total_loss = fspl + molecular_loss

        tx_power = self.channel_config.tx_power_user
        rx_power = tx_power * 10 ** (-total_loss / 10)

        k_boltzmann = 1.38e-23
        T = 290
        bandwidth = self.channel_config.bandwidth_uplink
        noise_power = k_boltzmann * T * bandwidth

        snr = rx_power / noise_power
        rate = bandwidth * np.log2(1 + snr)

        return max(rate, 1e6)

    def _compute_cloud_delay(self, cloud_flops: float) -> float:
        """计算云端延迟"""
        if cloud_flops <= 0:
            return 0.0
        f_cloud = self.system_config.cloud.F_c
        T_prop = self.system_config.cloud.T_propagation
        compute_delay = cloud_flops / f_cloud
        return compute_delay + T_prop

    def _compute_subthz_backhaul_delay(self, cloud_flops: float) -> float:
        """计算Sub-THz回程延迟"""
        if cloud_flops <= 0:
            return 0.0

        backhaul_rate = self.channel_config.bandwidth_downlink * 10
        propagation_delay = 0.01
        tx_delay = cloud_flops / (backhaul_rate * 1e6)

        return propagation_delay + tx_delay

    def _estimate_energy_subthz(
        self,
        edge_flops: float,
        cloud_flops: float,
        upload_delay: float,
        edge_delay: float,
        distance: float
    ) -> float:
        """估算Sub-THz任务能耗"""
        kappa_edge = self.system_config.energy.kappa_edge
        f_edge = self.system_config.uav.f_max
        edge_energy = kappa_edge * (f_edge ** 2) * edge_flops if edge_flops > 0 else 0

        tx_power = self.channel_config.tx_power_user
        molecular_factor = 1.0 + 0.1 * distance / 100
        tx_energy = tx_power * upload_delay * molecular_factor

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
        """计算Sub-THz任务的时延和能耗"""
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

        split_ratio = split_layer / total_layers if total_layers > 0 else 0.5
        edge_flops = total_flops * split_ratio
        cloud_flops = total_flops * (1 - split_ratio)

        upload_rate = self._compute_subthz_upload_rate(user_pos, uav_pos)
        if upload_rate <= 0:
            return float('inf'), 0, False

        upload_delay = data_size / upload_rate

        f_edge = uav.get('f_max', self.system_config.uav.f_max)
        edge_delay = edge_flops / f_edge if f_edge > 0 and edge_flops > 0 else 0

        cloud_delay = self._compute_cloud_delay(cloud_flops)
        backhaul_delay = self._compute_subthz_backhaul_delay(cloud_flops)

        # 添加DNN推理时延
        dnn_inference_delay = self.system_config.dnn.T_inference

        total_delay = upload_delay + edge_delay + cloud_delay + backhaul_delay + dnn_inference_delay

        energy = self._estimate_energy_subthz(
            edge_flops, cloud_flops, upload_delay, edge_delay, dist
        )

        return total_delay, energy, True

    def _update_resource_usage(
        self,
        uav_id: int,
        split_layer: int,
        task: Dict,
        uav_resources: List[Dict]
    ):
        """更新资源使用状态"""
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


# 保留原始类名作为别名
MAPPOAttentionBaseline = MAPPOAttentionBaselineV2
