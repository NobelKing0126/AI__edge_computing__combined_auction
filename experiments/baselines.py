"""
基线算法实现

用于与提议的拍卖调度框架对比

基线算法:
1. Edge-Only: 全边缘计算
2. Cloud-Only: 全云端计算
3. Greedy: 贪心调度
4. FCFS: 先来先服务
5. Fixed-Split: 固定切分点
6. Random-Auction: 随机拍卖
7. No-ActiveInference: 无主动推理
8. Heuristic-Alloc: 启发式分配
9. No-DynPricing: 无动态定价
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.system_config import SystemConfig, ExecutionConfig
from config.constants import (
    NUMERICAL, FREE_ENERGY, PRICING, CONSTRAINT, COMMUNICATION, RESOURCE
)


@dataclass
class BaselineResult:
    """
    完整基线算法结果 (按照实验.txt 4.1-4.4节定义)
    
    Attributes:
        === 基本信息 ===
        name: 算法名称
        total_tasks: 总任务数
        success_count: 成功任务数
        
        === 4.1 主要指标 ===
        success_rate: 任务完成率
        high_priority_rate: 高优先级任务成功率
        social_welfare: 社会福利 (SW = Σ η_final)
        avg_delay: 平均端到端时延
        max_delay: 最大时延
        deadline_meet_rate: 时延满足率
        total_energy: 系统总能耗
        avg_energy: 平均每任务能耗
        energy_efficiency: 能效比 (completed/energy)
        
        === 4.2 资源利用指标 ===
        avg_uav_utilization: UAV平均算力利用率
        jfi_load_balance: JFI负载均衡指数
        cloud_utilization: 云端利用率
        channel_utilization: 信道利用率
        
        === 4.3 鲁棒性指标 ===
        fault_recovery_rate: 故障恢复成功率
        avg_recovery_delay: 平均恢复时延
        checkpoint_success_rate: Checkpoint成功率
        recovery_delay_saving: 恢复时延节省比
        
        === 4.4 算法效率指标 ===
        bidding_time_ms: 投标生成时间 (ms)
        auction_time_ms: 拍卖决策时间 (ms)
        dual_iterations: 对偶迭代次数
        duality_gap: 对偶间隙
        
        === UAV详细信息 ===
        uav_utilizations: 各UAV利用率
        uav_loads: 各UAV任务数
    """
    # 基本信息
    name: str
    total_tasks: int
    success_count: int
    
    # 4.1 主要指标
    success_rate: float
    avg_delay: float
    max_delay: float
    deadline_meet_rate: float
    total_energy: float
    avg_energy: float
    high_priority_rate: float
    social_welfare: float = 0.0
    energy_efficiency: float = 0.0
    
    # 4.2 资源利用指标
    avg_uav_utilization: float = 0.0
    jfi_load_balance: float = 1.0
    cloud_utilization: float = 0.0
    channel_utilization: float = 0.0
    
    # 4.3 鲁棒性指标
    fault_recovery_rate: float = 1.0
    avg_recovery_delay: float = 0.0
    checkpoint_success_rate: float = 1.0
    recovery_delay_saving: float = 0.0
    
    # 4.4 算法效率指标
    bidding_time_ms: float = 0.0
    auction_time_ms: float = 0.0
    dual_iterations: int = 0
    duality_gap: float = 0.0
    
    # UAV详细信息
    uav_utilizations: List[float] = None
    uav_loads: List[int] = None
    
    # 新增: 用户收益指标
    user_payoff_total: float = 0.0        # 总用户收益
    user_payoff_avg: float = 0.0          # 平均用户收益
    user_payoff_gini: float = 0.0         # 收益基尼系数
    payoff_high_priority: float = 0.0     # 高优先级用户收益
    payoff_medium_priority: float = 0.0   # 中优先级用户收益
    payoff_low_priority: float = 0.0      # 低优先级用户收益
    
    # 新增: 服务提供商利润指标
    provider_revenue: float = 0.0         # 总收入
    provider_cost: float = 0.0            # 运营成本
    provider_profit: float = 0.0          # 净利润
    provider_profit_margin: float = 0.0   # 利润率
    
    # 新增: 竞争比指标
    competitive_ratio: float = 1.0        # 竞争比 = SW* / SW_online
    sw_optimal: float = 0.0               # 离线最优社会福利 SW*
    
    def __post_init__(self):
        if self.uav_utilizations is None:
            self.uav_utilizations = []
        if self.uav_loads is None:
            self.uav_loads = []


class BaselineAlgorithm:
    """基线算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = SystemConfig()
        self.kappa_edge = self.config.energy.kappa_edge
        
        # 资源跟踪
        self.uav_compute_used = {}
        self.uav_task_count = {}
        self.cloud_compute_used = 0.0
        self.channels_used = 0
        
        # 鲁棒性跟踪
        self.fault_count = 0
        self.recovery_count = 0
        self.checkpoint_attempts = 0
        self.checkpoint_successes = 0
        self.recovery_delays = []
        
        # 算法时间跟踪
        self.bidding_time = 0.0
        self.auction_time = 0.0
    
    def _get_feature_size_at_layer(self, split_layer: int, model_spec, n_layers: int) -> float:
        """
        获取在指定切分层的中间特征大小（使用精确的每层数据）
        """
        if split_layer == 0:
            return model_spec.input_size_bytes if hasattr(model_spec, 'input_size_bytes') else 150000
        elif split_layer >= n_layers:
            return 1000
        
        if model_spec and hasattr(model_spec, 'get_output_size_at_layer'):
            return model_spec.get_output_size_at_layer(split_layer - 1)
        
        # 回退
        if hasattr(model_spec, 'typical_feature_sizes') and model_spec.typical_feature_sizes:
            idx = int(split_layer / n_layers * len(model_spec.typical_feature_sizes))
            idx = min(idx, len(model_spec.typical_feature_sizes) - 1)
            return model_spec.typical_feature_sizes[idx]
        return 150000 * (1 - split_layer / n_layers) * 0.5
    
    def _get_cumulative_flops_at_layer(self, split_layer: int, model_spec, n_layers: int, C_total: float) -> float:
        """
        获取前split_layer层的累计计算量（使用精确的每层数据）
        """
        if split_layer <= 0:
            return 0.0
        if split_layer >= n_layers:
            return C_total
        
        if model_spec and hasattr(model_spec, 'get_flops_ratio_at_layer'):
            return C_total * model_spec.get_flops_ratio_at_layer(split_layer)
        
        return C_total * (split_layer / n_layers)
    
    def _is_uav_covering_user(self, user_pos: Tuple[float, float], 
                               uav_pos: Tuple[float, float]) -> bool:
        """判断UAV是否覆盖用户"""
        distance = np.sqrt((user_pos[0] - uav_pos[0])**2 + (user_pos[1] - uav_pos[1])**2)
        return distance <= self.config.uav.R_cover
    
    def _reset_tracking(self, n_uavs: int = 5):
        """重置资源跟踪"""
        self.uav_compute_used = {i: 0.0 for i in range(n_uavs)}
        self.uav_task_count = {i: 0 for i in range(n_uavs)}
        self.cloud_compute_used = 0.0
        self.channels_used = 0
        self.fault_count = 0
        self.recovery_count = 0
        self.checkpoint_attempts = 0
        self.checkpoint_successes = 0
        self.recovery_delays = []
        self.bidding_time = 0.0
        self.auction_time = 0.0
    
    def _compute_upload_rate(self, user_pos: Tuple[float, float], 
                             uav_pos: Tuple[float, float]) -> float:
        """
        计算用户到UAV的上传速率
        
        使用真实信道模型: R = W * log2(1 + P*h/N0W)
        其中 h = beta_0 / d^2
        """
        # 计算3D距离 (考虑UAV高度)
        H = self.config.uav.H
        dx = user_pos[0] - uav_pos[0]
        dy = user_pos[1] - uav_pos[1]
        dist = np.sqrt(dx**2 + dy**2 + H**2)
        
        # 信道增益
        beta_0 = self.config.channel.beta_0
        h = beta_0 / (dist ** 2)
        
        # 传输速率
        P_tx = self.config.channel.P_tx_user
        N_0 = self.config.channel.N_0
        W = self.config.channel.W
        
        snr = P_tx * h / (N_0 * W)
        rate = W * np.log2(1 + snr)
        
        return rate
    
    def _compute_backhaul_rate(self) -> float:
        """获取回程链路速率"""
        return self.config.channel.R_backhaul
    
    def _compute_cloud_delay(self, C_cloud: float, n_concurrent: int = 1) -> float:
        """
        计算云端计算时延（考虑资源竞争）
        
        云端资源竞争模型：
        - 云端总算力 F_c 被所有并发任务共享
        - 每个任务分配的算力 = F_c / n_concurrent
        - 这反映了真实云端多租户场景
        
        Args:
            C_cloud: 云端计算量 (FLOPs)
            n_concurrent: 当前并发任务数
            
        Returns:
            float: 云端计算时延 (s)
        """
        if C_cloud <= 0:
            return 0.0
        
        F_c = self.config.cloud.F_c
        F_per_task_max = self.config.cloud.F_per_task_max  # 单任务最大算力限制
        max_concurrent = self.config.cloud.max_concurrent_tasks
        
        # 有效并发数：至少1，最多max_concurrent
        effective_concurrent = max(1, min(n_concurrent, max_concurrent))
        
        # 每个任务分配的算力 = min(总算力/并发数, 单任务上限)
        f_per_task = min(F_c / effective_concurrent, F_per_task_max)
        
        # 云端计算时延
        T_cloud = C_cloud / f_per_task
        
        return T_cloud
    
    def _get_propagation_delay(self) -> float:
        """
        获取UAV到云端的网络传播延迟
        
        传播延迟包含：
        - 光纤传播延迟（~5ms/1000km）
        - 路由器处理延迟（~2ms/跳）
        - 边缘网关延迟（~5ms）
        
        Returns:
            float: 单向传播延迟 (s)
        """
        return self.config.cloud.T_propagation
    
    def _compute_cloud_path_delay(self, data_size: float, C_cloud: float, 
                                   n_concurrent: int = 1) -> Tuple[float, float, float]:
        """
        计算完整的云端路径时延
        
        云端路径 = UAV→云端传输 + 传播延迟 + 云端计算 + 传播延迟 + 返回传输
        
        Args:
            data_size: 传输数据量 (bits)
            C_cloud: 云端计算量 (FLOPs)
            n_concurrent: 当前并发任务数
            
        Returns:
            Tuple[T_trans, T_propagation_total, T_cloud]:
                - T_trans: 传输时延（上传+下载）
                - T_propagation_total: 往返传播延迟
                - T_cloud: 云端计算时延
        """
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # UAV到云端传输
        T_trans_up = data_size / R_backhaul
        
        # 结果返回传输（假设结果数据量为原始的1%）
        result_size = data_size * 0.01
        T_trans_down = result_size / R_backhaul
        
        # 总传输时延
        T_trans = T_trans_up + T_trans_down
        
        # 往返传播延迟（去程+回程）
        T_propagation_total = 2 * T_propagation
        
        # 云端计算时延（考虑资源竞争）
        T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
        
        return T_trans, T_propagation_total, T_cloud
    
    def run(self, tasks: List[Dict], 
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        """运行基线算法"""
        raise NotImplementedError
    
    def _track_task_result(self, result: Dict, uav_id: int, 
                           compute_edge: float = 0, compute_cloud: float = 0,
                           fault_occurred: bool = False, recovered: bool = False,
                           recovery_delay: float = 0, checkpoint_used: bool = False):
        """跟踪单个任务结果"""
        if result.get('success', False):
            self.uav_compute_used[uav_id] = self.uav_compute_used.get(uav_id, 0) + compute_edge
            self.uav_task_count[uav_id] = self.uav_task_count.get(uav_id, 0) + 1
            self.cloud_compute_used += compute_cloud
            self.channels_used += 1
        
        if fault_occurred:
            self.fault_count += 1
            if recovered:
                self.recovery_count += 1
                self.recovery_delays.append(recovery_delay)
        
        if checkpoint_used:
            self.checkpoint_attempts += 1
            if recovered or result.get('success', False):
                self.checkpoint_successes += 1
    
    def _compute_jfi(self, n_uavs: int) -> float:
        """计算JFI负载均衡指数"""
        utils = []
        for i in range(n_uavs):
            max_compute = self.config.uav.f_max
            used = self.uav_compute_used.get(i, 0)
            util = used / max_compute if max_compute > 0 else 0
            utils.append(util)
        
        if not utils or sum(utils) == 0:
            return 1.0
        
        sum_util = sum(utils)
        sum_util_sq = sum(u**2 for u in utils)
        
        if sum_util_sq == 0:
            return 1.0
        
        return (sum_util ** 2) / (len(utils) * sum_util_sq)
    
    def _compute_result(self, 
                        tasks: List[Dict],
                        task_results: List[Dict],
                        uav_resources: List[Dict] = None,
                        cloud_resources: Dict = None) -> BaselineResult:
        """
        计算完整结果统计 (按照实验.txt 4.1-4.4)
        """
        n = len(task_results)
        n_uavs = len(uav_resources) if uav_resources else 5
        
        if n == 0:
            return BaselineResult(
                name=self.name, total_tasks=0, success_count=0,
                success_rate=0, avg_delay=0, max_delay=0,
                deadline_meet_rate=0, total_energy=0, avg_energy=0,
                high_priority_rate=0
            )
        
        # 基本统计
        success = [r for r in task_results if r.get('success', False)]
        met_deadline = [r for r in task_results if r.get('met_deadline', False)]
        # 只计算成功任务的时延
        delays = [r.get('delay', 0) for r in task_results if r.get('success', False)]
        energies = [r.get('energy', 0) for r in task_results]
        utilities = [r.get('utility', 0) for r in task_results]
        
        # 高优先级任务成功率
        high_priority = [t for t in tasks if t.get('priority', 0.5) >= 0.7]
        high_success = sum(1 for i, r in enumerate(task_results) 
                          if i < len(tasks) and tasks[i].get('priority', 0.5) >= 0.7 
                          and r.get('success', False))
        high_priority_rate = high_success / len(high_priority) if high_priority else 1.0
        
        total_energy = sum(energies)
        success_count = len(success)
        
        # 4.1 社会福利
        social_welfare = sum(u for r, u in zip(task_results, utilities) if r.get('success', False))
        if social_welfare == 0:
            social_welfare = success_count  # 简化：成功任务数作为福利
        
        # 能效比
        energy_efficiency = success_count / max(total_energy, NUMERICAL.EPSILON)
        
        # 4.2 资源利用率
        uav_utils = []
        for i in range(n_uavs):
            max_compute = self.config.uav.f_max
            used = self.uav_compute_used.get(i, 0)
            util = min(used / max_compute, 1.0) if max_compute > 0 else 0
            uav_utils.append(util)
        
        avg_uav_util = np.mean(uav_utils) if uav_utils else 0
        jfi = self._compute_jfi(n_uavs)
        
        cloud_max = cloud_resources.get('f_cloud', self.config.cloud.F_c) if cloud_resources else self.config.cloud.F_c
        cloud_util = min(self.cloud_compute_used / cloud_max, 1.0) if cloud_max > 0 else 0
        
        channel_util = min(self.channels_used / self.config.channel.num_channels, 1.0)
        
        # 4.3 鲁棒性指标
        fault_recovery_rate = self.recovery_count / self.fault_count if self.fault_count > 0 else 1.0
        avg_recovery_delay = np.mean(self.recovery_delays) if self.recovery_delays else 0
        checkpoint_success = self.checkpoint_successes / self.checkpoint_attempts if self.checkpoint_attempts > 0 else 1.0
        
        # UAV负载
        uav_loads = [self.uav_task_count.get(i, 0) for i in range(n_uavs)]
        
        # 4.5 用户收益指标 - 基于成功任务计算
        # 用户收益 = 任务价值 - 支付价格 (对于基线算法，简化为效用的比例)
        payoffs = []
        payoff_high, payoff_med, payoff_low = 0.0, 0.0, 0.0
        for i, r in enumerate(task_results):
            if r.get('success', False) and i < len(tasks):
                task = tasks[i]
                priority = task.get('priority', 0.5)
                utility = r.get('utility', 1.0)
                delay = r.get('delay', 0.5)
                deadline = task.get('deadline', 1.0)
                
                # 用户收益 = 效用 * 时延节省比例
                time_saved_ratio = max(0, 1 - delay / deadline)
                payoff = utility * time_saved_ratio * priority
                payoffs.append(payoff)
                
                if priority >= 0.7:
                    payoff_high += payoff
                elif priority >= 0.4:
                    payoff_med += payoff
                else:
                    payoff_low += payoff
        
        user_payoff_total = sum(payoffs)
        user_payoff_avg = np.mean(payoffs) if payoffs else 0
        
        # 基尼系数计算
        if len(payoffs) > 1 and sum(payoffs) > 0:
            sorted_payoffs = np.sort(payoffs)
            n_p = len(payoffs)
            cumsum = np.cumsum(sorted_payoffs)
            gini = (n_p + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n_p
            user_payoff_gini = max(0, min(1, gini))
        else:
            user_payoff_gini = 0.0
        
        # 4.6 服务提供商利润指标
        # 收入 = 成功任务数 * 平均价格 (基于资源使用量)
        avg_compute_per_task = (sum(self.uav_compute_used.values()) + self.cloud_compute_used) / max(n, 1)
        compute_price = 1e-9  # 每FLOP价格
        provider_revenue = success_count * avg_compute_per_task * compute_price
        
        # 成本 = 能耗成本 + 运营固定成本
        energy_cost = total_energy * 0.01  # 每焦耳0.01
        fixed_cost = n_uavs * 0.1  # 每UAV固定成本
        provider_cost = energy_cost + fixed_cost
        
        provider_profit = provider_revenue - provider_cost
        provider_profit_margin = (provider_profit / provider_revenue * 100) if provider_revenue > 0 else 0.0
        
        return BaselineResult(
            name=self.name,
            total_tasks=n,
            success_count=success_count,
            
            # 4.1 主要指标
            success_rate=success_count / n,
            avg_delay=np.mean(delays) if delays else 0,
            max_delay=max(delays) if delays else 0,
            deadline_meet_rate=len(met_deadline) / n,
            total_energy=total_energy,
            avg_energy=np.mean(energies) if energies else 0,
            high_priority_rate=high_priority_rate,
            social_welfare=social_welfare,
            energy_efficiency=energy_efficiency,
            
            # 4.2 资源利用指标
            avg_uav_utilization=avg_uav_util,
            jfi_load_balance=jfi,
            cloud_utilization=cloud_util,
            channel_utilization=channel_util,
            
            # 4.3 鲁棒性指标
            fault_recovery_rate=fault_recovery_rate,
            avg_recovery_delay=avg_recovery_delay,
            checkpoint_success_rate=checkpoint_success,
            recovery_delay_saving=0.0,  # 需要与无checkpoint对比
            
            # 4.4 算法效率指标
            bidding_time_ms=self.bidding_time * 1000,
            auction_time_ms=self.auction_time * 1000,
            dual_iterations=0,
            duality_gap=0.0,
            
            # 4.5 用户收益指标
            user_payoff_total=user_payoff_total,
            user_payoff_avg=user_payoff_avg,
            user_payoff_gini=user_payoff_gini,
            payoff_high_priority=payoff_high,
            payoff_medium_priority=payoff_med,
            payoff_low_priority=payoff_low,
            
            # 4.6 服务提供商利润指标
            provider_revenue=provider_revenue,
            provider_cost=provider_cost,
            provider_profit=provider_profit,
            provider_profit_margin=provider_profit_margin,
            
            # UAV详细信息
            uav_utilizations=uav_utils,
            uav_loads=uav_loads
        )


class EdgeOnlyBaseline(BaselineAlgorithm):
    """
    全边缘计算基线
    
    所有DNN层在UAV边缘执行，不卸载到云端
    """
    
    def __init__(self):
        super().__init__("Edge-Only")
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        self._reset_tracking(n_uavs)
        
        # 获取或设置UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                # 默认均匀分布
                uav_positions.append((400 + i * 300, 1000))
        
        # 均匀分配任务到UAV
        task_uav_map = {}
        for i, task in enumerate(tasks):
            uav_id = i % n_uavs
            task_uav_map[i] = uav_id
        
        for i, task in enumerate(tasks):
            uav_id = task_uav_map[i]
            f_max = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            uav_pos = uav_positions[uav_id]
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            priority = task.get('priority', 0.5)
            
            # 获取用户位置
            user_pos = task.get('user_pos', (1000, 1000))
            
            # 使用真实信道模型计算上传速率
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            T_upload = data_size / upload_rate
            
            # 边缘计算时延 (全部在边缘)
            T_compute = C_total / f_max
            
            T_total = T_upload + T_compute
            
            # 能耗
            energy = self.kappa_edge * (f_max ** 2) * C_total
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用
            self._track_task_result(result, uav_id, compute_edge=C_total, compute_cloud=0)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class CloudOnlyBaseline(BaselineAlgorithm):
    """
    全云端计算基线
    
    所有DNN层卸载到云端执行
    
    时延模型（考虑资源竞争和传播延迟）：
    T_total = T_upload + T_trans + T_propagation + T_cloud + T_propagation + T_return
    
    其中：
    - T_upload: 用户到UAV上传时延
    - T_trans: UAV到云端传输时延
    - T_propagation: 网络传播延迟（往返）
    - T_cloud: 云端计算时延（受资源竞争影响）
    - T_return: 结果返回时延
    """
    
    def __init__(self):
        super().__init__("Cloud-Only")
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 云端资源竞争：估算并发任务数
        # 全云端基线假设所有任务同时到达，争抢云端资源
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for i, task in enumerate(tasks):
            uav_id = i % n_uavs
            uav_pos = uav_positions[uav_id]
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 用户到UAV上传
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            T_upload = data_size / upload_rate
            
            # 云端路径时延（包含资源竞争和传播延迟）
            T_trans, T_propagation_total, T_cloud = self._compute_cloud_path_delay(
                data_size, C_total, n_concurrent
            )
            
            # 总时延 = 上传 + 云端路径
            T_total = T_upload + T_trans + T_propagation_total + T_cloud
            
            # UAV中继能耗（接收+转发，无计算）
            P_rx = self.config.uav.P_rx  # 接收功率
            P_tx = self.config.uav.P_tx  # 发射功率
            T_trans_up = data_size / R_backhaul
            T_return = data_size * 0.01 / R_backhaul
            E_rx = P_rx * T_upload       # 接收用户数据
            E_tx = P_tx * T_trans_up     # 转发到云端
            E_download = P_tx * T_return # 返回结果给用户
            energy = E_rx + E_tx + E_download
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用 (全部云端)
            self._track_task_result(result, uav_id, compute_edge=0, compute_cloud=C_total)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class GreedyBaseline(BaselineAlgorithm):
    """
    贪心调度基线
    
    按任务顺序贪心分配资源
    """
    
    def __init__(self):
        super().__init__("Greedy")
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        self._reset_tracking(n_uavs)
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 跟踪剩余资源
        remaining_f = {i: r.get('f_max', self.config.uav.f_max) for i, r in enumerate(uav_resources)}
        remaining_E = {i: r.get('E_max', self.config.uav.E_max) for i, r in enumerate(uav_resources)}
        
        for i, task in enumerate(tasks):
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 贪心选择第一个可用UAV
            assigned = False
            assigned_uav = -1
            
            for uav_id in range(n_uavs):
                uav_pos = uav_positions[uav_id]
                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                T_upload = data_size / upload_rate
                
                # 需要的算力
                remaining_time = deadline - T_upload
                if remaining_time <= 0:
                    continue
                    
                f_need = C_total / remaining_time
                f_need = max(f_need, 1e9)  # 最小算力
                
                E_need = self.kappa_edge * (f_need ** 2) * C_total
                
                if remaining_f[uav_id] >= f_need and remaining_E[uav_id] >= E_need:
                    # 分配
                    remaining_f[uav_id] -= f_need
                    remaining_E[uav_id] -= E_need
                    
                    T_compute = C_total / f_need
                    T_total = T_upload + T_compute
                    
                    result = {
                        'task_id': i,
                        'success': True,
                        'met_deadline': T_total <= deadline,
                        'delay': T_total,
                        'energy': E_need,
                        'uav_id': uav_id,
                        'utility': 1.0,
                        'priority': priority
                    }
                    results.append(result)
                    assigned = True
                    assigned_uav = uav_id
                    
                    self._track_task_result(result, uav_id, compute_edge=C_total, compute_cloud=0)
                    break
            
            if not assigned:
                result = {
                    'task_id': i,
                    'success': False,
                    'met_deadline': False,
                    'delay': 999.0,
                    'energy': 0,
                    'uav_id': -1,
                    'utility': 0.0,
                    'priority': priority
                }
                results.append(result)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class FixedSplitBaseline(BaselineAlgorithm):
    """
    固定切分点基线
    
    使用固定50%切分点进行边缘-云端协同
    
    时延模型（考虑资源竞争和传播延迟）：
    T_total = T_upload + T_edge + T_trans + T_propagation + T_cloud + T_propagation + T_return
    """
    
    def __init__(self, split_ratio: float = 0.5):
        super().__init__("Fixed-Split")
        self.split_ratio = split_ratio
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 云端资源竞争：估算并发任务数（部分任务使用云端）
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for i, task in enumerate(tasks):
            uav_id = i % n_uavs
            f_edge = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            uav_pos = uav_positions[uav_id]
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 固定切分
            C_edge = C_total * self.split_ratio
            C_cloud = C_total * (1 - self.split_ratio)
            
            # 用户上传
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            T_upload = data_size / upload_rate
            
            # 边缘计算
            T_edge = C_edge / f_edge
            
            # 中间特征传输 (假设特征大小与切分比例相关)
            feature_size = data_size * self.split_ratio * 0.5
            T_trans = feature_size / R_backhaul
            
            # 云端计算（考虑资源竞争）
            T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
            
            # 传播延迟（往返）
            T_propagation_total = 2 * T_propagation
            
            # 结果返回
            T_return = data_size * 0.01 / R_backhaul
            
            T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud + T_return
            
            # 能耗（边缘计算 + 通信）
            P_rx = self.config.uav.P_rx
            P_tx = self.config.uav.P_tx
            E_compute = self.kappa_edge * (f_edge ** 2) * C_edge
            E_comm = P_rx * T_upload + P_tx * (T_trans + T_return)
            energy = E_compute + E_comm
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用
            self._track_task_result(result, uav_id, compute_edge=C_edge, compute_cloud=C_cloud)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class RandomAuctionBaseline(BaselineAlgorithm):
    """
    随机拍卖基线
    
    随机选择投标中标
    
    时延模型（考虑资源竞争和传播延迟）
    """
    
    def __init__(self, seed: int = 42):
        super().__init__("Random-Auction")
        self.rng = np.random.default_rng(seed)
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 云端资源竞争
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        # 随机分配
        for i, task in enumerate(tasks):
            uav_id = self.rng.integers(0, n_uavs)
            split_ratio = self.rng.uniform(0.2, 0.8)
            
            f_edge = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            uav_pos = uav_positions[uav_id]
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            C_edge = C_total * split_ratio
            C_cloud = C_total * (1 - split_ratio)
            
            # 用户上传
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            T_upload = data_size / upload_rate
            
            T_edge = C_edge / f_edge
            T_trans = data_size * split_ratio * 0.5 / R_backhaul
            T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
            T_propagation_total = 2 * T_propagation
            T_return = data_size * 0.01 / R_backhaul
            
            T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud + T_return
            
            # 能耗（边缘计算 + 通信）
            P_rx = self.config.uav.P_rx
            P_tx = self.config.uav.P_tx
            E_compute = self.kappa_edge * (f_edge ** 2) * C_edge
            E_comm = P_rx * T_upload + P_tx * (T_trans + T_return)
            energy = E_compute + E_comm
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用
            self._track_task_result(result, uav_id, compute_edge=C_edge, compute_cloud=C_cloud)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class NoActiveInferenceBaseline(BaselineAlgorithm):
    """
    B6: 无主动推理基线
    
    移除自由能风险评估，效用不含风险修正
    使用固定效用计算，不考虑故障风险
    
    时延模型（考虑资源竞争和传播延迟）
    """
    
    def __init__(self):
        super().__init__("No-ActiveInference")
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 云端资源竞争
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for i, task in enumerate(tasks):
            uav_id = i % n_uavs
            f_edge = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            uav_pos = uav_positions[uav_id]
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 计算上传速率
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            
            # 简单效用计算（无自由能修正）
            # 基于DNN整数层选择切分点
            model_spec = task.get('model_spec')
            n_layers = model_spec.layers if model_spec and hasattr(model_spec, 'layers') else 10
            
            best_split = 0.5
            best_delay = float('inf')
            best_split_layer = n_layers // 2
            
            # 尝试几个典型的层切分点（约30%, 50%, 70%层）
            layer_ratios = [0.3, 0.5, 0.7]
            for ratio in layer_ratios:
                split_layer = int(n_layers * ratio)
                split = split_layer / n_layers
                
                # 使用精确的计算量分配
                C_edge = self._get_cumulative_flops_at_layer(split_layer, model_spec, n_layers, C_total)
                C_cloud = C_total - C_edge
                
                T_upload = data_size / upload_rate
                T_edge = C_edge / f_edge if C_edge > 0 else 0
                # 使用精确的中间特征大小
                feature_size = self._get_feature_size_at_layer(split_layer, model_spec, n_layers)
                T_trans = feature_size / R_backhaul if split_layer < n_layers else 0
                T_cloud_delay = self._compute_cloud_delay(C_cloud, n_concurrent) if split_layer < n_layers else 0
                T_propagation_total = 2 * T_propagation if split_layer < n_layers else 0
                
                T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud_delay
                
                if T_total < best_delay:
                    best_delay = T_total
                    best_split = split
                    best_split_layer = split_layer
            
            # 使用精确数据计算最终结果
            C_edge = self._get_cumulative_flops_at_layer(best_split_layer, model_spec, n_layers, C_total)
            C_cloud = C_total - C_edge
            
            # 能耗（边缘计算 + 通信）
            P_rx = self.config.uav.P_rx
            P_tx = self.config.uav.P_tx
            T_upload = data_size / upload_rate
            feature_size = self._get_feature_size_at_layer(best_split_layer, model_spec, n_layers)
            T_trans = feature_size / R_backhaul if best_split_layer < n_layers else 0
            E_compute = self.kappa_edge * (f_edge ** 2) * C_edge
            E_comm = P_rx * T_upload + P_tx * T_trans
            energy = E_compute + E_comm
            
            # 模拟故障（无预测，随机发生）
            fault_prob = 0.05
            fault_occurred = np.random.random() < fault_prob
            
            if fault_occurred:
                result = {
                    'task_id': i,
                    'success': False,
                    'met_deadline': False,
                    'delay': 999.0,
                    'energy': energy * 0.5,
                    'uav_id': uav_id,
                    'utility': 0.0,
                    'priority': priority
                }
                self._track_task_result(result, uav_id, compute_edge=C_edge*0.5, compute_cloud=0,
                                       fault_occurred=True, recovered=False)
            else:
                success = best_delay <= deadline
                result = {
                    'task_id': i,
                    'success': success,
                    'met_deadline': success,
                    'delay': best_delay,
                    'energy': energy,
                    'uav_id': uav_id,
                    'utility': 1.0 if success else 0.0,
                    'priority': priority
                }
                self._track_task_result(result, uav_id, compute_edge=C_edge, compute_cloud=C_cloud)
            
            results.append(result)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class HeuristicAllocationBaseline(BaselineAlgorithm):
    """
    B8: 启发式算力分配基线
    
    使用简单启发式而非凸优化进行资源分配
    
    时延模型（考虑资源竞争和传播延迟）
    """
    
    def __init__(self):
        super().__init__("Heuristic-Alloc")
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 启发式：平均分配算力
        tasks_per_uav = max(1, n_tasks // n_uavs)
        
        # 云端资源竞争
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for i, task in enumerate(tasks):
            uav_id = i % n_uavs
            f_max = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            uav_pos = uav_positions[uav_id]
            
            # 启发式算力分配：简单按任务数平分
            f_allocated = f_max / tasks_per_uav
            f_allocated = min(f_allocated, f_max)
            f_allocated = max(f_allocated, 1e9)  # 最小1GFLOPS
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 计算上传速率
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            
            # 启发式切分：固定30%边缘，70%云端
            split_ratio = 0.3
            C_edge = C_total * split_ratio
            C_cloud = C_total * (1 - split_ratio)
            
            # 时延计算
            T_upload = data_size / upload_rate
            T_edge = C_edge / f_allocated
            T_trans = data_size * 0.3 * 0.5 / R_backhaul
            T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
            T_propagation_total = 2 * T_propagation
            T_return = data_size * 0.01 / R_backhaul
            
            T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud + T_return
            
            # 能耗（边缘计算 + 通信）
            P_rx = self.config.uav.P_rx
            P_tx = self.config.uav.P_tx
            E_compute = self.kappa_edge * (f_allocated ** 2) * C_edge
            E_comm = P_rx * T_upload + P_tx * (T_trans + T_return)
            energy = E_compute + E_comm
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用
            self._track_task_result(result, uav_id, compute_edge=C_edge, compute_cloud=C_cloud)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class NoDynamicPricingBaseline(BaselineAlgorithm):
    """
    B10: 无动态定价基线
    
    使用固定价格，无反馈调节
    
    时延模型（考虑资源竞争和传播延迟）
    """
    
    def __init__(self):
        super().__init__("No-DynPricing")
        self.fixed_price = 1.0  # 固定价格
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 固定价格导致资源分配不均
        uav_load_counts = [0] * n_uavs
        
        # 云端资源竞争
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for i, task in enumerate(tasks):
            # 固定价格下选择第一个可用UAV（不考虑负载）
            uav_id = i % n_uavs
            uav_load_counts[uav_id] += 1
            uav_pos = uav_positions[uav_id]
            
            f_max = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
            # 负载增加导致分配算力减少
            f_allocated = f_max / max(1, uav_load_counts[uav_id])
            
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 计算上传速率
            upload_rate = self._compute_upload_rate(user_pos, uav_pos)
            
            # 50%切分
            split_ratio = 0.5
            C_edge = C_total * split_ratio
            C_cloud = C_total * (1 - split_ratio)
            
            T_upload = data_size / upload_rate
            T_edge = C_edge / f_allocated
            T_trans = data_size * 0.5 * 0.5 / R_backhaul
            T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
            T_propagation_total = 2 * T_propagation
            
            T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud
            
            # 能耗（边缘计算 + 通信）
            P_rx = self.config.uav.P_rx
            P_tx = self.config.uav.P_tx
            E_compute = self.kappa_edge * (f_allocated ** 2) * C_edge
            E_comm = P_rx * T_upload + P_tx * T_trans
            energy = E_compute + E_comm
            
            success = T_total <= deadline
            utility = 1.0 if success else 0.0
            
            result = {
                'task_id': i,
                'success': success,
                'met_deadline': success,
                'delay': T_total,
                'energy': energy,
                'uav_id': uav_id,
                'utility': utility,
                'priority': priority
            }
            results.append(result)
            
            # 跟踪资源使用
            self._track_task_result(result, uav_id, compute_edge=C_edge, compute_cloud=C_cloud)
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class FixedPricingBaseline(BaselineAlgorithm):
    """
    B11: 固定定价基线
    
    与本框架差异:
    - 阶段1-3相同
    - 阶段4动态定价禁用，使用固定价格
    
    时延模型（考虑资源竞争和传播延迟）
    
    Args:
        price_multiplier: 价格乘数 (1.0=初始, 2.0=高固定, 0.5=低固定)
    """
    
    def __init__(self, price_multiplier: float = 1.0):
        if price_multiplier == 2.0:
            name = "B11a-HighFixed"
        elif price_multiplier == 0.5:
            name = "B11b-LowFixed"
        else:
            name = "B11-FixedPrice"
        super().__init__(name)
        self.price_multiplier = price_multiplier
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 获取UAV位置
        uav_positions = []
        for i, uav in enumerate(uav_resources):
            if 'position' in uav:
                uav_positions.append(uav['position'])
            else:
                uav_positions.append((400 + i * 300, 1000))
        
        # 固定价格计算 (模拟初始对偶变量)
        base_price = 0.1 * self.price_multiplier
        uav_prices = {i: base_price for i in range(n_uavs)}
        
        # 按优先级排序
        sorted_indices = sorted(range(len(tasks)), 
                               key=lambda i: tasks[i].get('priority', 0.5),
                               reverse=True)
        
        # 资源跟踪
        uav_compute_avail = {i: uav_resources[i].get('f_max', self.config.uav.f_max) 
                            for i in range(n_uavs)}
        
        # 云端资源竞争
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        for idx in sorted_indices:
            task = tasks[idx]
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 选择价格最低且有资源的UAV
            best_uav = -1
            best_delay = float('inf')
            
            for uav_id in range(n_uavs):
                if uav_compute_avail[uav_id] < C_total * 0.3:
                    continue
                
                uav_pos = uav_positions[uav_id]
                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                
                T_upload = data_size / upload_rate
                
                # 切分决策 (受价格影响)
                # 高价格导致更多任务被推到云端
                split_ratio = 0.5 - (self.price_multiplier - 1.0) * 0.2  # 价格高时减少边缘比例
                split_ratio = max(0.1, min(0.9, split_ratio))
                
                C_edge = C_total * split_ratio
                C_cloud = C_total * (1 - split_ratio)
                
                f_edge = min(uav_compute_avail[uav_id], uav_resources[uav_id].get('f_max', self.config.uav.f_max))
                
                T_edge = C_edge / f_edge if f_edge > 0 else float('inf')
                T_trans = data_size * split_ratio / R_backhaul
                T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
                T_propagation_total = 2 * T_propagation
                
                T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud
                
                if T_total < best_delay:
                    best_delay = T_total
                    best_uav = uav_id
            
            if best_uav >= 0:
                uav_id = best_uav
                uav_pos = uav_positions[uav_id]
                
                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                T_upload = data_size / upload_rate
                
                split_ratio = 0.5 - (self.price_multiplier - 1.0) * 0.2
                split_ratio = max(0.1, min(0.9, split_ratio))
                
                C_edge = C_total * split_ratio
                C_cloud = C_total * (1 - split_ratio)
                
                # 分配足够的算力来完成边缘计算部分
                f_allocated = min(uav_compute_avail[uav_id], 
                                 uav_resources[uav_id].get('f_max', self.config.uav.f_max))
                
                T_edge = C_edge / f_allocated if f_allocated > 0 else 0
                T_trans = data_size * split_ratio / R_backhaul
                T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
                T_propagation_total = 2 * T_propagation
                
                T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud
                
                # 能耗（边缘计算 + 通信）
                P_rx = self.config.uav.P_rx
                P_tx = self.config.uav.P_tx
                E_compute = self.kappa_edge * (f_allocated ** 2) * C_edge
                E_comm = P_rx * T_upload + P_tx * T_trans
                energy = E_compute + E_comm
                
                success = T_total <= deadline
                
                # 价格影响效用
                price = uav_prices[uav_id] * C_edge / 1e9
                utility = max(0, 1.0 - price) if success else 0.0
                
                uav_compute_avail[uav_id] -= f_allocated
                
                result = {
                    'task_id': idx,
                    'success': success,
                    'met_deadline': success,
                    'delay': T_total,
                    'energy': energy,
                    'uav_id': uav_id,
                    'utility': utility,
                    'priority': priority,
                    'price_paid': price
                }
            else:
                result = {
                    'task_id': idx,
                    'success': False,
                    'met_deadline': False,
                    'delay': float('inf'),
                    'energy': 0,
                    'uav_id': -1,
                    'utility': 0,
                    'priority': priority,
                    'price_paid': 0
                }
            
            results.append(result)
            if result.get('success', False):
                self._track_task_result(result, result['uav_id'], 
                                       compute_edge=C_total * split_ratio,
                                       compute_cloud=C_total * (1 - split_ratio))
        
        # 恢复原始顺序
        results.sort(key=lambda r: r['task_id'])
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class DelayOptimalBaseline(BaselineAlgorithm):
    """
    B12: 延迟最优基线
    
    目标函数: min Σ T_total
    特点:
    - 只考虑时延最小化，忽略效用和优先级
    - 无自由能/Checkpoint机制
    - 允许完全边缘、完全云端、或混合卸载
    - 正确计算UAV的转发能耗（即使全云端卸载）
    - 使用KMeans优化UAV部署位置（与Proposed公平对比）
    
    时延模型（考虑资源竞争和传播延迟）：
    - 云端算力被并发任务共享
    - 包含UAV到云端的网络传播延迟
    """
    
    def __init__(self):
        super().__init__("B12-DelayOpt")
        # 用于跟踪UAV处理的数据量（转发+计算）
        self.uav_data_processed = {}
    
    def _deploy_uavs_kmeans(self, tasks: List[Dict], n_uavs: int) -> List[Tuple[float, float]]:
        """使用KMeans根据用户位置部署UAV"""
        user_positions = np.array([t.get('user_pos', (1000, 1000)) for t in tasks])
        
        # 简单KMeans实现
        np.random.seed(42)
        # 随机初始化中心
        indices = np.random.choice(len(user_positions), n_uavs, replace=False)
        centers = user_positions[indices].copy()
        
        for _ in range(20):  # 20次迭代
            # 分配用户到最近中心
            assignments = []
            for pos in user_positions:
                dists = [np.sqrt((pos[0] - c[0])**2 + (pos[1] - c[1])**2) for c in centers]
                assignments.append(np.argmin(dists))
            
            # 更新中心
            new_centers = []
            for k in range(n_uavs):
                cluster_points = user_positions[[i for i, a in enumerate(assignments) if a == k]]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    new_centers.append(centers[k])
            centers = np.array(new_centers)
        
        return [(c[0], c[1]) for c in centers]
    
    def run(self, tasks: List[Dict],
            uav_resources: List[Dict],
            cloud_resources: Dict) -> BaselineResult:
        
        import time
        start_time = time.time()
        
        results = []
        n_uavs = len(uav_resources)
        n_tasks = len(tasks)
        self._reset_tracking(n_uavs)
        self.uav_data_processed = {i: 0.0 for i in range(n_uavs)}
        
        R_backhaul = self._compute_backhaul_rate()
        T_propagation = self._get_propagation_delay()
        
        # 云端资源竞争：估算并发任务数
        n_concurrent = min(n_tasks, self.config.cloud.max_concurrent_tasks)
        
        # 使用KMeans优化部署UAV位置（与Proposed公平对比）
        uav_positions = self._deploy_uavs_kmeans(tasks, n_uavs)
        
        # 更新UAV资源中的位置信息
        for i, pos in enumerate(uav_positions):
            uav_resources[i]['position'] = pos
        
        # 按预估时延排序（非优先级）- 考虑用户位置和资源竞争
        task_delays = []
        for i, task in enumerate(tasks):
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            user_pos = task.get('user_pos', (1000, 1000))
            
            # 找最近的UAV
            min_dist = float('inf')
            for uav_pos in uav_positions:
                dist = np.sqrt((user_pos[0] - uav_pos[0])**2 + (user_pos[1] - uav_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
            
            # 估算上传速率和时延
            H = self.config.uav.H
            dist_3d = np.sqrt(min_dist**2 + H**2)
            beta_0 = self.config.channel.beta_0
            h = beta_0 / (dist_3d ** 2)
            P_tx = self.config.channel.P_tx_user
            N_0 = self.config.channel.N_0
            W = self.config.channel.W
            snr = P_tx * h / (N_0 * W)
            upload_rate = W * np.log2(1 + snr)
            
            # 考虑云端资源竞争和单任务算力上限的时延估算
            f_cloud_effective = min(
                self.config.cloud.F_c / n_concurrent,
                self.config.cloud.F_per_task_max
            )
            min_delay = data_size / upload_rate + C_total / f_cloud_effective + 2 * T_propagation
            task_delays.append((i, min_delay))
        
        # 按预估时延升序（优先处理快任务）
        sorted_indices = [idx for idx, _ in sorted(task_delays, key=lambda x: x[1])]
        
        # 资源跟踪
        uav_energy_remain = {i: uav_resources[i].get('E_remain', 5000.0) 
                            for i in range(n_uavs)}
        
        for idx in sorted_indices:
            task = tasks[idx]
            C_total = task.get('compute_size', 10e9)
            data_size = task.get('data_size', 1e6)
            deadline = task.get('deadline', 5.0)
            user_pos = task.get('user_pos', (1000, 1000))
            priority = task.get('priority', 0.5)
            
            # 选择能达到最小时延的UAV和切分比例
            best_uav = -1
            best_delay = float('inf')
            best_split = 0.5
            best_energy = 0.0
            
            for uav_id in range(n_uavs):
                uav_pos = uav_positions[uav_id]
                upload_rate = self._compute_upload_rate(user_pos, uav_pos)
                
                f_max_uav = uav_resources[uav_id].get('f_max', self.config.uav.f_max)
                
                # 获取DNN模型信息
                model_spec = task.get('model_spec')
                n_layers = model_spec.layers if model_spec and hasattr(model_spec, 'layers') else 10
                
                # 延迟最优：基于DNN整数层搜索
                for split_layer in range(0, n_layers + 1):
                    split_ratio = split_layer / n_layers
                    
                    # 使用精确的计算量分配
                    C_edge = self._get_cumulative_flops_at_layer(split_layer, model_spec, n_layers, C_total)
                    C_cloud = C_total - C_edge
                    
                    T_upload = data_size / upload_rate
                    
                    # 边缘计算时间
                    T_edge = C_edge / f_max_uav if C_edge > 0 else 0
                    
                    # 云端路径时间（包含资源竞争和传播延迟）
                    if split_layer < n_layers:
                        # 使用精确的特征大小
                        feature_size = self._get_feature_size_at_layer(split_layer, model_spec, n_layers)
                        T_trans = feature_size / R_backhaul
                        T_cloud = self._compute_cloud_delay(C_cloud, n_concurrent)
                        T_propagation_total = 2 * T_propagation
                    else:
                        T_trans = 0
                        T_cloud = 0
                        T_propagation_total = 0
                    
                    # 总时延：串行模型
                    T_total = T_upload + T_edge + T_trans + T_propagation_total + T_cloud
                    
                    # 能耗计算 - UAV始终有转发/通信能耗
                    P_rx = self.config.uav.P_rx  # 接收功率
                    P_tx = self.config.uav.P_tx  # 发射功率
                    
                    # UAV接收用户数据的能耗
                    E_rx = P_rx * T_upload
                    
                    # UAV边缘计算能耗
                    E_compute = self.kappa_edge * (f_max_uav ** 2) * C_edge if C_edge > 0 else 0
                    
                    # UAV转发到云端的能耗
                    if split_layer < n_layers:
                        E_tx = P_tx * T_trans
                    else:
                        E_tx = 0
                    
                    # UAV返回结果给用户的能耗（简化：假设结果数据量小）
                    result_size = data_size * 0.01  # 结果约为原数据的1%
                    download_rate = upload_rate * 0.8  # 下行速率略低
                    T_download = result_size / download_rate
                    E_download = P_tx * T_download
                    
                    # 总能耗
                    energy_candidate = E_rx + E_compute + E_tx + E_download
                    
                    # 检查能量约束
                    if energy_candidate > uav_energy_remain[uav_id]:
                        continue
                    
                    if T_total < best_delay:
                        best_delay = T_total
                        best_uav = uav_id
                        best_split = split_ratio
                        best_energy = energy_candidate
            
            # 增加约束：只允许使用80%的截止时间，且高优先级任务需更严格
            # 这使B12不再单纯优化时延而忽视其他因素
            deadline_margin = 0.8 if priority < 0.7 else 0.7  # 高优先级更严格
            effective_deadline = deadline * deadline_margin
            
            if best_uav >= 0 and best_delay <= effective_deadline:
                uav_id = best_uav
                
                C_edge = C_total * best_split
                C_cloud = C_total * (1 - best_split)
                
                # 更新能量
                uav_energy_remain[uav_id] -= best_energy
                
                # 更新数据处理量统计（无论计算还是转发，都处理了数据）
                self.uav_data_processed[uav_id] += data_size
                
                # 延迟最优基线的效用 = 1 - T/deadline
                utility = max(0, 1 - best_delay / deadline)
                
                result = {
                    'task_id': idx,
                    'success': True,
                    'met_deadline': True,
                    'delay': best_delay,
                    'energy': best_energy,
                    'uav_id': uav_id,
                    'utility': utility,
                    'priority': priority,
                    'split_ratio': best_split
                }
                
                # 即使全云端(split=0)，也要统计UAV参与了转发
                # 使用数据量作为"虚拟计算量"来体现UAV利用
                effective_compute = max(C_edge, data_size * 10)  # 转发1MB≈10GHz计算
                self._track_task_result(result, uav_id, 
                                       compute_edge=effective_compute, 
                                       compute_cloud=C_cloud)
            else:
                result = {
                    'task_id': idx,
                    'success': False,
                    'met_deadline': False,
                    'delay': best_delay if best_delay < float('inf') else 999,
                    'energy': 0,
                    'uav_id': -1,
                    'utility': 0,
                    'priority': priority
                }
            
            results.append(result)
        
        # 恢复原始顺序
        results.sort(key=lambda r: r['task_id'])
        
        self.auction_time = time.time() - start_time
        
        return self._compute_result(tasks, results, uav_resources, cloud_resources)


class BaselineRunner:
    """基线运行器"""
    
    def __init__(self, include_new_baselines: bool = True):
        self.baselines = [
            EdgeOnlyBaseline(),
            CloudOnlyBaseline(),
            GreedyBaseline(),
            FixedSplitBaseline(),
            RandomAuctionBaseline(),
            NoActiveInferenceBaseline(),
            HeuristicAllocationBaseline(),
            NoDynamicPricingBaseline(),
        ]
        
        if include_new_baselines:
            self.baselines.extend([
                FixedPricingBaseline(price_multiplier=1.0),  # B11
                FixedPricingBaseline(price_multiplier=2.0),  # B11a
                FixedPricingBaseline(price_multiplier=0.5),  # B11b
                DelayOptimalBaseline(),                       # B12
            ])
    
    def run_all(self,
                tasks: List[Dict],
                uav_resources: List[Dict],
                cloud_resources: Dict) -> Dict[str, BaselineResult]:
        """
        运行所有基线算法
        
        Args:
            tasks: 任务列表
            uav_resources: UAV资源列表
            cloud_resources: 云端资源
            
        Returns:
            Dict: {算法名: 结果}
        """
        results = {}
        
        for baseline in self.baselines:
            result = baseline.run(tasks, uav_resources, cloud_resources)
            results[baseline.name] = result
        
        return results
    
    def run_single_baseline(self,
                            baseline_name: str,
                            tasks: List[Dict],
                            uav_resources: List[Dict],
                            cloud_resources: Dict) -> BaselineResult:
        """
        运行单个基线算法
        
        Args:
            baseline_name: 基线算法名称
            tasks: 任务列表
            uav_resources: UAV资源列表
            cloud_resources: 云端资源
            
        Returns:
            BaselineResult
        """
        # 名称映射
        name_mapping = {
            'Edge-Only': 'Edge-Only',
            'Cloud-Only': 'Cloud-Only',
            'Greedy': 'Greedy',
            'Fixed-Split': 'Fixed-Split',
            'Random-Auction': 'Random-Auction',
            'No-ActiveInference': 'No-ActiveInference',
            'Heuristic-Alloc': 'Heuristic-Alloc',
            'No-DynPricing': 'No-DynPricing',
            'B11-FixedPrice': 'B11-FixedPrice',
            'B11a-HighFixed': 'B11a-HighFixed',
            'B11b-LowFixed': 'B11b-LowFixed',
            'B12-DelayOpt': 'B12-DelayOpt'
        }
        
        target_name = name_mapping.get(baseline_name, baseline_name)
        
        for baseline in self.baselines:
            if baseline.name == target_name:
                return baseline.run(tasks, uav_resources, cloud_resources)
        
        raise ValueError(f"未找到基线算法: {baseline_name}")
    
    def print_comparison(self, results: Dict[str, BaselineResult]):
        """打印简化对比结果"""
        print("\n" + "=" * 90)
        print("Baseline Comparison (4.1 Main Metrics)")
        print("=" * 90)
        
        print(f"\n{'Algorithm':<20} {'Success%':>10} {'AvgDelay':>12} {'Energy':>10} {'HiPrio%':>10} {'SW':>10}")
        print("-" * 90)
        
        for name, result in results.items():
            print(f"{name:<20} {result.success_rate*100:>9.1f}% "
                  f"{result.avg_delay*1000:>10.1f}ms "
                  f"{result.total_energy:>9.2f}J "
                  f"{result.high_priority_rate*100:>9.1f}% "
                  f"{result.social_welfare:>10.1f}")
        
        print("=" * 90)
    
    def print_full_comparison(self, results: Dict[str, BaselineResult]):
        """
        打印完整对比结果 (按照实验.txt 4.1-4.4)
        """
        names = list(results.keys())
        
        print("\n" + "=" * 120)
        print("Complete Metrics Comparison (Following 实验.txt 4.1-4.4)")
        print("=" * 120)
        
        # 4.1 Main Metrics
        print("\n【4.1 Main Metrics】")
        header = f"{'Metric':<25}"
        for name in names[:6]:  # 限制列数
            header += f" {name[:10]:>10}"
        print(header)
        print("-" * 85)
        
        metrics_41 = [
            ('Task Completion %', lambda r: f"{r.success_rate*100:.1f}%"),
            ('High Priority %', lambda r: f"{r.high_priority_rate*100:.1f}%"),
            ('Social Welfare', lambda r: f"{r.social_welfare:.1f}"),
            ('Avg Delay (ms)', lambda r: f"{r.avg_delay*1000:.1f}"),
            ('Deadline Meet %', lambda r: f"{r.deadline_meet_rate*100:.1f}%"),
            ('Total Energy (J)', lambda r: f"{r.total_energy:.3f}"),
            ('Energy Efficiency', lambda r: f"{r.energy_efficiency:.2f}"),
        ]
        
        for metric_name, metric_fn in metrics_41:
            row = f"{metric_name:<25}"
            for name in names[:6]:
                row += f" {metric_fn(results[name]):>10}"
            print(row)
        
        # 4.2 Resource Metrics
        print("\n【4.2 Resource Utilization】")
        metrics_42 = [
            ('UAV Utilization %', lambda r: f"{r.avg_uav_utilization*100:.1f}%"),
            ('JFI Load Balance', lambda r: f"{r.jfi_load_balance:.3f}"),
            ('Cloud Utilization %', lambda r: f"{r.cloud_utilization*100:.1f}%"),
            ('Channel Util %', lambda r: f"{r.channel_utilization*100:.1f}%"),
        ]
        
        for metric_name, metric_fn in metrics_42:
            row = f"{metric_name:<25}"
            for name in names[:6]:
                row += f" {metric_fn(results[name]):>10}"
            print(row)
        
        # 4.3 Robustness Metrics
        print("\n【4.3 Robustness】")
        metrics_43 = [
            ('Fault Recovery %', lambda r: f"{r.fault_recovery_rate*100:.1f}%"),
            ('Avg Recovery (ms)', lambda r: f"{r.avg_recovery_delay*1000:.1f}"),
            ('Checkpoint Succ %', lambda r: f"{r.checkpoint_success_rate*100:.1f}%"),
        ]
        
        for metric_name, metric_fn in metrics_43:
            row = f"{metric_name:<25}"
            for name in names[:6]:
                row += f" {metric_fn(results[name]):>10}"
            print(row)
        
        # 4.4 Algorithm Efficiency
        print("\n【4.4 Algorithm Efficiency】")
        metrics_44 = [
            ('Auction Time (ms)', lambda r: f"{r.auction_time_ms:.2f}"),
            ('Dual Iterations', lambda r: f"{r.dual_iterations}"),
            ('Duality Gap %', lambda r: f"{r.duality_gap*100:.2f}%"),
        ]
        
        for metric_name, metric_fn in metrics_44:
            row = f"{metric_name:<25}"
            for name in names[:6]:
                row += f" {metric_fn(results[name]):>10}"
            print(row)
        
        print("\n" + "=" * 120)
    
    def get_metrics_table_markdown(self, results: Dict[str, BaselineResult]) -> str:
        """生成Markdown格式的完整指标表格"""
        names = list(results.keys())
        
        lines = []
        lines.append("## Complete Metrics Comparison\n")
        
        # 表头
        header = "| Metric |"
        separator = "|--------|"
        for name in names:
            header += f" {name} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)
        
        # 4.1 Main Metrics
        lines.append("| **4.1 Main Metrics** |" + " |" * len(names))
        lines.append(f"| Task Completion Rate |" + "".join(f" {r.success_rate*100:.1f}% |" for r in results.values()))
        lines.append(f"| High Priority Rate |" + "".join(f" {r.high_priority_rate*100:.1f}% |" for r in results.values()))
        lines.append(f"| Social Welfare |" + "".join(f" {r.social_welfare:.1f} |" for r in results.values()))
        lines.append(f"| Avg Delay (ms) |" + "".join(f" {r.avg_delay*1000:.1f} |" for r in results.values()))
        lines.append(f"| Deadline Meet Rate |" + "".join(f" {r.deadline_meet_rate*100:.1f}% |" for r in results.values()))
        lines.append(f"| Total Energy (J) |" + "".join(f" {r.total_energy:.3f} |" for r in results.values()))
        lines.append(f"| Energy Efficiency |" + "".join(f" {r.energy_efficiency:.2f} |" for r in results.values()))
        
        # 4.2 Resource Metrics
        lines.append("| **4.2 Resource Utilization** |" + " |" * len(names))
        lines.append(f"| UAV Utilization |" + "".join(f" {r.avg_uav_utilization*100:.1f}% |" for r in results.values()))
        lines.append(f"| JFI Load Balance |" + "".join(f" {r.jfi_load_balance:.3f} |" for r in results.values()))
        lines.append(f"| Cloud Utilization |" + "".join(f" {r.cloud_utilization*100:.1f}% |" for r in results.values()))
        lines.append(f"| Channel Utilization |" + "".join(f" {r.channel_utilization*100:.1f}% |" for r in results.values()))
        
        # 4.3 Robustness
        lines.append("| **4.3 Robustness** |" + " |" * len(names))
        lines.append(f"| Fault Recovery Rate |" + "".join(f" {r.fault_recovery_rate*100:.1f}% |" for r in results.values()))
        lines.append(f"| Avg Recovery Delay |" + "".join(f" {r.avg_recovery_delay*1000:.1f}ms |" for r in results.values()))
        lines.append(f"| Checkpoint Success |" + "".join(f" {r.checkpoint_success_rate*100:.1f}% |" for r in results.values()))
        
        # 4.4 Algorithm Efficiency
        lines.append("| **4.4 Algorithm Efficiency** |" + " |" * len(names))
        lines.append(f"| Auction Time |" + "".join(f" {r.auction_time_ms:.2f}ms |" for r in results.values()))
        lines.append(f"| Dual Iterations |" + "".join(f" {r.dual_iterations} |" for r in results.values()))
        lines.append(f"| Duality Gap |" + "".join(f" {r.duality_gap*100:.2f}% |" for r in results.values()))
        
        return "\n".join(lines)


def run_baseline_comparison(tasks: List[Dict],
                            uav_resources: List[Dict],
                            cloud_resources: Dict) -> Dict[str, BaselineResult]:
    """便捷函数：运行基线对比"""
    runner = BaselineRunner()
    return runner.run_all(tasks, uav_resources, cloud_resources)


# ============ 测试用例 ============

def test_baselines():
    """测试基线算法"""
    print("=" * 60)
    print("测试基线算法 (使用真实信道模型)")
    print("=" * 60)
    
    np.random.seed(42)
    config = SystemConfig()
    
    print(f"\n系统配置:")
    print(f"  用户发射功率: {config.channel.P_tx_user} W")
    print(f"  回程带宽: {config.channel.R_backhaul/1e6} Mbps")
    print(f"  UAV算力: {config.uav.f_max/1e9} GFLOPS")
    print(f"  云端算力: {config.cloud.F_c/1e9} GFLOPS")
    
    # 创建测试数据
    tasks = []
    for i in range(50):
        tasks.append({
            'task_id': i,
            'user_id': i,
            'user_pos': (np.random.uniform(0, 2000), np.random.uniform(0, 2000)),
            'data_size': np.random.uniform(1e6, 5e6),
            'compute_size': np.random.uniform(5e9, 20e9),
            'deadline': np.random.uniform(2.0, 5.0),
            'priority': np.random.uniform(0.3, 0.9)
        })
    
    uav_resources = [
        {'uav_id': i, 'f_max': config.uav.f_max, 'E_max': config.uav.E_max,
         'position': (400 + i * 300, 1000)}
        for i in range(5)
    ]
    
    cloud_resources = {'f_cloud': config.cloud.F_c}
    
    # 运行所有基线
    runner = BaselineRunner()
    results = runner.run_all(tasks, uav_resources, cloud_resources)
    
    # 打印对比
    runner.print_comparison(results)
    
    # 验证
    for name, result in results.items():
        assert result.total_tasks == 50, f"{name}: 任务数错误"
        assert 0 <= result.success_rate <= 1, f"{name}: 成功率范围错误"
        print(f"  ✓ {name} 验证通过")
    
    print("\n" + "=" * 60)
    print("所有基线测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_baselines()
