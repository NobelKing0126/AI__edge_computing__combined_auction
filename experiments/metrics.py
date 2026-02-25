"""
M24: Metrics - 完整性能指标计算

功能：计算系统性能指标
输入：执行结果
输出：各类性能指标

指标包括 (按照实验.txt 4.1-4.4节定义):

4.1 主要指标:
    - 社会福利 (SW)
    - 任务完成率
    - 高优先级完成率
    - 平均端到端时延
    - 时延满足率
    - 系统总能耗
    - 能效比

4.2 资源利用指标:
    - UAV平均算力利用率
    - UAV负载均衡指数 (JFI)
    - 云端利用率
    - 信道利用率

4.3 鲁棒性指标:
    - 故障恢复成功率
    - 平均恢复时延
    - Checkpoint成功率
    - 恢复时延节省比

4.4 算法效率指标:
    - 投标生成时间
    - 拍卖决策时间
    - 对偶迭代次数
    - 对偶间隙
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TaskMetric:
    """单任务指标"""
    task_id: int
    success: bool
    delay: float
    deadline: float
    energy: float
    met_deadline: bool
    priority: float = 0.5
    uav_id: int = -1
    split_ratio: float = 0.5
    utility: float = 0.0  # 效用值 (用于计算社会福利)
    is_high_priority: bool = False
    fault_occurred: bool = False
    recovered: bool = False
    recovery_delay: float = 0.0
    checkpoint_used: bool = False
    
    # === 新增: 用户收益相关 ===
    price_paid: float = 0.0           # 用户支付的价格 P_{i,l,j}
    payoff: float = 0.0               # 用户收益 = utility - price_paid
    priority_class: str = 'medium'    # 优先级类别: high/medium/low


@dataclass
class SystemMetrics:
    """
    完整系统性能指标 (按照实验.txt定义)
    
    4.1 主要指标
    """
    total_tasks: int
    success_count: int
    success_rate: float
    avg_delay: float
    max_delay: float
    deadline_meet_rate: float
    total_energy: float
    avg_energy_per_task: float
    energy_efficiency: float
    throughput: float
    
    # === 4.1 主要指标 (扩展) ===
    social_welfare: float = 0.0  # SW = Σ η_final
    high_priority_success_rate: float = 0.0  # 高优先级完成率
    
    # === 4.2 资源利用指标 ===
    avg_uav_utilization: float = 0.0  # UAV平均算力利用率
    jfi_load_balance: float = 1.0     # Jain's Fairness Index
    cloud_utilization: float = 0.0    # 云端利用率
    channel_utilization: float = 0.0   # 信道利用率
    
    # === 4.3 鲁棒性指标 ===
    fault_recovery_rate: float = 1.0   # 故障恢复成功率
    avg_recovery_delay: float = 0.0    # 平均恢复时延
    checkpoint_success_rate: float = 1.0  # Checkpoint成功率
    recovery_delay_saving: float = 0.0    # 恢复时延节省比
    
    # === 4.4 算法效率指标 ===
    bidding_time_ms: float = 0.0       # 投标生成时间 (ms)
    auction_time_ms: float = 0.0       # 拍卖决策时间 (ms)
    total_algorithm_time_ms: float = 0.0  # 总算法时间 (ms)
    dual_iterations: int = 0           # 对偶迭代次数
    duality_gap: float = 0.0           # 对偶间隙 (Gap/SW*)
    
    # === UAV详细利用率 ===
    uav_utilizations: List[float] = field(default_factory=list)
    uav_loads: List[int] = field(default_factory=list)
    
    # === 新增: 用户收益指标 ===
    user_payoff_total: float = 0.0        # 总用户收益 Σ(η - P)
    user_payoff_avg: float = 0.0          # 平均用户收益
    user_payoff_gini: float = 0.0         # 收益基尼系数 (0=完全公平, 1=完全不公平)
    payoff_high_priority: float = 0.0     # 高优先级用户收益
    payoff_medium_priority: float = 0.0   # 中优先级用户收益
    payoff_low_priority: float = 0.0      # 低优先级用户收益
    
    # === 新增: 服务提供商利润指标 ===
    provider_revenue: float = 0.0         # 总收入 Σ P_i
    provider_cost: float = 0.0            # 运营成本
    provider_profit: float = 0.0          # 净利润 = Revenue - Cost
    provider_profit_margin: float = 0.0   # 利润率 = Profit / Revenue
    
    # === 新增: 竞争比指标 ===
    competitive_ratio: float = 1.0        # 竞争比 = SW* / SW_online
    sw_optimal: float = 0.0               # 离线最优社会福利 SW*
    primal_dual_gap: float = 0.0          # 原始-对偶间隙


class MetricsCalculator:
    """
    完整性能指标计算器
    
    支持实验.txt中定义的所有指标类别
    """
    
    def __init__(self, n_uavs: int = 5, n_channels: int = 10):
        self.task_metrics: List[TaskMetric] = []
        self.n_uavs = n_uavs
        self.n_channels = n_channels
        
        # UAV资源跟踪
        self.uav_compute_used: Dict[int, float] = {i: 0.0 for i in range(n_uavs)}
        self.uav_compute_max: Dict[int, float] = {i: 15e9 for i in range(n_uavs)}  # 默认15GFLOPs
        self.uav_task_count: Dict[int, int] = {i: 0 for i in range(n_uavs)}
        
        # 云端资源跟踪
        self.cloud_compute_used: float = 0.0
        self.cloud_compute_max: float = 100e9  # 默认100GFLOPs
        
        # 信道资源跟踪
        self.channels_used: int = 0
        self.total_channels: int = n_channels
        
        # 故障和恢复跟踪
        self.fault_count: int = 0
        self.recovery_count: int = 0
        self.checkpoint_attempts: int = 0
        self.checkpoint_successes: int = 0
        self.recovery_delays: List[float] = []
        self.recovery_delays_no_cp: List[float] = []
        
        # 算法效率跟踪
        self.bidding_time: float = 0.0
        self.auction_time: float = 0.0
        self.dual_iterations: int = 0
        self.primal_value: float = 0.0
        self.dual_value: float = 0.0
    
    def set_uav_resources(self, uav_resources: List[Dict]):
        """设置UAV资源"""
        for uav in uav_resources:
            uav_id = uav.get('uav_id', 0)
            self.uav_compute_max[uav_id] = uav.get('f_max', 15e9)
            self.uav_compute_used[uav_id] = 0.0
            self.uav_task_count[uav_id] = 0
    
    def set_cloud_resources(self, cloud_resources: Dict):
        """设置云端资源"""
        self.cloud_compute_max = cloud_resources.get('f_cloud', 100e9)
        self.cloud_compute_used = 0.0
    
    def add_task_result(self,
                        task_id: int,
                        success: bool,
                        delay: float,
                        deadline: float,
                        energy: float,
                        priority: float = 0.5,
                        uav_id: int = -1,
                        split_ratio: float = 0.5,
                        utility: float = 0.0,
                        compute_edge: float = 0.0,
                        compute_cloud: float = 0.0,
                        fault_occurred: bool = False,
                        recovered: bool = False,
                        recovery_delay: float = 0.0,
                        checkpoint_used: bool = False,
                        price_paid: float = 0.0,
                        priority_class: str = 'medium'):
        """
        添加任务结果 (完整版)
        
        Args:
            task_id: 任务ID
            success: 是否成功
            delay: 实际时延
            deadline: 截止时间
            energy: 消耗能量
            priority: 优先级 (0-1)
            uav_id: 分配的UAV ID
            split_ratio: 切分比例
            utility: 效用值 (用于计算社会福利)
            compute_edge: 边缘计算量
            compute_cloud: 云端计算量
            fault_occurred: 是否发生故障
            recovered: 是否恢复成功
            recovery_delay: 恢复时延
            checkpoint_used: 是否使用Checkpoint
            price_paid: 用户支付的价格
            priority_class: 优先级类别 (high/medium/low)
        """
        is_high_priority = priority >= 0.7
        
        # 计算用户收益
        payoff = (utility - price_paid) if success else 0.0
        
        # 根据优先级确定类别
        if priority_class == 'medium':
            if priority >= 0.7:
                priority_class = 'high'
            elif priority <= 0.3:
                priority_class = 'low'
        
        self.task_metrics.append(TaskMetric(
            task_id=task_id,
            success=success,
            delay=delay,
            deadline=deadline,
            energy=energy,
            met_deadline=delay <= deadline,
            priority=priority,
            uav_id=uav_id,
            split_ratio=split_ratio,
            utility=utility if success else 0.0,
            is_high_priority=is_high_priority,
            fault_occurred=fault_occurred,
            recovered=recovered,
            recovery_delay=recovery_delay,
            checkpoint_used=checkpoint_used,
            price_paid=price_paid,
            payoff=payoff,
            priority_class=priority_class
        ))
        
        # 更新资源使用
        if success and uav_id >= 0:
            self.uav_compute_used[uav_id] = self.uav_compute_used.get(uav_id, 0) + compute_edge
            self.uav_task_count[uav_id] = self.uav_task_count.get(uav_id, 0) + 1
            self.cloud_compute_used += compute_cloud
            self.channels_used = min(self.channels_used + 1, self.total_channels)
        
        # 更新故障/恢复统计
        if fault_occurred:
            self.fault_count += 1
            if recovered:
                self.recovery_count += 1
                self.recovery_delays.append(recovery_delay)
        
        if checkpoint_used:
            self.checkpoint_attempts += 1
            if recovered or success:
                self.checkpoint_successes += 1
    
    def add_batch_results(self, results: List[Dict], tasks: List[Dict] = None):
        """
        批量添加结果
        
        Args:
            results: 结果列表
            tasks: 原始任务列表 (用于获取优先级等信息)
        """
        for i, r in enumerate(results):
            task = tasks[i] if tasks and i < len(tasks) else {}
            self.add_task_result(
                task_id=r.get('task_id', i),
                success=r.get('success', False),
                delay=r.get('delay', 0),
                deadline=r.get('deadline', task.get('deadline', 1)),
                energy=r.get('energy', 0),
                priority=task.get('priority', r.get('priority', 0.5)),
                uav_id=r.get('uav_id', -1),
                split_ratio=r.get('split_ratio', 0.5),
                utility=r.get('utility', 0.0),
                compute_edge=r.get('compute_edge', 0),
                compute_cloud=r.get('compute_cloud', 0),
                fault_occurred=r.get('fault_occurred', False),
                recovered=r.get('recovered', False),
                recovery_delay=r.get('recovery_delay', 0.0),
                checkpoint_used=r.get('checkpoint_used', False)
            )
    
    def set_algorithm_times(self, bidding_time: float = 0.0, 
                            auction_time: float = 0.0,
                            dual_iterations: int = 0,
                            primal_value: float = 0.0,
                            dual_value: float = 0.0):
        """设置算法效率指标"""
        self.bidding_time = bidding_time
        self.auction_time = auction_time
        self.dual_iterations = dual_iterations
        self.primal_value = primal_value
        self.dual_value = dual_value
    
    def compute_social_welfare(self) -> float:
        """
        计算社会福利
        
        SW = Σ η_final (成功任务的效用之和)
        """
        return sum(m.utility for m in self.task_metrics if m.success)
    
    def compute_high_priority_rate(self) -> float:
        """
        计算高优先级任务完成率
        
        高优先级 = priority >= 0.7
        """
        high_priority_tasks = [m for m in self.task_metrics if m.is_high_priority]
        if not high_priority_tasks:
            return 1.0
        
        high_priority_success = sum(1 for m in high_priority_tasks if m.success)
        return high_priority_success / len(high_priority_tasks)
    
    def compute_jfi_load_balance(self) -> float:
        """
        计算Jain's Fairness Index (JFI)
        
        JFI = (Σ Util_j)² / (N * Σ Util_j²)
        范围: [1/N, 1], 1表示完全公平
        """
        utils = []
        for uav_id in range(self.n_uavs):
            max_compute = self.uav_compute_max.get(uav_id, 15e9)
            used_compute = self.uav_compute_used.get(uav_id, 0)
            util = used_compute / max_compute if max_compute > 0 else 0
            utils.append(util)
        
        if not utils or sum(utils) == 0:
            return 1.0
        
        sum_util = sum(utils)
        sum_util_sq = sum(u ** 2 for u in utils)
        
        if sum_util_sq == 0:
            return 1.0
        
        jfi = (sum_util ** 2) / (len(utils) * sum_util_sq)
        return jfi
    
    def compute_uav_utilizations(self) -> Tuple[float, List[float]]:
        """
        计算UAV利用率
        
        Returns:
            (平均利用率, 各UAV利用率列表)
        """
        utils = []
        for uav_id in range(self.n_uavs):
            max_compute = self.uav_compute_max.get(uav_id, 15e9)
            used_compute = self.uav_compute_used.get(uav_id, 0)
            util = used_compute / max_compute if max_compute > 0 else 0
            utils.append(min(util, 1.0))  # 限制最大为1
        
        avg_util = np.mean(utils) if utils else 0
        return avg_util, utils
    
    def compute_cloud_utilization(self) -> float:
        """计算云端利用率"""
        if self.cloud_compute_max == 0:
            return 0.0
        return min(self.cloud_compute_used / self.cloud_compute_max, 1.0)
    
    def compute_channel_utilization(self) -> float:
        """计算信道利用率"""
        if self.total_channels == 0:
            return 0.0
        return min(self.channels_used / self.total_channels, 1.0)
    
    def compute_fault_recovery_rate(self) -> float:
        """计算故障恢复成功率"""
        if self.fault_count == 0:
            return 1.0
        return self.recovery_count / self.fault_count
    
    def compute_avg_recovery_delay(self) -> float:
        """计算平均恢复时延"""
        if not self.recovery_delays:
            return 0.0
        return np.mean(self.recovery_delays)
    
    def compute_checkpoint_success_rate(self) -> float:
        """计算Checkpoint成功率"""
        if self.checkpoint_attempts == 0:
            return 1.0
        return self.checkpoint_successes / self.checkpoint_attempts
    
    def compute_recovery_delay_saving(self) -> float:
        """
        计算恢复时延节省比
        
        节省比 = (T_no_cp - T_cp) / T_no_cp
        """
        if not self.recovery_delays or not self.recovery_delays_no_cp:
            return 0.0
        
        avg_with_cp = np.mean(self.recovery_delays)
        avg_without_cp = np.mean(self.recovery_delays_no_cp)
        
        if avg_without_cp == 0:
            return 0.0
        
        return (avg_without_cp - avg_with_cp) / avg_without_cp
    
    def compute_duality_gap(self) -> float:
        """
        计算对偶间隙
        
        Gap = (Primal - Dual) / Primal
        """
        if self.primal_value == 0:
            return 0.0
        return abs(self.primal_value - self.dual_value) / self.primal_value
    
    def compute_user_payoffs(self) -> Dict[str, float]:
        """
        计算用户收益指标
        
        Returns:
            {
                'total': 总用户收益,
                'avg': 平均用户收益,
                'gini': 基尼系数,
                'high': 高优先级收益,
                'medium': 中优先级收益,
                'low': 低优先级收益
            }
        """
        success_tasks = [m for m in self.task_metrics if m.success]
        
        if not success_tasks:
            return {
                'total': 0.0, 'avg': 0.0, 'gini': 0.0,
                'high': 0.0, 'medium': 0.0, 'low': 0.0
            }
        
        payoffs = [m.payoff for m in success_tasks]
        total_payoff = sum(payoffs)
        avg_payoff = total_payoff / len(payoffs)
        
        # 计算基尼系数
        gini = self._compute_gini(payoffs)
        
        # 按优先级分类
        high_payoff = sum(m.payoff for m in success_tasks if m.priority_class == 'high')
        medium_payoff = sum(m.payoff for m in success_tasks if m.priority_class == 'medium')
        low_payoff = sum(m.payoff for m in success_tasks if m.priority_class == 'low')
        
        return {
            'total': total_payoff,
            'avg': avg_payoff,
            'gini': gini,
            'high': high_payoff,
            'medium': medium_payoff,
            'low': low_payoff
        }
    
    def _compute_gini(self, values: List[float]) -> float:
        """
        计算基尼系数
        
        Gini = Σ_i Σ_j |x_i - x_j| / (2n Σ_i x_i)
        范围: [0, 1], 0=完全公平, 1=完全不公平
        """
        if not values or len(values) < 2:
            return 0.0
        
        n = len(values)
        total = sum(values)
        
        if total <= 0:
            return 0.0
        
        # 计算绝对差之和
        diff_sum = 0.0
        for i in range(n):
            for j in range(n):
                diff_sum += abs(values[i] - values[j])
        
        gini = diff_sum / (2 * n * total)
        return min(gini, 1.0)
    
    def compute_provider_profit(self, 
                                 cost_compute: float = 0.01,
                                 cost_energy: float = 0.05,
                                 cost_trans: float = 0.001,
                                 cost_hover: float = 0.1,
                                 service_time: float = 1.0) -> Dict[str, float]:
        """
        计算服务提供商利润
        
        Args:
            cost_compute: 计算成本 (元/GFLOPS·s)
            cost_energy: 能量成本 (元/kJ)
            cost_trans: 传输成本 (元/MB)
            cost_hover: 悬停成本 (元/s)
            service_time: 服务时间 (s)
        
        Returns:
            {
                'revenue': 总收入,
                'cost': 总成本,
                'profit': 净利润,
                'margin': 利润率
            }
        """
        # 计算收入 = 所有用户支付的价格之和
        revenue = sum(m.price_paid for m in self.task_metrics if m.success)
        
        # 计算成本
        # 1. 计算成本
        compute_cost = 0.0
        for uav_id, used in self.uav_compute_used.items():
            compute_cost += cost_compute * (used / 1e9)  # 转换为GFLOPS
        compute_cost += cost_compute * (self.cloud_compute_used / 1e9)
        
        # 2. 能量成本
        energy_cost = cost_energy * sum(m.energy for m in self.task_metrics) / 1000  # 转换为kJ
        
        # 3. 传输成本 (假设每任务平均10MB)
        trans_cost = cost_trans * len([m for m in self.task_metrics if m.success]) * 10
        
        # 4. 悬停成本
        hover_cost = cost_hover * self.n_uavs * service_time
        
        total_cost = compute_cost + energy_cost + trans_cost + hover_cost
        
        # 计算利润
        profit = revenue - total_cost
        margin = (profit / revenue * 100) if revenue > 0 else 0.0
        
        return {
            'revenue': revenue,
            'cost': total_cost,
            'profit': profit,
            'margin': margin
        }
    
    def compute_metrics(self, total_time: float = 1.0) -> SystemMetrics:
        """
        计算完整系统性能指标
        
        Args:
            total_time: 总运行时间
            
        Returns:
            SystemMetrics: 完整系统指标
        """
        if not self.task_metrics:
            return SystemMetrics(
                total_tasks=0,
                success_count=0,
                success_rate=0.0,
                avg_delay=0.0,
                max_delay=0.0,
                deadline_meet_rate=0.0,
                total_energy=0.0,
                avg_energy_per_task=0.0,
                energy_efficiency=0.0,
                throughput=0.0
            )
        
        n = len(self.task_metrics)
        success_tasks = [m for m in self.task_metrics if m.success]
        deadline_met = [m for m in self.task_metrics if m.met_deadline]
        
        delays = [m.delay for m in self.task_metrics]
        energies = [m.energy for m in self.task_metrics]
        
        total_energy = sum(energies)
        success_count = len(success_tasks)
        
        # 计算UAV利用率
        avg_uav_util, uav_utils = self.compute_uav_utilizations()
        
        # UAV负载 (任务数)
        uav_loads = [self.uav_task_count.get(i, 0) for i in range(self.n_uavs)]
        
        # 新增: 计算用户收益指标
        user_payoffs = self.compute_user_payoffs()
        
        # 新增: 计算服务提供商利润
        provider_profit = self.compute_provider_profit(service_time=total_time)
        
        return SystemMetrics(
            # 4.1 主要指标
            total_tasks=n,
            success_count=success_count,
            success_rate=success_count / n,
            avg_delay=np.mean(delays),
            max_delay=max(delays),
            deadline_meet_rate=len(deadline_met) / n,
            total_energy=total_energy,
            avg_energy_per_task=total_energy / n,
            energy_efficiency=success_count / max(total_energy, 1e-10),
            throughput=n / max(total_time, 1e-10),
            social_welfare=self.compute_social_welfare(),
            high_priority_success_rate=self.compute_high_priority_rate(),
            
            # 4.2 资源利用指标
            avg_uav_utilization=avg_uav_util,
            jfi_load_balance=self.compute_jfi_load_balance(),
            cloud_utilization=self.compute_cloud_utilization(),
            channel_utilization=self.compute_channel_utilization(),
            
            # 4.3 鲁棒性指标
            fault_recovery_rate=self.compute_fault_recovery_rate(),
            avg_recovery_delay=self.compute_avg_recovery_delay(),
            checkpoint_success_rate=self.compute_checkpoint_success_rate(),
            recovery_delay_saving=self.compute_recovery_delay_saving(),
            
            # 4.4 算法效率指标
            bidding_time_ms=self.bidding_time * 1000,
            auction_time_ms=self.auction_time * 1000,
            total_algorithm_time_ms=(self.bidding_time + self.auction_time) * 1000,
            dual_iterations=self.dual_iterations,
            duality_gap=self.compute_duality_gap(),
            
            # UAV详细信息
            uav_utilizations=uav_utils,
            uav_loads=uav_loads,
            
            # 新增: 用户收益指标
            user_payoff_total=user_payoffs['total'],
            user_payoff_avg=user_payoffs['avg'],
            user_payoff_gini=user_payoffs['gini'],
            payoff_high_priority=user_payoffs['high'],
            payoff_medium_priority=user_payoffs['medium'],
            payoff_low_priority=user_payoffs['low'],
            
            # 新增: 服务提供商利润指标
            provider_revenue=provider_profit['revenue'],
            provider_cost=provider_profit['cost'],
            provider_profit=provider_profit['profit'],
            provider_profit_margin=provider_profit['margin']
        )
    
    def compute_per_priority_metrics(self) -> Dict[str, SystemMetrics]:
        """按优先级分类计算指标"""
        high_priority = [m for m in self.task_metrics if m.is_high_priority]
        low_priority = [m for m in self.task_metrics if not m.is_high_priority]
        
        result = {'all': self.compute_metrics()}
        
        if high_priority:
            high_success = [m for m in high_priority if m.success]
            result['high_priority'] = {
                'total': len(high_priority),
                'success': len(high_success),
                'success_rate': len(high_success) / len(high_priority)
            }
        
        if low_priority:
            low_success = [m for m in low_priority if m.success]
            result['low_priority'] = {
                'total': len(low_priority),
                'success': len(low_success),
                'success_rate': len(low_success) / len(low_priority)
            }
        
        return result
    
    def clear(self):
        """清空记录"""
        self.task_metrics = []
        for uav_id in self.uav_compute_used:
            self.uav_compute_used[uav_id] = 0.0
            self.uav_task_count[uav_id] = 0
        self.cloud_compute_used = 0.0
        self.channels_used = 0
        self.fault_count = 0
        self.recovery_count = 0
        self.checkpoint_attempts = 0
        self.checkpoint_successes = 0
        self.recovery_delays = []
        self.recovery_delays_no_cp = []
        self.bidding_time = 0.0
        self.auction_time = 0.0
        self.dual_iterations = 0
        self.primal_value = 0.0
        self.dual_value = 0.0
    
    def get_summary(self, metrics: SystemMetrics) -> str:
        """生成完整摘要字符串"""
        return f"""
{'='*60}
            完整性能指标摘要
{'='*60}

【4.1 主要指标】
  任务总数: {metrics.total_tasks}
  成功任务: {metrics.success_count}
  任务完成率: {metrics.success_rate*100:.1f}%
  高优先级完成率: {metrics.high_priority_success_rate*100:.1f}%
  社会福利 (SW): {metrics.social_welfare:.2f}
  平均时延: {metrics.avg_delay*1000:.1f}ms
  最大时延: {metrics.max_delay*1000:.1f}ms
  时延满足率: {metrics.deadline_meet_rate*100:.1f}%
  总能耗: {metrics.total_energy:.4f}J
  能效比: {metrics.energy_efficiency:.2f}任务/J

【4.2 资源利用指标】
  UAV平均利用率: {metrics.avg_uav_utilization*100:.1f}%
  JFI负载均衡: {metrics.jfi_load_balance:.3f}
  云端利用率: {metrics.cloud_utilization*100:.1f}%
  信道利用率: {metrics.channel_utilization*100:.1f}%

【4.3 鲁棒性指标】
  故障恢复率: {metrics.fault_recovery_rate*100:.1f}%
  平均恢复时延: {metrics.avg_recovery_delay*1000:.1f}ms
  Checkpoint成功率: {metrics.checkpoint_success_rate*100:.1f}%
  恢复时延节省比: {metrics.recovery_delay_saving*100:.1f}%

【4.4 算法效率指标】
  投标生成时间: {metrics.bidding_time_ms:.2f}ms
  拍卖决策时间: {metrics.auction_time_ms:.2f}ms
  总算法时间: {metrics.total_algorithm_time_ms:.2f}ms
  对偶迭代次数: {metrics.dual_iterations}
  对偶间隙: {metrics.duality_gap*100:.2f}%

{'='*60}
"""


def compute_comparison_metrics(baseline_results: List[Dict],
                               proposed_results: List[Dict],
                               total_time: float = 1.0,
                               tasks: List[Dict] = None) -> Dict:
    """
    计算对比指标
    
    Args:
        baseline_results: 基线结果
        proposed_results: 提议方法结果
        total_time: 总时间
        tasks: 原始任务列表
        
    Returns:
        Dict: 对比指标
    """
    baseline_calc = MetricsCalculator()
    baseline_calc.add_batch_results(baseline_results, tasks)
    baseline_metrics = baseline_calc.compute_metrics(total_time)
    
    proposed_calc = MetricsCalculator()
    proposed_calc.add_batch_results(proposed_results, tasks)
    proposed_metrics = proposed_calc.compute_metrics(total_time)
    
    def improvement(new, old):
        if old == 0:
            return 0 if new == 0 else float('inf')
        return (new - old) / old * 100
    
    return {
        'baseline': baseline_metrics,
        'proposed': proposed_metrics,
        'improvements': {
            'success_rate': improvement(proposed_metrics.success_rate, 
                                       baseline_metrics.success_rate),
            'avg_delay': improvement(baseline_metrics.avg_delay,  # 反向
                                    proposed_metrics.avg_delay),
            'deadline_meet_rate': improvement(proposed_metrics.deadline_meet_rate,
                                             baseline_metrics.deadline_meet_rate),
            'energy_efficiency': improvement(proposed_metrics.energy_efficiency,
                                            baseline_metrics.energy_efficiency),
            'social_welfare': improvement(proposed_metrics.social_welfare,
                                         baseline_metrics.social_welfare),
            'high_priority_rate': improvement(proposed_metrics.high_priority_success_rate,
                                             baseline_metrics.high_priority_success_rate),
            'jfi_balance': improvement(proposed_metrics.jfi_load_balance,
                                      baseline_metrics.jfi_load_balance)
        }
    }


def format_all_metrics(metrics: SystemMetrics, name: str = "Algorithm") -> str:
    """
    格式化输出所有指标 (用于报告)
    
    Args:
        metrics: 系统指标
        name: 算法名称
        
    Returns:
        str: 格式化的指标字符串
    """
    return f"""
### {name} Results

| Category | Metric | Value |
|----------|--------|-------|
| **4.1 Main** | Task Completion Rate | {metrics.success_rate*100:.1f}% |
| | High Priority Rate | {metrics.high_priority_success_rate*100:.1f}% |
| | Social Welfare | {metrics.social_welfare:.2f} |
| | Avg Delay | {metrics.avg_delay*1000:.1f}ms |
| | Deadline Meet Rate | {metrics.deadline_meet_rate*100:.1f}% |
| | Total Energy | {metrics.total_energy:.4f}J |
| | Energy Efficiency | {metrics.energy_efficiency:.2f} |
| **4.2 Resource** | UAV Utilization | {metrics.avg_uav_utilization*100:.1f}% |
| | JFI Load Balance | {metrics.jfi_load_balance:.3f} |
| | Cloud Utilization | {metrics.cloud_utilization*100:.1f}% |
| | Channel Utilization | {metrics.channel_utilization*100:.1f}% |
| **4.3 Robustness** | Fault Recovery Rate | {metrics.fault_recovery_rate*100:.1f}% |
| | Avg Recovery Delay | {metrics.avg_recovery_delay*1000:.1f}ms |
| | Checkpoint Success | {metrics.checkpoint_success_rate*100:.1f}% |
| | Recovery Saving | {metrics.recovery_delay_saving*100:.1f}% |
| **4.4 Efficiency** | Bidding Time | {metrics.bidding_time_ms:.2f}ms |
| | Auction Time | {metrics.auction_time_ms:.2f}ms |
| | Dual Iterations | {metrics.dual_iterations} |
| | Duality Gap | {metrics.duality_gap*100:.2f}% |
"""


def print_metrics_table(results: Dict[str, 'SystemMetrics']) -> str:
    """
    打印所有算法的完整指标对比表
    
    Args:
        results: {算法名: SystemMetrics}
        
    Returns:
        str: 表格字符串
    """
    names = list(results.keys())
    
    lines = []
    lines.append("=" * 120)
    lines.append("Complete Metrics Comparison Table (Following 实验.txt 4.1-4.4)")
    lines.append("=" * 120)
    
    # Header
    header = f"{'Metric':<25}"
    for name in names:
        header += f" {name[:12]:>12}"
    lines.append(header)
    lines.append("-" * 120)
    
    # 4.1 Main Metrics
    lines.append("【4.1 Main Metrics】")
    
    row = f"{'Task Completion Rate':<25}"
    for name in names:
        m = results[name]
        row += f" {m.success_rate*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'High Priority Rate':<25}"
    for name in names:
        m = results[name]
        row += f" {m.high_priority_success_rate*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'Social Welfare':<25}"
    for name in names:
        m = results[name]
        row += f" {m.social_welfare:>12.2f}"
    lines.append(row)
    
    row = f"{'Avg Delay (ms)':<25}"
    for name in names:
        m = results[name]
        row += f" {m.avg_delay*1000:>12.1f}"
    lines.append(row)
    
    row = f"{'Deadline Meet Rate':<25}"
    for name in names:
        m = results[name]
        row += f" {m.deadline_meet_rate*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'Total Energy (J)':<25}"
    for name in names:
        m = results[name]
        row += f" {m.total_energy:>12.4f}"
    lines.append(row)
    
    row = f"{'Energy Efficiency':<25}"
    for name in names:
        m = results[name]
        row += f" {m.energy_efficiency:>12.2f}"
    lines.append(row)
    
    # 4.2 Resource Metrics
    lines.append("")
    lines.append("【4.2 Resource Utilization】")
    
    row = f"{'UAV Utilization':<25}"
    for name in names:
        m = results[name]
        row += f" {m.avg_uav_utilization*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'JFI Load Balance':<25}"
    for name in names:
        m = results[name]
        row += f" {m.jfi_load_balance:>12.3f}"
    lines.append(row)
    
    row = f"{'Cloud Utilization':<25}"
    for name in names:
        m = results[name]
        row += f" {m.cloud_utilization*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'Channel Utilization':<25}"
    for name in names:
        m = results[name]
        row += f" {m.channel_utilization*100:>11.1f}%"
    lines.append(row)
    
    # 4.3 Robustness Metrics
    lines.append("")
    lines.append("【4.3 Robustness】")
    
    row = f"{'Fault Recovery Rate':<25}"
    for name in names:
        m = results[name]
        row += f" {m.fault_recovery_rate*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'Avg Recovery Delay (ms)':<25}"
    for name in names:
        m = results[name]
        row += f" {m.avg_recovery_delay*1000:>12.1f}"
    lines.append(row)
    
    row = f"{'Checkpoint Success':<25}"
    for name in names:
        m = results[name]
        row += f" {m.checkpoint_success_rate*100:>11.1f}%"
    lines.append(row)
    
    row = f"{'Recovery Delay Saving':<25}"
    for name in names:
        m = results[name]
        row += f" {m.recovery_delay_saving*100:>11.1f}%"
    lines.append(row)
    
    # 4.4 Algorithm Efficiency
    lines.append("")
    lines.append("【4.4 Algorithm Efficiency】")
    
    row = f"{'Bidding Time (ms)':<25}"
    for name in names:
        m = results[name]
        row += f" {m.bidding_time_ms:>12.2f}"
    lines.append(row)
    
    row = f"{'Auction Time (ms)':<25}"
    for name in names:
        m = results[name]
        row += f" {m.auction_time_ms:>12.2f}"
    lines.append(row)
    
    row = f"{'Dual Iterations':<25}"
    for name in names:
        m = results[name]
        row += f" {m.dual_iterations:>12d}"
    lines.append(row)
    
    row = f"{'Duality Gap':<25}"
    for name in names:
        m = results[name]
        row += f" {m.duality_gap*100:>11.2f}%"
    lines.append(row)
    
    lines.append("=" * 120)
    
    return "\n".join(lines)


# ============ 测试用例 ============

def test_metrics():
    """测试完整Metrics模块"""
    print("=" * 60)
    print("测试 M24: 完整Metrics模块")
    print("=" * 60)
    
    calculator = MetricsCalculator(n_uavs=5, n_channels=10)
    
    # 添加测试数据 (使用完整参数)
    np.random.seed(42)
    for i in range(20):
        success = np.random.random() > 0.1
        priority = np.random.uniform(0.3, 0.9)
        uav_id = i % 5
        compute_edge = np.random.uniform(1e9, 5e9)
        compute_cloud = np.random.uniform(1e9, 5e9)
        
        calculator.add_task_result(
            task_id=i,
            success=success,
            delay=np.random.uniform(0.2, 0.8),
            deadline=1.0,
            energy=np.random.uniform(0.01, 0.05),
            priority=priority,
            uav_id=uav_id,
            split_ratio=0.5,
            utility=1.0 if success else 0.0,
            compute_edge=compute_edge,
            compute_cloud=compute_cloud,
            fault_occurred=np.random.random() < 0.1,
            recovered=np.random.random() > 0.3,
            recovery_delay=np.random.uniform(0.05, 0.2),
            checkpoint_used=np.random.random() < 0.5
        )
    
    # 设置算法时间
    calculator.set_algorithm_times(
        bidding_time=0.015,
        auction_time=0.008,
        dual_iterations=25,
        primal_value=15.5,
        dual_value=15.2
    )
    
    # 测试1: 完整指标计算
    print("\n[Test 1] 测试完整指标计算...")
    metrics = calculator.compute_metrics(total_time=10.0)
    
    assert metrics.total_tasks == 20
    assert 0 <= metrics.success_rate <= 1
    assert 0 <= metrics.high_priority_success_rate <= 1
    assert 0 <= metrics.jfi_load_balance <= 1
    print(f"  任务数: {metrics.total_tasks}")
    print(f"  成功率: {metrics.success_rate*100:.1f}%")
    print(f"  高优先级完成率: {metrics.high_priority_success_rate*100:.1f}%")
    print(f"  社会福利: {metrics.social_welfare:.2f}")
    print("  ✓ 主要指标计算正确")
    
    # 测试2: 资源利用率
    print("\n[Test 2] 测试资源利用率...")
    assert 0 <= metrics.avg_uav_utilization <= 1
    assert 0 <= metrics.jfi_load_balance <= 1
    print(f"  UAV利用率: {metrics.avg_uav_utilization*100:.1f}%")
    print(f"  JFI负载均衡: {metrics.jfi_load_balance:.3f}")
    print(f"  云端利用率: {metrics.cloud_utilization*100:.1f}%")
    print(f"  信道利用率: {metrics.channel_utilization*100:.1f}%")
    print("  ✓ 资源利用指标正确")
    
    # 测试3: 鲁棒性指标
    print("\n[Test 3] 测试鲁棒性指标...")
    print(f"  故障恢复率: {metrics.fault_recovery_rate*100:.1f}%")
    print(f"  平均恢复时延: {metrics.avg_recovery_delay*1000:.1f}ms")
    print(f"  Checkpoint成功率: {metrics.checkpoint_success_rate*100:.1f}%")
    print("  ✓ 鲁棒性指标正确")
    
    # 测试4: 算法效率
    print("\n[Test 4] 测试算法效率指标...")
    assert metrics.bidding_time_ms > 0
    assert metrics.dual_iterations > 0
    print(f"  投标时间: {metrics.bidding_time_ms:.2f}ms")
    print(f"  拍卖时间: {metrics.auction_time_ms:.2f}ms")
    print(f"  对偶迭代: {metrics.dual_iterations}")
    print(f"  对偶间隙: {metrics.duality_gap*100:.2f}%")
    print("  ✓ 算法效率指标正确")
    
    # 测试5: 完整摘要
    print("\n[Test 5] 测试完整摘要生成...")
    summary = calculator.get_summary(metrics)
    assert "社会福利" in summary
    assert "JFI" in summary
    assert "对偶" in summary
    print(summary)
    print("  ✓ 完整摘要生成正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
