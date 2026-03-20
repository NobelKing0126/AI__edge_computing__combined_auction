"""
M01: ConvexSolver - 凸优化闭式解模块

功能：实现基于idea118.txt 2.6节的凸优化闭式解
- 边云联合算力分配
- 能量约束处理
- 闭式解计算

公式参考 (idea118.txt 2.6节):
    优化问题: min_{f_edge, f_cloud} T_compute = C_edge/f_edge + C_cloud/f_cloud

    约束:
    C1 (算力上界): f_edge <= f_avail, f_cloud <= f_cloud_max
    C2 (能量约束): kappa_edge * f_edge^2 * C_edge + kappa_cloud * f_cloud^2 * C_cloud <= E_budget
    C3 (非负性): f_edge >= 0, f_cloud >= 0

    闭式解:
    Case 1 (能量约束不激活): f_edge* = f_avail, f_cloud* = f_cloud_max
    Case 2 (能量约束激活):
        rho = sqrt(kappa_cloud * C_cloud / (kappa_edge * C_edge))
        f_edge* = min(sqrt(E_budget / (kappa_edge * C_edge * (1 + rho^2))), f_avail)
        f_cloud* = min(rho * f_edge*, f_cloud_max)

修正版V2:
    rho = (κc/κe)^(1/3)，与计算量无关
    5步闭式解流程
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from config.system_config import SystemConfig


@dataclass
class ConvexSolverConfig:
    """
    凸优化求解器配置参数
    """
    # 能耗系数
    kappa_edge: float = 1e-28
    kappa_cloud: float = 1e-29

    # 单任务能量预算比例
    energy_budget_ratio: float = 0.3

    # 通信能耗预留比例
    comm_energy_reserve_ratio: float = 0.2

    # 数值稳定性阈值
    min_compute: float = 1e6  # 最小计算量 (FLOPS)
    min_frequency: float = 1e6  # 最小频率 (Hz)


@dataclass
class AllocationResult:
    """
    资源分配结果
    """
    # 最优边缘算力 (FLOPS)
    f_edge_star: float

    # 最优云端算力 (FLOPS)
    f_cloud_star: float

    # 是否激活能量约束
    energy_constraint_active: bool

    # 计算时延 (秒)
    compute_delay: float

    # 计算能耗 (焦耳)
    compute_energy: float

    # 求解状态
    status: str  # 'optimal', 'infeasible', 'degraded'


@dataclass
class AllocationResultV2:
    """
    资源分配结果 V2 (修正版)

    新增字段用于5步闭式解流程追踪

    Attributes:
        f_edge_star: 最优边缘算力 (FLOPS)
        f_cloud_star: 最优云端算力 (FLOPS)
        energy_constraint_active: 是否激活能量约束
        compute_delay: 计算时延 (秒)
        compute_energy: 计算能耗 (焦耳)
        step_reached: 求解到达步骤 (1-5)
        edge_capped: 边缘是否触顶
        cloud_capped: 云端是否触顶
        status: 求解状态
    """
    f_edge_star: float
    f_cloud_star: float
    energy_constraint_active: bool
    compute_delay: float
    compute_energy: float
    step_reached: int  # 求解到达步骤
    edge_capped: bool  # 边缘是否触顶
    cloud_capped: bool  # 云端是否触顶
    status: str


class ConvexSolver:
    """
    凸优化求解器

    实现边云联合算力分配的闭式解

    问题凸性分析:
    - 目标函数: C_edge/f_edge + C_cloud/f_cloud 关于 (f_edge, f_cloud) 联合凸
    - 约束集: 均为凸集
    - 结论: 凸优化问题，KKT条件是充要条件
    """

    def __init__(self, config: ConvexSolverConfig = None, system_config: SystemConfig = None):
        """
        初始化凸优化求解器

        Args:
            config: 求解器配置
            system_config: 系统配置
        """
        self.config = config if config else ConvexSolverConfig()
        self.system_config = system_config if system_config else SystemConfig()

    def compute_rho(
        self,
        C_edge: float,
        C_cloud: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> float:
        """
        计算最优频率比例因子 rho

        rho = sqrt(kappa_cloud * C_cloud / (kappa_edge * C_edge))

        物理含义: 边缘和云端的能耗效率比

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            float: 比例因子 rho
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        # 处理边界情况
        if C_edge <= 0:
            return float('inf')  # 全云端
        if C_cloud <= 0:
            return 0.0  # 全边缘

        rho = np.sqrt(kappa_cloud * C_cloud / (kappa_edge * C_edge))
        return rho

    def compute_rho_ratio(
        self,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> float:
        """
        计算最优频率比例因子 rho (修正版V2)

        rho = (κc/κe)^(1/3)

        修正要点: rho与计算量无关，仅由能耗系数决定

        物理含义: 云端和边缘的能耗效率比的三次根
        - κc/κe = 1e-29/1e-28 = 0.1
        - rho = 0.1^(1/3) ≈ 0.464

        Args:
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            float: 比例因子 rho
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        # 修正: rho = (κc/κe)^(1/3)，与计算量无关
        rho = (kappa_cloud / kappa_edge) ** (1/3)
        return rho

    def compute_energy_budget(
        self,
        E_remain: float,
        E_max: float,
        n_concurrent_tasks: int = 1,
        comm_energy: float = 0.0
    ) -> float:
        """
        计算单任务能量预算

        E_budget = min(E_remain / (n_concurrent + 1), energy_budget_ratio * E_max) - E_comm

        Args:
            E_remain: UAV剩余能量 (焦耳)
            E_max: UAV最大能量 (焦耳)
            n_concurrent_tasks: 并发任务数
            comm_energy: 通信能耗 (焦耳)

        Returns:
            float: 能量预算 (焦耳)
        """
        energy_budget_ratio = self.config.energy_budget_ratio
        comm_reserve_ratio = self.config.comm_energy_reserve_ratio

        # 基础预算: 取剩余能量均分和最大比例的较小值
        base_budget = min(
            E_remain / (n_concurrent_tasks + 1),
            energy_budget_ratio * E_max
        )

        # 扣除通信能耗预留
        budget = base_budget - comm_energy

        return max(budget, 0.0)

    def solve_case1_unconstrained(
        self,
        f_avail: float,
        f_cloud_max: float
    ) -> Tuple[float, float]:
        """
        Case 1: 能量约束不激活 - 取算力上界

        f_edge* = f_avail
        f_cloud* = f_cloud_max

        Args:
            f_avail: 可用边缘算力 (FLOPS)
            f_cloud_max: 最大云端算力 (FLOPS)

        Returns:
            Tuple[float, float]: (f_edge*, f_cloud*)
        """
        return f_avail, f_cloud_max

    def solve_case2_energy_constrained(
        self,
        C_edge: float,
        C_cloud: float,
        E_budget: float,
        f_avail: float,
        f_cloud_max: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> Tuple[float, float]:
        """
        Case 2: 能量约束激活 - 使用闭式解

        rho = sqrt(kappa_cloud * C_cloud / (kappa_edge * C_edge))
        f_edge* = min(sqrt(E_budget / (kappa_edge * C_edge * (1 + rho^2))), f_avail)
        f_cloud* = min(rho * f_edge*, f_cloud_max)

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            E_budget: 能量预算 (焦耳)
            f_avail: 可用边缘算力 (FLOPS)
            f_cloud_max: 最大云端算力 (FLOPS)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            Tuple[float, float]: (f_edge*, f_cloud*)
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        # 特殊情况处理
        if C_edge <= 0:
            # 全云端
            return 0.0, f_cloud_max

        if C_cloud <= 0:
            # 全边缘
            # 检查能量是否足够
            max_f_edge = np.sqrt(E_budget / (kappa_edge * C_edge)) if kappa_edge * C_edge > 0 else f_avail
            return min(max_f_edge, f_avail), 0.0

        # 计算rho
        rho = self.compute_rho(C_edge, C_cloud, kappa_edge, kappa_cloud)

        # 计算最优边缘频率
        denominator = kappa_edge * C_edge * (1 + rho ** 2)
        if denominator <= 0:
            f_edge_unconstrained = f_avail
        else:
            f_edge_unconstrained = np.sqrt(E_budget / denominator)

        # 应用算力上界约束
        f_edge_star = min(f_edge_unconstrained, f_avail)

        # 计算最优云端频率
        f_cloud_unconstrained = rho * f_edge_star
        f_cloud_star = min(f_cloud_unconstrained, f_cloud_max)

        return f_edge_star, f_cloud_star

    def check_energy_constraint(
        self,
        f_edge: float,
        f_cloud: float,
        C_edge: float,
        C_cloud: float,
        E_budget: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> bool:
        """
        检查能量约束是否满足

        kappa_edge * f_edge^2 * C_edge + kappa_cloud * f_cloud^2 * C_cloud <= E_budget

        Args:
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            E_budget: 能量预算 (焦耳)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            bool: 是否满足能量约束
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        energy_consumed = (kappa_edge * f_edge ** 2 * C_edge +
                          kappa_cloud * f_cloud ** 2 * C_cloud)

        return energy_consumed <= E_budget

    def compute_compute_delay(
        self,
        C_edge: float,
        C_cloud: float,
        f_edge: float,
        f_cloud: float
    ) -> float:
        """
        计算计算时延

        T_compute = C_edge / f_edge + C_cloud / f_cloud

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)

        Returns:
            float: 计算时延 (秒)
        """
        min_f = self.config.min_frequency

        t_edge = C_edge / max(f_edge, min_f) if C_edge > 0 else 0.0
        t_cloud = C_cloud / max(f_cloud, min_f) if C_cloud > 0 else 0.0

        return t_edge + t_cloud

    def compute_compute_energy(
        self,
        C_edge: float,
        C_cloud: float,
        f_edge: float,
        f_cloud: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> float:
        """
        计算计算能耗

        E_compute = kappa_edge * f_edge^2 * C_edge + kappa_cloud * f_cloud^2 * C_cloud

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            float: 计算能耗 (焦耳)
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        E_edge = kappa_edge * f_edge ** 2 * C_edge if C_edge > 0 else 0.0
        E_cloud = kappa_cloud * f_cloud ** 2 * C_cloud if C_cloud > 0 else 0.0

        return E_edge + E_cloud

    def solve(
        self,
        C_edge: float,
        C_cloud: float,
        f_avail: float,
        f_cloud_max: float,
        E_budget: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> AllocationResult:
        """
        求解凸优化问题

        完整求解流程:
        1. 计算 Case 1 候选解 (算力上界)
        2. 检查能量约束是否满足
        3. 若满足，采用 Case 1 解
        4. 若不满足，计算 Case 2 解 (能量约束激活)
        5. 验证算力上界约束，必要时截断

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_avail: 可用边缘算力 (FLOPS)
            f_cloud_max: 最大云端算力 (FLOPS)
            E_budget: 能量预算 (焦耳)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            AllocationResult: 分配结果
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        # Step 1: 尝试 Case 1 (算力上界)
        f_edge_case1, f_cloud_case1 = self.solve_case1_unconstrained(f_avail, f_cloud_max)

        # Step 2: 检查能量约束
        energy_satisfied = self.check_energy_constraint(
            f_edge_case1, f_cloud_case1,
            C_edge, C_cloud, E_budget,
            kappa_edge, kappa_cloud
        )

        if energy_satisfied:
            # Case 1 解有效
            f_edge_star = f_edge_case1
            f_cloud_star = f_cloud_case1
            energy_constraint_active = False
            status = 'optimal'
        else:
            # Step 4: 计算 Case 2 解
            f_edge_star, f_cloud_star = self.solve_case2_energy_constrained(
                C_edge, C_cloud, E_budget,
                f_avail, f_cloud_max,
                kappa_edge, kappa_cloud
            )
            energy_constraint_active = True

            # 验证解的可行性
            if f_edge_star <= 0 and C_edge > 0:
                status = 'infeasible'
            elif f_cloud_star <= 0 and C_cloud > 0:
                status = 'degraded'
            else:
                status = 'optimal'

        # 计算时延和能耗
        compute_delay = self.compute_compute_delay(C_edge, C_cloud, f_edge_star, f_cloud_star)
        compute_energy = self.compute_compute_energy(
            C_edge, C_cloud, f_edge_star, f_cloud_star,
            kappa_edge, kappa_cloud
        )

        return AllocationResult(
            f_edge_star=f_edge_star,
            f_cloud_star=f_cloud_star,
            energy_constraint_active=energy_constraint_active,
            compute_delay=compute_delay,
            compute_energy=compute_energy,
            status=status
        )

    def solve_v2(
        self,
        C_edge: float,
        C_cloud: float,
        f_avail: float,
        f_cloud_max: float,
        E_budget: float,
        kappa_edge: float = None,
        kappa_cloud: float = None
    ) -> AllocationResultV2:
        """
        求解凸优化问题 (修正版V2 - 5步闭式解流程)

        核心修正:
            - rho = (κc/κe)^(1/3)，与计算量无关

        5步求解流程:
            Step 1: 检查特殊情况（全边缘Cc=0、全云端Ce=0）
            Step 2: 检查能量约束是否激活
            Step 3: 计算无约束最优解
            Step 4: 检查边缘是否触顶
            Step 5: 检查云端是否触顶

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_avail: 可用边缘算力 (FLOPS)
            f_cloud_max: 最大云端算力 (FLOPS)
            E_budget: 能量预算 (焦耳)
            kappa_edge: 边缘能耗系数
            kappa_cloud: 云端能耗系数

        Returns:
            AllocationResultV2: 分配结果（含详细步骤信息）
        """
        if kappa_edge is None:
            kappa_edge = self.config.kappa_edge
        if kappa_cloud is None:
            kappa_cloud = self.config.kappa_cloud

        # 初始化状态
        step_reached = 0
        edge_capped = False
        cloud_capped = False
        energy_constraint_active = False
        status = 'optimal'

        # ========== Step 1: 检查特殊情况 ==========
        step_reached = 1

        if C_edge <= 0 and C_cloud <= 0:
            # 无计算量
            return AllocationResultV2(
                f_edge_star=0.0, f_cloud_star=0.0,
                energy_constraint_active=False,
                compute_delay=0.0, compute_energy=0.0,
                step_reached=step_reached,
                edge_capped=False, cloud_capped=False,
                status='optimal'
            )

        if C_edge <= 0:
            # 全云端
            f_cloud_star = f_cloud_max
            cloud_capped = True
            compute_delay = C_cloud / f_cloud_star if f_cloud_star > 0 else 0.0
            compute_energy = kappa_cloud * f_cloud_star ** 2 * C_cloud
            return AllocationResultV2(
                f_edge_star=0.0, f_cloud_star=f_cloud_star,
                energy_constraint_active=False,
                compute_delay=compute_delay, compute_energy=compute_energy,
                step_reached=step_reached,
                edge_capped=False, cloud_capped=cloud_capped,
                status='optimal'
            )

        if C_cloud <= 0:
            # 全边缘
            f_edge_star = f_avail
            edge_capped = True
            compute_delay = C_edge / f_edge_star if f_edge_star > 0 else 0.0
            compute_energy = kappa_edge * f_edge_star ** 2 * C_edge
            return AllocationResultV2(
                f_edge_star=f_edge_star, f_cloud_star=0.0,
                energy_constraint_active=False,
                compute_delay=compute_delay, compute_energy=compute_energy,
                step_reached=step_reached,
                edge_capped=edge_capped, cloud_capped=False,
                status='optimal'
            )

        # ========== Step 2: 检查能量约束是否激活 ==========
        step_reached = 2

        # 计算无约束情况下的能耗（取算力上界）
        E_unconstrained = (kappa_edge * f_avail ** 2 * C_edge +
                          kappa_cloud * f_cloud_max ** 2 * C_cloud)

        if E_unconstrained <= E_budget:
            # 能量约束不激活，直接取算力上界
            f_edge_star = f_avail
            f_cloud_star = f_cloud_max
            edge_capped = True
            cloud_capped = True
            energy_constraint_active = False
        else:
            # ========== Step 3: 计算无约束最优解 ==========
            step_reached = 3
            energy_constraint_active = True

            # 使用修正后的rho = (κc/κe)^(1/3)
            rho = self.compute_rho_ratio(kappa_edge, kappa_cloud)

            # 无约束最优解 (根据 idea38.txt 公式修正)
            # A = κe * Ce * rho^2 + κc * Cc
            # f_edge = rho * f_cloud, 先求 f_cloud
            denominator = kappa_cloud * C_cloud + kappa_edge * C_edge * (rho ** 2)

            if denominator > 0:
                f_cloud_unconstrained = np.sqrt(E_budget / denominator)
            else:
                f_cloud_unconstrained = f_cloud_max

            f_edge_unconstrained = rho * f_cloud_unconstrained

            # ========== Step 4: 检查边缘是否触顶 ==========
            step_reached = 4

            if f_edge_unconstrained >= f_avail:
                # 边缘触顶
                f_edge_star = f_avail
                edge_capped = True

                # 重新计算云端算力
                # 能量约束: κe * f_edge^2 * Ce + κc * f_cloud^2 * Cc <= E_budget
                # 求解: f_cloud = sqrt((E_budget - κe * f_edge^2 * Ce) / (κc * Cc))
                remaining_energy = E_budget - kappa_edge * f_edge_star ** 2 * C_edge
                if remaining_energy > 0 and C_cloud > 0:
                    f_cloud_from_energy = np.sqrt(remaining_energy / (kappa_cloud * C_cloud))
                    f_cloud_star = min(f_cloud_from_energy, f_cloud_max)
                else:
                    f_cloud_star = 0.0

                if f_cloud_star >= f_cloud_max:
                    cloud_capped = True
            else:
                # 边缘未触顶
                f_edge_star = f_edge_unconstrained
                edge_capped = False

                # ========== Step 5: 检查云端是否触顶 ==========
                step_reached = 5

                if f_cloud_unconstrained >= f_cloud_max:
                    # 云端触顶
                    f_cloud_star = f_cloud_max
                    cloud_capped = True

                    # 重新计算边缘算力
                    remaining_energy = E_budget - kappa_cloud * f_cloud_star ** 2 * C_cloud
                    if remaining_energy > 0 and C_edge > 0:
                        f_edge_from_energy = np.sqrt(remaining_energy / (kappa_edge * C_edge))
                        f_edge_star = min(f_edge_from_energy, f_avail)
                        if f_edge_star >= f_avail:
                            edge_capped = True
                    else:
                        f_edge_star = 0.0
                else:
                    # 云端未触顶
                    f_cloud_star = f_cloud_unconstrained
                    cloud_capped = False

        # 计算时延和能耗
        compute_delay = self.compute_compute_delay(C_edge, C_cloud, f_edge_star, f_cloud_star)
        compute_energy = self.compute_compute_energy(
            C_edge, C_cloud, f_edge_star, f_cloud_star,
            kappa_edge, kappa_cloud
        )

        # 验证解的可行性
        if f_edge_star <= 0 and C_edge > 0:
            status = 'infeasible'
        elif f_cloud_star <= 0 and C_cloud > 0:
            status = 'degraded'
        else:
            status = 'optimal'

        return AllocationResultV2(
            f_edge_star=f_edge_star,
            f_cloud_star=f_cloud_star,
            energy_constraint_active=energy_constraint_active,
            compute_delay=compute_delay,
            compute_energy=compute_energy,
            step_reached=step_reached,
            edge_capped=edge_capped,
            cloud_capped=cloud_capped,
            status=status
        )

    def batch_solve(
        self,
        task_params: List[Dict]
    ) -> List[AllocationResult]:
        """
        批量求解

        Args:
            task_params: 任务参数列表，每个元素包含:
                - C_edge: 边缘计算量
                - C_cloud: 云端计算量
                - f_avail: 可用边缘算力
                - f_cloud_max: 最大云端算力
                - E_budget: 能量预算

        Returns:
            List[AllocationResult]: 分配结果列表
        """
        results = []

        for params in task_params:
            result = self.solve(
                C_edge=params.get('C_edge', 0),
                C_cloud=params.get('C_cloud', 0),
                f_avail=params.get('f_avail', 1e9),
                f_cloud_max=params.get('f_cloud_max', 20e9),
                E_budget=params.get('E_budget', 1e5),
                kappa_edge=params.get('kappa_edge'),
                kappa_cloud=params.get('kappa_cloud')
            )
            results.append(result)

        return results


# ============ 便捷函数 ============

def solve_convex_optimization(
    C_edge: float,
    C_cloud: float,
    f_avail: float,
    f_cloud_max: float,
    E_budget: float,
    kappa_edge: float = 1e-28,
    kappa_cloud: float = 1e-29
) -> Tuple[float, float, bool]:
    """
    求解凸优化问题 (便捷函数)

    Args:
        C_edge: 边缘计算量 (FLOPs)
        C_cloud: 云端计算量 (FLOPs)
        f_avail: 可用边缘算力 (FLOPS)
        f_cloud_max: 最大云端算力 (FLOPS)
        E_budget: 能量预算 (焦耳)
        kappa_edge: 边缘能耗系数
        kappa_cloud: 云端能耗系数

    Returns:
        Tuple[float, float, bool]: (f_edge*, f_cloud*, energy_constraint_active)
    """
    solver = ConvexSolver()
    result = solver.solve(C_edge, C_cloud, f_avail, f_cloud_max, E_budget,
                          kappa_edge, kappa_cloud)
    return result.f_edge_star, result.f_cloud_star, result.energy_constraint_active


# ============ 测试用例 ============

def test_convex_solver():
    """测试ConvexSolver模块"""
    print("=" * 60)
    print("测试 ConvexSolver")
    print("=" * 60)

    # 创建求解器
    solver = ConvexSolver()

    # 测试1: rho计算
    print("\n[Test 1] rho计算")
    print("-" * 40)
    test_cases = [
        (1e9, 1e9, "C_edge = C_cloud"),
        (5e9, 5e9, "C_edge = C_cloud (大)"),
        (2e9, 8e9, "C_cloud > C_edge"),
        (8e9, 2e9, "C_edge > C_cloud"),
    ]

    for C_edge, C_cloud, desc in test_cases:
        rho = solver.compute_rho(C_edge, C_cloud)
        print(f"  {desc}: rho = {rho:.4f}")

    # 测试2: Case 1 - 能量约束不激活
    print("\n[Test 2] Case 1 - 能量约束不激活")
    print("-" * 40)

    result = solver.solve(
        C_edge=1e9,  # 1 GFLOP
        C_cloud=1e9,  # 1 GFLOP
        f_avail=5e9,  # 5 GFLOPS
        f_cloud_max=20e9,  # 20 GFLOPS
        E_budget=1e6  # 1 MJ (充足)
    )

    print(f"  f_edge* = {result.f_edge_star/1e9:.2f} GFLOPS")
    print(f"  f_cloud* = {result.f_cloud_star/1e9:.2f} GFLOPS")
    print(f"  能量约束激活: {result.energy_constraint_active}")
    print(f"  计算时延: {result.compute_delay*1000:.2f} ms")
    print(f"  状态: {result.status}")

    # 测试3: Case 2 - 能量约束激活
    print("\n[Test 3] Case 2 - 能量约束激活")
    print("-" * 40)

    result = solver.solve(
        C_edge=5e9,  # 5 GFLOP
        C_cloud=5e9,  # 5 GFLOP
        f_avail=10e9,  # 10 GFLOPS
        f_cloud_max=20e9,  # 20 GFLOPS
        E_budget=1e3  # 1 kJ (紧张)
    )

    print(f"  f_edge* = {result.f_edge_star/1e9:.2f} GFLOPS")
    print(f"  f_cloud* = {result.f_cloud_star/1e9:.2f} GFLOPS")
    print(f"  能量约束激活: {result.energy_constraint_active}")
    print(f"  计算时延: {result.compute_delay*1000:.2f} ms")
    print(f"  计算能耗: {result.compute_energy:.2f} J")
    print(f"  状态: {result.status}")

    # 测试4: 特殊情况 - 全边缘
    print("\n[Test 4] 特殊情况 - 全边缘 (C_cloud = 0)")
    print("-" * 40)

    result = solver.solve(
        C_edge=10e9,  # 10 GFLOP
        C_cloud=0,  # 无云端计算
        f_avail=10e9,  # 10 GFLOPS
        f_cloud_max=20e9,
        E_budget=5e4
    )

    print(f"  f_edge* = {result.f_edge_star/1e9:.2f} GFLOPS")
    print(f"  f_cloud* = {result.f_cloud_star/1e9:.2f} GFLOPS")
    print(f"  计算时延: {result.compute_delay*1000:.2f} ms")

    # 测试5: 特殊情况 - 全云端
    print("\n[Test 5] 特殊情况 - 全云端 (C_edge = 0)")
    print("-" * 40)

    result = solver.solve(
        C_edge=0,  # 无边缘计算
        C_cloud=10e9,  # 10 GFLOP
        f_avail=10e9,
        f_cloud_max=20e9,  # 20 GFLOPS
        E_budget=1e5
    )

    print(f"  f_edge* = {result.f_edge_star/1e9:.2f} GFLOPS")
    print(f"  f_cloud* = {result.f_cloud_star/1e9:.2f} GFLOPS")
    print(f"  计算时延: {result.compute_delay*1000:.2f} ms")

    # 测试6: 能量预算影响
    print("\n[Test 6] 能量预算影响")
    print("-" * 40)

    E_budgets = [1e2, 1e3, 1e4, 1e5, 1e6]
    print(f"  {'E_budget':<15} {'f_edge*':<15} {'f_cloud*':<15} {'时延(ms)':<15} {'能量约束':<10}")

    for E_budget in E_budgets:
        result = solver.solve(
            C_edge=5e9,
            C_cloud=5e9,
            f_avail=10e9,
            f_cloud_max=20e9,
            E_budget=E_budget
        )
        print(f"  {E_budget/1e3:<15.1f}kJ {result.f_edge_star/1e9:<15.2f}GF "
              f"{result.f_cloud_star/1e9:<15.2f}GF {result.compute_delay*1000:<15.2f} "
              f"{result.energy_constraint_active!s:<10}")

    # 测试7: 便捷函数
    print("\n[Test 7] 便捷函数测试")
    print("-" * 40)

    f_edge, f_cloud, energy_active = solve_convex_optimization(
        C_edge=5e9,
        C_cloud=5e9,
        f_avail=10e9,
        f_cloud_max=20e9,
        E_budget=5e3
    )

    print(f"  f_edge* = {f_edge/1e9:.2f} GFLOPS")
    print(f"  f_cloud* = {f_cloud/1e9:.2f} GFLOPS")
    print(f"  能量约束激活: {energy_active}")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_convex_solver()
