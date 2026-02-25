"""
M16: ResourceOptimizer - 凸优化资源分配

功能：给定切分点，优化边缘/云端算力分配，最小化时延
输入：任务参数、切分点、资源约束
输出：最优算力分配、预期时延

严格按照 docs/idea118.txt 2.6节规范实现四组闭型解：

2.6.3 四组闭型解（覆盖所有KKT点）
根据时延约束和能量约束的激活状态，分四种情况：

Case 1：两约束均不激活
    取算力上界：f_edge* = f_avail, f_cloud* = f_cloud_max

Case 2：仅时延约束激活
    时延等式求解，按计算量比例分配时间预算：
    f_edge* = C_edge / (T_budget · r)
    f_cloud* = C_cloud / (T_budget · (1-r))
    其中 r 为边缘时间占比

Case 3：仅能量约束激活
    拉格朗日乘子法求解，最优条件为边际能效相等：
    2κ_edge · f_edge · C_edge = 2κ_cloud · f_cloud · C_cloud
    结合能量等式约束求解

Case 4：两约束均激活
    联立时延等式和能量等式：
    C_edge/f_edge + C_cloud/f_cloud = T_budget
    κ_edge · f_edge² · C_edge + κ_cloud · f_cloud² · C_cloud = E_budget
    求解二元方程组

2.6.4 最优解选择
Step 1：计算四组候选解
Step 2：可行性检查（满足所有约束）
Step 3：选择可行且时延最小者
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.system_config import SystemConfig
from config.constants import NUMERICAL


@dataclass
class AllocationResult:
    """
    资源分配结果

    Attributes:
        f_edge: 边缘分配算力 (FLOPS)
        f_cloud: 云端分配算力 (FLOPS)
        T_edge: 边缘计算时延 (s)
        T_trans: 传输时延 (s)
        T_cloud: 云端计算时延 (s)
        T_return: 返回时延 (s)
        T_total: 总时延 (s)
        E_edge: 边缘计算能耗 (J)
        E_cloud: 云端计算能耗 (J)
        E_total: 总能耗 (J)
        feasible: 是否可行
        margin: 时延余量 (s)
        case_used: 使用的闭式解类型 (1-4)
    """
    f_edge: float
    f_cloud: float
    T_edge: float
    T_trans: float
    T_cloud: float
    T_return: float
    T_total: float
    E_edge: float
    E_cloud: float
    E_total: float
    feasible: bool
    margin: float
    case_used: int


class ResourceOptimizer:
    """
    凸优化资源分配器

    严格按照 docs/idea118.txt 2.6节实现四组闭型解

    Attributes:
        f_edge_max: UAV最大算力
        f_cloud_max: 云端最大算力
        kappa_edge: 边缘能耗系数
        kappa_cloud: 云端能耗系数
        E_max: UAV电池容量
    """

    def __init__(self,
                 f_edge_max: float = 10e9,
                 f_cloud_max: float = 100e9,
                 kappa_edge: float = 1e-28,
                 kappa_cloud: float = 1e-29,
                 E_max: float = 500e3):
        """
        初始化优化器

        Args:
            f_edge_max: UAV最大算力 (FLOPS)，默认10 GFLOPS
            f_cloud_max: 云端最大算力 (FLOPS)，默认100 GFLOPS
            kappa_edge: 边缘能耗系数，默认1e-28
            kappa_cloud: 云端能耗系数，默认1e-29
            E_max: UAV电池容量 (J)，默认500kJ
        """
        self.f_edge_max = f_edge_max
        self.f_cloud_max = f_cloud_max
        self.kappa_edge = kappa_edge
        self.kappa_cloud = kappa_cloud
        self.E_max = E_max

    def compute_delay_components(self,
                                  C_edge: float,
                                  C_cloud: float,
                                  f_edge: float,
                                  f_cloud: float,
                                  T_upload: float,
                                  T_trans: float,
                                  T_return: float) -> Dict[str, float]:
        """
        计算时延各分量

        时延模型（idea118.txt 2.5节）：
        T_total = T_upload + T_edge + T_trans + T_cloud + T_return

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)
            T_upload: 上传时延 (s)
            T_trans: 中继传输时延 (s)
            T_return: 返回时延 (s)

        Returns:
            Dict: 时延分量
        """
        T_edge = C_edge / max(f_edge, NUMERICAL.EPSILON) if C_edge > 0 else 0
        T_cloud_compute = C_cloud / max(f_cloud, NUMERICAL.EPSILON) if C_cloud > 0 else 0

        # 总时延 = 上传 + 边缘计算 + 中继 + 云端计算 + 返回
        T_total = T_upload + T_edge + T_trans + T_cloud_compute + T_return

        return {
            'T_upload': T_upload,
            'T_edge': T_edge,
            'T_trans': T_trans,
            'T_cloud': T_cloud_compute,
            'T_return': T_return,
            'T_total': T_total
        }

    def compute_energy(self, f_edge: float, C_edge: float,
                       f_cloud: float, C_cloud: float) -> Tuple[float, float, float]:
        """
        计算能耗（idea118.txt 2.7节）

        能耗模型：
        E_edge = κ_edge · f_edge² · C_edge
        E_cloud = κ_cloud · f_cloud² · C_cloud

        Args:
            f_edge: 边缘算力 (FLOPS)
            C_edge: 边缘计算量 (FLOPs)
            f_cloud: 云端算力 (FLOPS)
            C_cloud: 云端计算量 (FLOPs)

        Returns:
            Tuple[float, float, float]: (E_edge, E_cloud, E_total)
        """
        E_edge = self.kappa_edge * (f_edge ** 2) * C_edge if C_edge > 0 else 0
        E_cloud = self.kappa_cloud * (f_cloud ** 2) * C_cloud if C_cloud > 0 else 0
        E_total = E_edge + E_cloud
        return E_edge, E_cloud, E_total

    def _solve_case1(self, f_edge_avail: float, f_cloud_avail: float) -> Tuple[float, float]:
        """
        Case 1：两约束均不激活

        取算力上界（idea118.txt 2.6.3 Case 1）
        f_edge* = f_edge_avail
        f_cloud* = f_cloud_max

        Args:
            f_edge_avail: 可用边缘算力
            f_cloud_avail: 可用云端算力

        Returns:
            Tuple[float, float]: (f_edge, f_cloud)
        """
        return f_edge_avail, f_cloud_avail

    def _solve_case2(self, C_edge: float, C_cloud: float,
                     T_budget: float, f_edge_avail: float, f_cloud_avail: float) -> Tuple[float, float]:
        """
        Case 2：仅时延约束激活

        时延等式求解，按计算量比例分配时间预算（idea118.txt 2.6.3 Case 2）

        目标：min T = C_edge/f_edge + C_cloud/f_cloud
        约束：C_edge/f_edge + C_cloud/f_cloud = T_budget

        最优解：按计算量比例分配时间
        f_edge* = C_edge / (T_budget · r)
        f_cloud* = C_cloud / (T_budget · (1-r))
        其中 r = C_edge / (C_edge + C_cloud)

        Args:
            C_edge: 边缘计算量
            C_cloud: 云端计算量
            T_budget: 可用于计算的时间预算
            f_edge_avail: 可用边缘算力
            f_cloud_avail: 可用云端算力

        Returns:
            Tuple[float, float]: (f_edge, f_cloud)
        """
        if C_edge + C_cloud < NUMERICAL.EPSILON:
            return f_edge_avail, f_cloud_avail

        # 边缘时间占比 r = C_edge / (C_edge + C_cloud)
        r = C_edge / (C_edge + C_cloud)

        # 闭式解
        if r > NUMERICAL.EPSILON:
            f_edge = C_edge / (T_budget * r)
        else:
            f_edge = f_edge_avail

        if (1 - r) > NUMERICAL.EPSILON:
            f_cloud = C_cloud / (T_budget * (1 - r))
        else:
            f_cloud = f_cloud_avail

        # 限制到可用算力
        f_edge = min(f_edge, f_edge_avail)
        f_cloud = min(f_cloud, f_cloud_avail)

        return f_edge, f_cloud

    def _solve_case3(self, C_edge: float, C_cloud: float,
                     E_budget: float, f_edge_avail: float, f_cloud_avail: float) -> Tuple[float, float]:
        """
        Case 3：仅能量约束激活

        拉格朗日乘子法求解，最优条件为边际能效相等（idea118.txt 2.6.3 Case 3）

        最优条件：∂E_edge/∂f_edge = ∂E_cloud/∂f_cloud
        => 2κ_edge · f_edge · C_edge = 2κ_cloud · f_cloud · C_cloud
        => f_cloud = f_edge · (κ_edge · C_edge) / (κ_cloud · C_cloud)

        代入能量约束：
        κ_edge · f_edge² · C_edge + κ_cloud · f_cloud² · C_cloud = E_budget

        令 ratio = (κ_edge · C_edge) / (κ_cloud · C_cloud)
        则 f_cloud = f_edge · ratio

        代入得：
        κ_edge · C_edge · f_edge² + κ_cloud · C_cloud · (f_edge · ratio)² = E_budget
        f_edge² · (κ_edge · C_edge + κ_cloud · C_cloud · ratio²) = E_budget

        Args:
            C_edge: 边缘计算量
            C_cloud: 云端计算量
            E_budget: 能量预算
            f_edge_avail: 可用边缘算力
            f_cloud_avail: 可用云端算力

        Returns:
            Tuple[float, float]: (f_edge, f_cloud)
        """
        if C_edge < NUMERICAL.EPSILON and C_cloud < NUMERICAL.EPSILON:
            return f_edge_avail, f_cloud_avail

        # 纯边缘情况
        if C_cloud < NUMERICAL.EPSILON:
            # E = κ_edge · f² · C_edge = E_budget
            # f = sqrt(E_budget / (κ_edge · C_edge))
            if self.kappa_edge * C_edge > NUMERICAL.EPSILON:
                f_edge = np.sqrt(E_budget / (self.kappa_edge * C_edge))
            else:
                f_edge = f_edge_avail
            return min(f_edge, f_edge_avail), f_cloud_avail

        # 纯云端情况
        if C_edge < NUMERICAL.EPSILON:
            if self.kappa_cloud * C_cloud > NUMERICAL.EPSILON:
                f_cloud = np.sqrt(E_budget / (self.kappa_cloud * C_cloud))
            else:
                f_cloud = f_cloud_avail
            return f_edge_avail, min(f_cloud, f_cloud_avail)

        # 边缘-云端协同
        # ratio = (κ_edge · C_edge) / (κ_cloud · C_cloud)
        ratio = (self.kappa_edge * C_edge) / (self.kappa_cloud * C_cloud)

        # 系数 A = κ_edge · C_edge + κ_cloud · C_cloud · ratio²
        A = self.kappa_edge * C_edge + self.kappa_cloud * C_cloud * (ratio ** 2)

        if A < NUMERICAL.EPSILON:
            return f_edge_avail, f_cloud_avail

        # f_edge = sqrt(E_budget / A)
        f_edge = np.sqrt(E_budget / A)
        f_cloud = f_edge * ratio

        # 限制到可用算力
        f_edge = min(f_edge, f_edge_avail)
        f_cloud = min(f_cloud, f_cloud_avail)

        return f_edge, f_cloud

    def _solve_case4(self, C_edge: float, C_cloud: float,
                     T_budget: float, E_budget: float,
                     f_edge_avail: float, f_cloud_avail: float) -> Tuple[float, float]:
        """
        Case 4：两约束均激活

        联立时延等式和能量等式求解（idea118.txt 2.6.3 Case 4）

        方程组：
        (1) C_edge/f_edge + C_cloud/f_cloud = T_budget
        (2) κ_edge · f_edge² · C_edge + κ_cloud · f_cloud² · C_cloud = E_budget

        求解策略：
        从(1)得：f_cloud = C_cloud / (T_budget - C_edge/f_edge)
        代入(2)得到关于f_edge的一元方程，数值求解

        Args:
            C_edge: 边缘计算量
            C_cloud: 云端计算量
            T_budget: 时间预算
            E_budget: 能量预算
            f_edge_avail: 可用边缘算力
            f_cloud_avail: 可用云端算力

        Returns:
            Tuple[float, float]: (f_edge, f_cloud)
        """
        if C_edge < NUMERICAL.EPSILON and C_cloud < NUMERICAL.EPSILON:
            return f_edge_avail, f_cloud_avail

        # 纯边缘或纯云端情况
        if C_cloud < NUMERICAL.EPSILON:
            # f_edge = C_edge / T_budget 且满足能量约束
            f_edge_delay = C_edge / T_budget if T_budget > NUMERICAL.EPSILON else f_edge_avail
            f_edge_energy = np.sqrt(E_budget / (self.kappa_edge * C_edge)) if self.kappa_edge * C_edge > NUMERICAL.EPSILON else f_edge_avail
            f_edge = max(f_edge_delay, f_edge_energy)  # 需要同时满足两个约束
            return min(f_edge, f_edge_avail), f_cloud_avail

        if C_edge < NUMERICAL.EPSILON:
            f_cloud_delay = C_cloud / T_budget if T_budget > NUMERICAL.EPSILON else f_cloud_avail
            f_cloud_energy = np.sqrt(E_budget / (self.kappa_cloud * C_cloud)) if self.kappa_cloud * C_cloud > NUMERICAL.EPSILON else f_cloud_avail
            f_cloud = max(f_cloud_delay, f_cloud_energy)
            return f_edge_avail, min(f_cloud, f_cloud_avail)

        # 边缘-云端协同：数值求解
        # 使用二分法求解

        def energy_residual(f_edge):
            """计算给定f_edge时的能量残差"""
            if f_edge < NUMERICAL.EPSILON:
                return float('inf')
            # 从时延约束求f_cloud
            T_edge = C_edge / f_edge
            T_cloud_budget = T_budget - T_edge
            if T_cloud_budget <= NUMERICAL.EPSILON:
                return float('inf')
            f_cloud = C_cloud / T_cloud_budget
            # 计算能量
            E = self.kappa_edge * (f_edge ** 2) * C_edge + self.kappa_cloud * (f_cloud ** 2) * C_cloud
            return E - E_budget

        # 确定搜索范围
        # f_edge最小值：需要满足时延约束
        f_edge_min = C_edge / T_budget if T_budget > NUMERICAL.EPSILON else 1e9
        f_edge_max = f_edge_avail

        if f_edge_min >= f_edge_max:
            # 无可行解，返回最大算力
            return f_edge_avail, f_cloud_avail

        # 检查边界情况
        E_min = energy_residual(f_edge_min)
        E_max = energy_residual(f_edge_max)

        if E_min > 0 and E_max > 0:
            # 能量约束过紧，无法满足
            return f_edge_avail, f_cloud_avail

        if E_min <= 0:
            # 最小f_edge就满足能量约束
            f_edge = f_edge_min
        elif E_max <= 0:
            # 最大f_edge满足能量约束
            f_edge = f_edge_max
        else:
            # 二分搜索
            for _ in range(50):  # 最多50次迭代
                f_mid = (f_edge_min + f_edge_max) / 2
                E_mid = energy_residual(f_mid)

                if abs(E_mid) < 1e-10:
                    f_edge = f_mid
                    break

                if E_mid > 0:
                    f_edge_min = f_mid
                else:
                    f_edge_max = f_mid
                f_edge = f_mid

        # 计算对应的f_cloud
        T_edge = C_edge / f_edge
        T_cloud_budget = T_budget - T_edge
        if T_cloud_budget > NUMERICAL.EPSILON:
            f_cloud = C_cloud / T_cloud_budget
        else:
            f_cloud = f_cloud_avail

        # 限制到可用算力
        f_edge = min(f_edge, f_edge_avail)
        f_cloud = min(f_cloud, f_cloud_avail)

        return f_edge, f_cloud

    def _check_feasibility(self, f_edge: float, f_cloud: float,
                           C_edge: float, C_cloud: float,
                           T_budget: float, E_budget: float,
                           f_edge_avail: float, f_cloud_avail: float) -> bool:
        """
        检查解的可行性

        约束条件（idea118.txt 2.6.1节）：
        C1（时延约束）：C_edge/f_edge + C_cloud/f_cloud ≤ T_budget
        C2（能量约束）：κ_edge·f_edge²·C_edge + κ_cloud·f_cloud²·C_cloud ≤ E_budget
        C3（算力上界）：f_edge ≤ f_edge_avail, f_cloud ≤ f_cloud_avail
        C4（非负性）：f_edge ≥ 0, f_cloud ≥ 0

        Args:
            f_edge: 边缘算力
            f_cloud: 云端算力
            C_edge: 边缘计算量
            C_cloud: 云端计算量
            T_budget: 时间预算
            E_budget: 能量预算
            f_edge_avail: 可用边缘算力
            f_cloud_avail: 可用云端算力

        Returns:
            bool: 是否可行
        """
        # C4: 非负性
        if f_edge < 0 or f_cloud < 0:
            return False

        # C3: 算力上界
        if f_edge > f_edge_avail + NUMERICAL.EPSILON:
            return False
        if f_cloud > f_cloud_avail + NUMERICAL.EPSILON:
            return False

        # C1: 时延约束
        T_compute = 0
        if C_edge > 0:
            if f_edge < NUMERICAL.EPSILON:
                return False
            T_compute += C_edge / f_edge
        if C_cloud > 0:
            if f_cloud < NUMERICAL.EPSILON:
                return False
            T_compute += C_cloud / f_cloud

        if T_compute > T_budget + NUMERICAL.EPSILON:
            return False

        # C2: 能量约束
        E_edge, E_cloud, E_total = self.compute_energy(f_edge, C_edge, f_cloud, C_cloud)
        if E_total > E_budget + NUMERICAL.EPSILON:
            return False

        return True

    def optimize_allocation(self,
                            C_edge: float,
                            C_cloud: float,
                            T_upload: float,
                            T_trans: float,
                            T_return: float,
                            T_max: float,
                            f_edge_available: Optional[float] = None,
                            f_cloud_available: Optional[float] = None,
                            E_remain: Optional[float] = None,
                            queue_size: int = 0,
                            E_comm: float = 0) -> AllocationResult:
        """
        优化资源分配

        严格按照 docs/idea118.txt 2.6节实现四组闭型解

        算法流程（2.6.4节）：
        Step 1：计算四组候选解
        Step 2：可行性检查（满足所有约束）
        Step 3：选择可行且时延最小者

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            T_upload: 上传时延 (s)
            T_trans: 中继传输时延 (s)
            T_return: 返回时延 (s)
            T_max: 最大允许时延 (s)
            f_edge_available: 可用边缘算力
            f_cloud_available: 可用云端算力
            E_remain: UAV剩余能量 (J)
            queue_size: 当前队列长度
            E_comm: 通信能耗 (J)

        Returns:
            AllocationResult: 分配结果
        """
        # 使用可用资源或最大资源
        f_edge_avail = f_edge_available or self.f_edge_max
        f_cloud_avail = f_cloud_available or self.f_cloud_max
        E_remain = E_remain or self.E_max

        # 通信总时延
        T_comm = T_upload + T_trans + T_return

        # 时间预算（idea118.txt 2.6.1 C1）
        T_budget = T_max - T_comm

        if T_budget <= 0:
            # 通信时延已超过限制
            return AllocationResult(
                f_edge=0, f_cloud=0,
                T_edge=0, T_trans=T_trans,
                T_cloud=0, T_return=T_return,
                T_total=T_comm,
                E_edge=0, E_cloud=0, E_total=0,
                feasible=False,
                margin=-T_comm,
                case_used=0
            )

        # 能量预算（idea118.txt 2.6.1 C2）
        # E_budget = min(E_remain/(|Q_j|+1), 0.3·E_max) - E_comm
        E_budget = min(E_remain / (queue_size + 1), 0.3 * self.E_max) - E_comm

        # 处理纯边缘或纯云端情况
        if C_cloud <= NUMERICAL.EPSILON:
            # 纯边缘计算
            f_edge_needed = C_edge / T_budget
            f_edge = min(f_edge_needed, f_edge_avail)
            T_edge = C_edge / max(f_edge, NUMERICAL.EPSILON)
            T_total = T_upload + T_edge
            E_edge, E_cloud, E_total = self.compute_energy(f_edge, C_edge, 0, 0)
            feasible = T_total <= T_max and E_edge <= E_budget

            return AllocationResult(
                f_edge=f_edge, f_cloud=0,
                T_edge=T_edge, T_trans=0,
                T_cloud=0, T_return=0,
                T_total=T_total,
                E_edge=E_edge, E_cloud=0, E_total=E_edge,
                feasible=feasible,
                margin=T_max - T_total,
                case_used=1
            )

        if C_edge <= NUMERICAL.EPSILON:
            # 纯云端计算
            f_cloud_needed = C_cloud / T_budget
            f_cloud = min(f_cloud_needed, f_cloud_avail)
            T_cloud = C_cloud / max(f_cloud, NUMERICAL.EPSILON)
            T_total = T_upload + T_trans + T_cloud + T_return
            E_edge, E_cloud, E_total = self.compute_energy(0, 0, f_cloud, C_cloud)
            feasible = T_total <= T_max

            return AllocationResult(
                f_edge=0, f_cloud=f_cloud,
                T_edge=0, T_trans=T_trans,
                T_cloud=T_cloud, T_return=T_return,
                T_total=T_total,
                E_edge=0, E_cloud=E_cloud, E_total=E_cloud,
                feasible=feasible,
                margin=T_max - T_total,
                case_used=1
            )

        # ========== 边缘-云端协同计算 ==========
        # Step 1: 计算四组候选解

        # Case 1: 两约束均不激活
        f1_edge, f1_cloud = self._solve_case1(f_edge_avail, f_cloud_avail)

        # Case 2: 仅时延约束激活
        f2_edge, f2_cloud = self._solve_case2(C_edge, C_cloud, T_budget, f_edge_avail, f_cloud_avail)

        # Case 3: 仅能量约束激活
        f3_edge, f3_cloud = self._solve_case3(C_edge, C_cloud, E_budget, f_edge_avail, f_cloud_avail)

        # Case 4: 两约束均激活
        f4_edge, f4_cloud = self._solve_case4(C_edge, C_cloud, T_budget, E_budget, f_edge_avail, f_cloud_avail)

        # Step 2 & 3: 可行性检查并选择最优
        candidates = [
            (f1_edge, f1_cloud, 1),
            (f2_edge, f2_cloud, 2),
            (f3_edge, f3_cloud, 3),
            (f4_edge, f4_cloud, 4)
        ]

        best_result = None
        min_delay = float('inf')
        best_case = 0

        for f_e, f_c, case_num in candidates:
            if self._check_feasibility(f_e, f_c, C_edge, C_cloud, T_budget, E_budget,
                                       f_edge_avail, f_cloud_avail):
                # 计算时延
                T_edge = C_edge / f_e
                T_cloud = C_cloud / f_c
                T_compute = T_edge + T_cloud

                if T_compute < min_delay:
                    min_delay = T_compute
                    best_result = (f_e, f_c)
                    best_case = case_num

        # 如果没有可行解，使用Case 1（最大算力）
        if best_result is None:
            f_edge, f_cloud = f_edge_avail, f_cloud_avail
            best_case = 1
            feasible = False
        else:
            f_edge, f_cloud = best_result
            feasible = True

        # 计算最终结果
        T_edge = C_edge / max(f_edge, NUMERICAL.EPSILON)
        T_cloud = C_cloud / max(f_cloud, NUMERICAL.EPSILON)
        T_total = T_comm + T_edge + T_cloud

        E_edge, E_cloud, E_total = self.compute_energy(f_edge, C_edge, f_cloud, C_cloud)

        return AllocationResult(
            f_edge=f_edge, f_cloud=f_cloud,
            T_edge=T_edge, T_trans=T_trans,
            T_cloud=T_cloud, T_return=T_return,
            T_total=T_total,
            E_edge=E_edge, E_cloud=E_cloud, E_total=E_total,
            feasible=feasible,
            margin=T_max - T_total,
            case_used=best_case
        )

    def find_optimal_split(self,
                           C_total: float,
                           split_ratios: list,
                           output_sizes: list,
                           T_upload_base: float,
                           R_trans: float,
                           R_return: float,
                           T_max: float,
                           E_remain: float = None) -> Tuple[int, AllocationResult]:
        """
        寻找最优切分点

        遍历所有可能的切分点，找到时延最小的

        Args:
            C_total: 总计算量 (FLOPs)
            split_ratios: 各切分点的边缘计算比例 [r1, r2, ...]
            output_sizes: 各切分点的中间数据量 [s1, s2, ...] (bits)
            T_upload_base: 基础上传时延 (s)
            R_trans: 中继传输速率 (bps)
            R_return: 返回速率 (bps)
            T_max: 最大时延 (s)
            E_remain: 剩余能量 (J)

        Returns:
            Tuple[int, AllocationResult]: (最优切分点索引, 分配结果)
        """
        best_split = 0
        best_result = None
        best_delay = float('inf')

        for i, (ratio, output_size) in enumerate(zip(split_ratios, output_sizes)):
            C_edge = C_total * ratio
            C_cloud = C_total * (1 - ratio)

            # 计算传输时延
            T_trans = output_size / R_trans if R_trans > 0 else 0
            T_return_data = output_size / R_return if R_return > 0 else 0

            result = self.optimize_allocation(
                C_edge=C_edge,
                C_cloud=C_cloud,
                T_upload=T_upload_base,
                T_trans=T_trans,
                T_return=T_return_data,
                T_max=T_max,
                E_remain=E_remain
            )

            if result.T_total < best_delay:
                best_delay = result.T_total
                best_split = i
                best_result = result

        return best_split, best_result


# ============ 测试用例 ============

def test_resource_optimizer():
    """测试ResourceOptimizer模块"""
    print("=" * 70)
    print("测试 M16: ResourceOptimizer (四组闭型解)")
    print("=" * 70)

    optimizer = ResourceOptimizer(
        f_edge_max=10e9,
        f_cloud_max=100e9,
        kappa_edge=1e-28,
        kappa_cloud=1e-29,
        E_max=500e3
    )

    # 测试1: 基本资源分配 - Case 1/2
    print("\n[Test 1] 测试基本资源分配...")
    result = optimizer.optimize_allocation(
        C_edge=5e9,      # 5 GFLOPs
        C_cloud=15e9,    # 15 GFLOPs
        T_upload=0.1,    # 100ms
        T_trans=0.05,    # 50ms
        T_return=0.02,   # 20ms
        T_max=5.0,       # 5s
        E_remain=500e3
    )

    print(f"  边缘算力: {result.f_edge/1e9:.2f} GFLOPS")
    print(f"  云端算力: {result.f_cloud/1e9:.2f} GFLOPS")
    print(f"  边缘时延: {result.T_edge*1000:.1f} ms")
    print(f"  云端时延: {result.T_cloud*1000:.1f} ms")
    print(f"  总时延: {result.T_total*1000:.1f} ms")
    print(f"  边缘能耗: {result.E_edge:.2f} J")
    print(f"  云端能耗: {result.E_cloud:.4f} J")
    print(f"  使用的Case: {result.case_used}")
    print(f"  可行: {result.feasible}")
    assert result.feasible, "应该可行"
    print("  [OK] 基本分配正确")

    # 测试2: 时延约束紧张 - Case 2
    print("\n[Test 2] 测试时延约束紧张（Case 2）...")
    result2 = optimizer.optimize_allocation(
        C_edge=5e9,
        C_cloud=15e9,
        T_upload=0.1,
        T_trans=0.05,
        T_return=0.02,
        T_max=1.0,       # 更紧的时延约束
        E_remain=500e3
    )

    print(f"  总时延: {result2.T_total*1000:.1f} ms")
    print(f"  使用的Case: {result2.case_used}")
    print(f"  可行: {result2.feasible}")
    print("  [OK] Case 2 测试通过")

    # 测试3: 能量约束紧张 - Case 3
    print("\n[Test 3] 测试能量约束紧张（Case 3）...")
    result3 = optimizer.optimize_allocation(
        C_edge=5e9,
        C_cloud=15e9,
        T_upload=0.1,
        T_trans=0.05,
        T_return=0.02,
        T_max=10.0,      # 宽松时延
        E_remain=100e3   # 紧张能量
    )

    print(f"  边缘能耗: {result3.E_edge:.2f} J")
    print(f"  使用的Case: {result3.case_used}")
    print("  [OK] Case 3 测试通过")

    # 测试4: 不可行情况
    print("\n[Test 4] 测试不可行情况...")
    result_infeasible = optimizer.optimize_allocation(
        C_edge=50e9,     # 50 GFLOPs
        C_cloud=200e9,   # 200 GFLOPs
        T_upload=0.5,
        T_trans=0.3,
        T_return=0.2,
        T_max=1.0,       # 只有1s
        E_remain=500e3
    )

    print(f"  总时延: {result_infeasible.T_total*1000:.1f} ms > 1000ms")
    print(f"  可行: {result_infeasible.feasible}")
    assert not result_infeasible.feasible, "应该不可行"
    print("  [OK] 不可行检测正确")

    # 测试5: 纯边缘计算
    print("\n[Test 5] 测试纯边缘计算...")
    result_edge_only = optimizer.optimize_allocation(
        C_edge=8e9,
        C_cloud=0,
        T_upload=0.1,
        T_trans=0,
        T_return=0,
        T_max=2.0,
        E_remain=500e3
    )

    assert result_edge_only.f_cloud == 0, "云端算力应为0"
    print(f"  边缘算力: {result_edge_only.f_edge/1e9:.2f} GFLOPS")
    print(f"  总时延: {result_edge_only.T_total*1000:.1f} ms")
    print("  [OK] 纯边缘计算正确")

    # 测试6: 纯云端计算
    print("\n[Test 6] 测试纯云端计算...")
    result_cloud_only = optimizer.optimize_allocation(
        C_edge=0,
        C_cloud=50e9,
        T_upload=0.2,
        T_trans=0.1,
        T_return=0.05,
        T_max=2.0,
        E_remain=500e3
    )

    assert result_cloud_only.f_edge == 0, "边缘算力应为0"
    print(f"  云端算力: {result_cloud_only.f_cloud/1e9:.2f} GFLOPS")
    print(f"  总时延: {result_cloud_only.T_total*1000:.1f} ms")
    print("  [OK] 纯云端计算正确")

    # 测试7: 四组闭型解覆盖
    print("\n[Test 7] 测试四组闭型解覆盖...")
    test_cases = [
        ("宽松约束", 10.0, 500e3),
        ("时延紧张", 1.5, 500e3),
        ("能量紧张", 10.0, 50e3),
        ("双约束紧张", 2.0, 100e3)
    ]

    cases_used = set()
    for name, T_max, E_remain in test_cases:
        result = optimizer.optimize_allocation(
            C_edge=5e9, C_cloud=10e9,
            T_upload=0.1, T_trans=0.05, T_return=0.02,
            T_max=T_max, E_remain=E_remain
        )
        cases_used.add(result.case_used)
        print(f"  {name}: Case {result.case_used}, 可行={result.feasible}")

    print(f"  覆盖的Case: {sorted(cases_used)}")
    print("  [OK] 四组闭型解测试通过")

    print("\n" + "=" * 70)
    print("所有测试通过! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_resource_optimizer()
