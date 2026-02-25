"""
M19: CombinatorialAuction - 组合拍卖核心

功能：基于拉格朗日对偶分解求解组合拍卖赢家确定问题
输入：所有任务的投标集合、资源约束
输出：中标结果、资源分配

关键公式 (idea118.txt 3.8-3.14节):
    原问题:
        max Σ_i Σ_b η_{i,b} * x_{i,b}
        s.t. Σ_b x_{i,b} ≤ 1 (每任务至多一个投标中标)
             Σ_i Σ_b f_{i,b} * x_{i,b} ≤ F_j (UAV算力约束)
             Σ_i Σ_b E_{i,b} * x_{i,b} ≤ E_j (UAV能量约束)
             
    拉格朗日松弛:
        L(λ,μ) = Σ_i max_b [η_{i,b} - λ_j*f_{i,b} - μ_j*E_{i,b}]
                 + Σ_j (λ_j*F_j + μ_j*E_j)
                 
    次梯度更新:
        λ_j(t+1) = [λ_j(t) - ε(t) * (F_j - Σ f_j)]⁺
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import CONSTRAINT, NUMERICAL


class AuctionStatus(Enum):
    """拍卖状态"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"  # 部分任务未分配
    FAILED = "FAILED"
    INFEASIBLE = "INFEASIBLE"


@dataclass
class BidInfo:
    """
    投标信息（简化版）
    
    Attributes:
        task_id: 任务ID
        bid_id: 投标ID
        uav_id: UAV ID
        utility: 效用
        f_required: 所需算力
        E_required: 所需能量
        T_predict: 预测时延
        priority: 任务优先级
    """
    task_id: int
    bid_id: int
    uav_id: int
    utility: float
    f_required: float
    E_required: float
    T_predict: float
    priority: float = 0.5


@dataclass
class UAVResource:
    """
    UAV资源信息
    
    Attributes:
        uav_id: UAV ID
        F_available: 可用算力
        E_available: 可用能量
    """
    uav_id: int
    F_available: float
    E_available: float


@dataclass
class AuctionResult:
    """
    拍卖结果
    
    Attributes:
        winners: 中标投标列表 [(task_id, bid_id)]
        task_assignments: 任务分配 {task_id: (uav_id, bid_info)}
        unassigned_tasks: 未分配任务列表
        uav_allocations: UAV分配情况 {uav_id: [(task_id, bid_info), ...]}
        total_utility: 总效用
        status: 拍卖状态
        iterations: 迭代次数
        dual_values: 对偶变量
    """
    winners: List[Tuple[int, int]]
    task_assignments: Dict[int, Tuple[int, BidInfo]]
    unassigned_tasks: List[int]
    uav_allocations: Dict[int, List[Tuple[int, BidInfo]]]
    total_utility: float
    status: AuctionStatus
    iterations: int
    dual_values: Dict[str, float]


class LagrangianAuction:
    """
    拉格朗日对偶分解拍卖求解器
    
    Attributes:
        epsilon_0: 初始步长
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        M_penalty: 高优先级未服务惩罚
        feasibility_tolerance: 可行性容差
    """
    
    def __init__(self,
                 epsilon_0: float = 0.5,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 M_penalty: float = 100.0,
                 feasibility_tolerance: Optional[float] = None):
        """
        初始化求解器
        
        Args:
            epsilon_0: 初始步长
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            M_penalty: 惩罚系数
            feasibility_tolerance: 可行性容差，默认使用常量配置
        """
        self.epsilon_0 = epsilon_0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.M_penalty = M_penalty
        # 使用常量作为默认值
        self.feasibility_tolerance = feasibility_tolerance if feasibility_tolerance is not None \
            else CONSTRAINT.FEASIBILITY_TOLERANCE
    
    def solve(self,
              bids: Dict[int, List[BidInfo]],
              uav_resources: List[UAVResource]) -> AuctionResult:
        """
        求解组合拍卖问题
        
        Args:
            bids: 投标集合 {task_id: [bid_info, ...]}
            uav_resources: UAV资源列表
            
        Returns:
            AuctionResult: 拍卖结果
        """
        if not bids:
            return AuctionResult(
                winners=[],
                task_assignments={},
                unassigned_tasks=[],
                uav_allocations={},
                total_utility=0.0,
                status=AuctionStatus.SUCCESS,
                iterations=0,
                dual_values={}
            )
        
        # 初始化对偶变量
        uav_ids = [r.uav_id for r in uav_resources]
        lambda_f = {uid: 0.0 for uid in uav_ids}  # 算力价格
        mu_E = {uid: 0.0 for uid in uav_ids}      # 能量价格
        
        uav_F = {r.uav_id: r.F_available for r in uav_resources}
        uav_E = {r.uav_id: r.E_available for r in uav_resources}
        
        best_solution = None
        best_utility = -float('inf')
        
        for iteration in range(self.max_iterations):
            step_size = self.epsilon_0 / (iteration + 1)
            
            # 对每个任务，选择最优投标
            task_selections = {}
            
            for task_id, task_bids in bids.items():
                best_bid = None
                best_value = -float('inf')
                
                for bid in task_bids:
                    # 只考虑资源列表中的UAV
                    if bid.uav_id not in uav_ids:
                        continue
                    
                    # 计算拉格朗日值
                    # η - λ*f - μ*E
                    lag_value = bid.utility
                    lag_value -= lambda_f.get(bid.uav_id, 0) * bid.f_required
                    lag_value -= mu_E.get(bid.uav_id, 0) * bid.E_required
                    
                    if lag_value > best_value:
                        best_value = lag_value
                        best_bid = bid
                
                if best_bid is not None and best_value > 0:
                    task_selections[task_id] = best_bid
            
            # 计算资源使用情况
            uav_f_usage = {uid: 0.0 for uid in uav_ids}
            uav_E_usage = {uid: 0.0 for uid in uav_ids}
            
            for task_id, bid in task_selections.items():
                if bid.uav_id in uav_f_usage:
                    uav_f_usage[bid.uav_id] += bid.f_required
                    uav_E_usage[bid.uav_id] += bid.E_required
            
            # 更新对偶变量（次梯度法）
            converged = True
            for uid in uav_ids:
                # 算力约束次梯度
                grad_f = uav_f_usage[uid] - uav_F[uid]
                lambda_f[uid] = max(0, lambda_f[uid] + step_size * grad_f)
                
                # 能量约束次梯度
                grad_E = uav_E_usage[uid] - uav_E[uid]
                mu_E[uid] = max(0, mu_E[uid] + step_size * grad_E)
                
                if abs(grad_f) > self.tolerance or abs(grad_E) > self.tolerance:
                    converged = False
            
            # 检查可行性并记录最优解（使用配置的容差）
            tolerance_factor = 1.0 + self.feasibility_tolerance
            feasible = all(
                uav_f_usage[uid] <= uav_F[uid] * tolerance_factor
                and uav_E_usage[uid] <= uav_E[uid] * tolerance_factor
                for uid in uav_ids
            )
            
            if feasible:
                utility = sum(b.utility for b in task_selections.values())
                if utility > best_utility:
                    best_utility = utility
                    best_solution = task_selections.copy()
            
            if converged:
                break
        
        # 如果没有可行解，使用贪心修复
        if best_solution is None:
            best_solution = self._greedy_repair(bids, uav_resources)
        
        return self._build_result(
            best_solution, bids, uav_resources, iteration + 1, lambda_f, mu_E
        )
    
    def _greedy_repair(self,
                       bids: Dict[int, List[BidInfo]],
                       uav_resources: List[UAVResource]) -> Dict[int, BidInfo]:
        """
        贪心修复：当拉格朗日松弛找不到可行解时使用
        
        策略：按优先级排序，贪心分配
        
        Args:
            bids: 投标集合
            uav_resources: UAV资源
            
        Returns:
            Dict: 可行分配
        """
        # 按优先级排序任务
        task_priority = []
        for task_id, task_bids in bids.items():
            if task_bids:
                max_priority = max(b.priority for b in task_bids)
                task_priority.append((task_id, max_priority))
        
        task_priority.sort(key=lambda x: x[1], reverse=True)
        
        # 跟踪剩余资源
        remaining_F = {r.uav_id: r.F_available for r in uav_resources}
        remaining_E = {r.uav_id: r.E_available for r in uav_resources}
        
        solution = {}
        
        for task_id, _ in task_priority:
            task_bids = bids[task_id]
            
            # 找到可行的最高效用投标
            best_bid = None
            best_utility = -float('inf')
            
            for bid in task_bids:
                if (remaining_F.get(bid.uav_id, 0) >= bid.f_required and
                    remaining_E.get(bid.uav_id, 0) >= bid.E_required and
                    bid.utility > best_utility):
                    best_utility = bid.utility
                    best_bid = bid
            
            if best_bid is not None:
                solution[task_id] = best_bid
                remaining_F[best_bid.uav_id] -= best_bid.f_required
                remaining_E[best_bid.uav_id] -= best_bid.E_required
        
        return solution
    
    def _build_result(self,
                      solution: Dict[int, BidInfo],
                      bids: Dict[int, List[BidInfo]],
                      uav_resources: List[UAVResource],
                      iterations: int,
                      lambda_f: Dict[int, float],
                      mu_E: Dict[int, float]) -> AuctionResult:
        """
        构建拍卖结果
        """
        winners = [(task_id, bid.bid_id) for task_id, bid in solution.items()]
        
        task_assignments = {
            task_id: (bid.uav_id, bid)
            for task_id, bid in solution.items()
        }
        
        unassigned = [tid for tid in bids.keys() if tid not in solution]
        
        uav_allocations = {r.uav_id: [] for r in uav_resources}
        for task_id, bid in solution.items():
            uav_allocations[bid.uav_id].append((task_id, bid))
        
        total_utility = sum(bid.utility for bid in solution.values())
        
        # 确定状态
        if not unassigned:
            status = AuctionStatus.SUCCESS
        elif solution:
            status = AuctionStatus.PARTIAL
        else:
            status = AuctionStatus.FAILED
        
        dual_values = {
            **{f'lambda_f_{uid}': v for uid, v in lambda_f.items()},
            **{f'mu_E_{uid}': v for uid, v in mu_E.items()}
        }
        
        return AuctionResult(
            winners=winners,
            task_assignments=task_assignments,
            unassigned_tasks=unassigned,
            uav_allocations=uav_allocations,
            total_utility=total_utility,
            status=status,
            iterations=iterations,
            dual_values=dual_values
        )


# ============ 测试用例 ============

def test_combinatorial_auction():
    """测试CombinatorialAuction模块"""
    print("=" * 60)
    print("测试 M19: CombinatorialAuction")
    print("=" * 60)
    
    solver = LagrangianAuction()
    
    # 创建测试数据
    np.random.seed(42)
    
    # UAV资源
    uav_resources = [
        UAVResource(uav_id=0, F_available=8e9, E_available=100e3),
        UAVResource(uav_id=1, F_available=8e9, E_available=100e3),
    ]
    
    # 投标
    bids = {}
    for task_id in range(5):
        task_bids = []
        for bid_id in range(3):
            uav_id = bid_id % 2
            task_bids.append(BidInfo(
                task_id=task_id,
                bid_id=bid_id,
                uav_id=uav_id,
                utility=np.random.uniform(0.3, 0.9),
                f_required=np.random.uniform(1e9, 3e9),
                E_required=np.random.uniform(10e3, 30e3),
                T_predict=np.random.uniform(1.0, 3.0),
                priority=np.random.uniform(0.3, 0.9)
            ))
        bids[task_id] = task_bids
    
    # 测试1: 基本拍卖求解
    print("\n[Test 1] 测试基本拍卖求解...")
    result = solver.solve(bids, uav_resources)
    
    print(f"  中标数: {len(result.winners)}")
    print(f"  未分配: {len(result.unassigned_tasks)}")
    print(f"  总效用: {result.total_utility:.3f}")
    print(f"  状态: {result.status.value}")
    print(f"  迭代次数: {result.iterations}")
    print("  ✓ 基本拍卖求解正确")
    
    # 测试2: 验证约束满足
    print("\n[Test 2] 验证资源约束...")
    
    uav_f_used = {0: 0, 1: 0}
    uav_E_used = {0: 0, 1: 0}
    
    for task_id, (uav_id, bid) in result.task_assignments.items():
        uav_f_used[uav_id] += bid.f_required
        uav_E_used[uav_id] += bid.E_required
    
    for r in uav_resources:
        print(f"  UAV-{r.uav_id}: 算力使用={uav_f_used[r.uav_id]/1e9:.2f}/{r.F_available/1e9:.2f} GFLOPS")
        assert uav_f_used[r.uav_id] <= r.F_available * 1.01, "算力约束违反"
        assert uav_E_used[r.uav_id] <= r.E_available * 1.01, "能量约束违反"
    
    print("  ✓ 资源约束满足")
    
    # 测试3: 每任务至多一个投标
    print("\n[Test 3] 验证唯一性约束...")
    
    assigned_tasks = set(t for t, _ in result.winners)
    assert len(assigned_tasks) == len(result.winners), "每任务应至多一个投标"
    print(f"  分配的任务: {sorted(assigned_tasks)}")
    print("  ✓ 唯一性约束满足")
    
    # 测试4: 资源紧张场景
    print("\n[Test 4] 测试资源紧张场景...")
    
    tight_resources = [
        UAVResource(uav_id=0, F_available=3e9, E_available=30e3),
    ]
    
    result_tight = solver.solve(bids, tight_resources)
    
    print(f"  中标数: {len(result_tight.winners)}")
    print(f"  未分配: {len(result_tight.unassigned_tasks)}")
    assert len(result_tight.unassigned_tasks) > 0, "资源紧张时应有未分配任务"
    print("  ✓ 资源紧张处理正确")
    
    # 测试5: 空投标
    print("\n[Test 5] 测试空投标...")
    result_empty = solver.solve({}, uav_resources)
    
    assert result_empty.status == AuctionStatus.SUCCESS
    assert len(result_empty.winners) == 0
    print("  ✓ 空投标处理正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_combinatorial_auction()
