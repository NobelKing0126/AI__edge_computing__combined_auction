"""
离线最优求解器与竞争比计算器

用于:
1. 求解离线最优社会福利 (ILP/LP松弛)
2. 计算在线算法的竞争比

竞争比定义:
    ρ = SW* / SW_online
    
其中:
    SW* = 离线最优社会福利（Oracle知道所有任务）
    SW_online = 在线算法社会福利
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class BidInfo:
    """投标信息"""
    user_id: int
    layer: int          # 切分层
    uav_id: int         # 目标UAV
    utility: float      # 效用值 η_final
    f_edge: float       # 边缘算力需求
    f_cloud: float      # 云端算力需求
    energy: float       # 能量消耗
    priority_class: str # 优先级类别: high/medium/low
    

class OfflineOptimalSolver:
    """
    离线最优问题求解器
    
    问题形式:
        max  Σ η_{i,l,j} * x_{i,l,j}
        s.t.
            Σ_{l,j} x_{i,l,j} = 1,  ∀i ∈ U_high       (高优先级必须服务)
            Σ_{l,j} x_{i,l,j} ≤ 1,  ∀i ∈ U_med ∪ U_low (其他可拒绝)
            Σ_{i,l} f_{i,j} * x_{i,l,j} ≤ f_j^avail, ∀j (UAV算力约束)
            Σ_{i,l} E_{i,l,j} * x_{i,l,j} ≤ E_j^avail, ∀j (UAV能量约束)
            Σ_{i,l,j} f_{c,i} * x_{i,l,j} ≤ F_c       (云端算力约束)
            x_{i,l,j} ∈ {0, 1}
    """
    
    def __init__(self, n_uavs: int = 5, n_users: int = 50):
        self.n_uavs = n_uavs
        self.n_users = n_users
        
        # 资源约束
        self.uav_compute_max = {}   # {uav_id: f_max}
        self.uav_energy_max = {}    # {uav_id: E_max}
        self.cloud_compute_max = 100e9
        
    def set_resources(self, uav_resources: List[Dict], cloud_resources: Dict):
        """设置资源约束"""
        for uav in uav_resources:
            uav_id = uav.get('uav_id', 0)
            self.uav_compute_max[uav_id] = uav.get('f_max', 15e9)
            self.uav_energy_max[uav_id] = uav.get('E_max', 500e3)  # 500kJ
        
        self.cloud_compute_max = cloud_resources.get('f_cloud', 100e9)
    
    def solve(self, bids: List[BidInfo], use_lp_relaxation: bool = True) -> Tuple[float, Dict]:
        """
        求解离线最优问题
        
        Args:
            bids: 所有投标列表
            use_lp_relaxation: 是否使用LP松弛（更快但可能不精确）
            
        Returns:
            (SW*, 最优分配详情)
        """
        if not SCIPY_AVAILABLE:
            return self._solve_greedy(bids)
        
        if use_lp_relaxation:
            return self._solve_lp_relaxation(bids)
        else:
            return self._solve_ilp(bids)
    
    def _solve_lp_relaxation(self, bids: List[BidInfo]) -> Tuple[float, Dict]:
        """
        使用LP松弛求解
        
        LP松弛将 x ∈ {0,1} 放松为 x ∈ [0,1]
        得到的是最优社会福利的上界
        """
        if not bids:
            return 0.0, {}
        
        n_vars = len(bids)
        
        # 目标函数: max Σ η * x  => min -Σ η * x
        c = np.array([-b.utility for b in bids])
        
        # 构建约束
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # 约束1: 用户唯一性约束
        # 按用户分组
        user_bids = {}
        for idx, bid in enumerate(bids):
            if bid.user_id not in user_bids:
                user_bids[bid.user_id] = []
            user_bids[bid.user_id].append(idx)
        
        for user_id, bid_indices in user_bids.items():
            row = np.zeros(n_vars)
            for idx in bid_indices:
                row[idx] = 1.0
            
            # 检查优先级
            priority_class = bids[bid_indices[0]].priority_class if bid_indices else 'medium'
            if priority_class == 'high':
                # 等式约束: Σ x = 1
                A_eq.append(row)
                b_eq.append(1.0)
            else:
                # 不等式约束: Σ x ≤ 1
                A_ub.append(row)
                b_ub.append(1.0)
        
        # 约束2: UAV算力约束
        for uav_id in range(self.n_uavs):
            row = np.zeros(n_vars)
            for idx, bid in enumerate(bids):
                if bid.uav_id == uav_id:
                    row[idx] = bid.f_edge
            
            f_max = self.uav_compute_max.get(uav_id, 15e9)
            A_ub.append(row)
            b_ub.append(f_max)
        
        # 约束3: UAV能量约束
        for uav_id in range(self.n_uavs):
            row = np.zeros(n_vars)
            for idx, bid in enumerate(bids):
                if bid.uav_id == uav_id:
                    row[idx] = bid.energy
            
            E_max = self.uav_energy_max.get(uav_id, 500e3)
            A_ub.append(row)
            b_ub.append(E_max)
        
        # 约束4: 云端算力约束
        row = np.zeros(n_vars)
        for idx, bid in enumerate(bids):
            row[idx] = bid.f_cloud
        A_ub.append(row)
        b_ub.append(self.cloud_compute_max)
        
        # 边界: 0 ≤ x ≤ 1
        bounds = [(0, 1) for _ in range(n_vars)]
        
        # 转换为numpy数组
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
            
            if result.success:
                sw_optimal = -result.fun
                solution = result.x
                
                # 整理结果
                allocation = {}
                for idx, x_val in enumerate(solution):
                    if x_val > 0.01:  # 阈值过滤
                        bid = bids[idx]
                        allocation[(bid.user_id, bid.layer, bid.uav_id)] = {
                            'x': x_val,
                            'utility': bid.utility,
                            'contribution': x_val * bid.utility
                        }
                
                return sw_optimal, allocation
            else:
                return self._solve_greedy(bids)
                
        except Exception as e:
            print(f"LP求解异常: {e}")
            return self._solve_greedy(bids)
    
    def _solve_ilp(self, bids: List[BidInfo]) -> Tuple[float, Dict]:
        """
        使用ILP精确求解（小规模问题）
        
        需要scipy >= 1.9
        """
        if not bids:
            return 0.0, {}
        
        try:
            n_vars = len(bids)
            
            # 目标函数
            c = np.array([-b.utility for b in bids])
            
            # 构建约束
            constraints = []
            
            # 用户唯一性约束
            user_bids = {}
            for idx, bid in enumerate(bids):
                if bid.user_id not in user_bids:
                    user_bids[bid.user_id] = []
                user_bids[bid.user_id].append(idx)
            
            for user_id, bid_indices in user_bids.items():
                A_row = np.zeros(n_vars)
                for idx in bid_indices:
                    A_row[idx] = 1.0
                
                priority_class = bids[bid_indices[0]].priority_class if bid_indices else 'medium'
                if priority_class == 'high':
                    constraints.append(LinearConstraint(A_row, lb=1, ub=1))
                else:
                    constraints.append(LinearConstraint(A_row, lb=0, ub=1))
            
            # UAV算力约束
            for uav_id in range(self.n_uavs):
                A_row = np.zeros(n_vars)
                for idx, bid in enumerate(bids):
                    if bid.uav_id == uav_id:
                        A_row[idx] = bid.f_edge
                
                f_max = self.uav_compute_max.get(uav_id, 15e9)
                constraints.append(LinearConstraint(A_row, lb=0, ub=f_max))
            
            # 云端算力约束
            A_row = np.zeros(n_vars)
            for idx, bid in enumerate(bids):
                A_row[idx] = bid.f_cloud
            constraints.append(LinearConstraint(A_row, lb=0, ub=self.cloud_compute_max))
            
            # 边界和整数约束
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n_vars)
            
            result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
            
            if result.success:
                sw_optimal = -result.fun
                return sw_optimal, {'solution': result.x}
            else:
                return self._solve_greedy(bids)
                
        except Exception as e:
            print(f"ILP求解异常: {e}, 回退到贪心")
            return self._solve_greedy(bids)
    
    def _solve_greedy(self, bids: List[BidInfo]) -> Tuple[float, Dict]:
        """
        贪心求解（作为备选方案）
        
        按效用密度排序，贪心分配
        """
        if not bids:
            return 0.0, {}
        
        # 按效用降序排序
        sorted_bids = sorted(bids, key=lambda b: b.utility, reverse=True)
        
        # 资源跟踪
        uav_compute_used = {i: 0.0 for i in range(self.n_uavs)}
        uav_energy_used = {i: 0.0 for i in range(self.n_uavs)}
        cloud_used = 0.0
        user_assigned = set()
        
        sw_total = 0.0
        allocation = {}
        
        for bid in sorted_bids:
            # 检查用户是否已分配
            if bid.user_id in user_assigned:
                continue
            
            # 检查资源约束
            uav_id = bid.uav_id
            f_max = self.uav_compute_max.get(uav_id, 15e9)
            E_max = self.uav_energy_max.get(uav_id, 500e3)
            
            if (uav_compute_used[uav_id] + bid.f_edge <= f_max and
                uav_energy_used[uav_id] + bid.energy <= E_max and
                cloud_used + bid.f_cloud <= self.cloud_compute_max):
                
                # 分配
                uav_compute_used[uav_id] += bid.f_edge
                uav_energy_used[uav_id] += bid.energy
                cloud_used += bid.f_cloud
                user_assigned.add(bid.user_id)
                
                sw_total += bid.utility
                allocation[(bid.user_id, bid.layer, bid.uav_id)] = {
                    'x': 1.0,
                    'utility': bid.utility,
                    'contribution': bid.utility
                }
        
        return sw_total, allocation


class CompetitiveRatioCalculator:
    """
    竞争比计算器
    
    竞争比 ρ = SW* / SW_online
    
    理论界 (基于对偶间隙):
        ρ ≤ 1 + Gap / SW_online
    """
    
    def __init__(self):
        self.offline_solver = OfflineOptimalSolver()
    
    def set_resources(self, uav_resources: List[Dict], cloud_resources: Dict):
        """设置资源约束"""
        self.offline_solver.set_resources(uav_resources, cloud_resources)
    
    def compute(self, 
                bids: List[BidInfo],
                sw_online: float,
                sw_dual: float = None) -> Dict[str, float]:
        """
        计算竞争比
        
        Args:
            bids: 所有投标列表
            sw_online: 在线算法社会福利
            sw_dual: 对偶问题最优值 (可选)
            
        Returns:
            {
                'sw_optimal': 离线最优社会福利,
                'sw_online': 在线算法社会福利,
                'actual_ratio': 实际竞争比,
                'dual_gap': 对偶间隙,
                'theoretical_bound': 理论竞争比上界,
                'gap_percentage': 间隙百分比
            }
        """
        # 求解离线最优
        sw_optimal, _ = self.offline_solver.solve(bids, use_lp_relaxation=True)
        
        # 计算实际竞争比
        if sw_online > 0:
            actual_ratio = sw_optimal / sw_online
        else:
            actual_ratio = float('inf')
        
        # 计算对偶间隙
        if sw_dual is not None:
            dual_gap = sw_dual - sw_online
        else:
            dual_gap = sw_optimal - sw_online
        
        # 理论竞争比上界
        if sw_online > 0:
            theoretical_bound = 1 + dual_gap / sw_online
        else:
            theoretical_bound = float('inf')
        
        # 间隙百分比
        if sw_optimal > 0:
            gap_percentage = (sw_optimal - sw_online) / sw_optimal * 100
        else:
            gap_percentage = 0.0
        
        return {
            'sw_optimal': sw_optimal,
            'sw_online': sw_online,
            'actual_ratio': actual_ratio,
            'dual_gap': dual_gap,
            'theoretical_bound': theoretical_bound,
            'gap_percentage': gap_percentage
        }
    
    def compute_from_tasks(self,
                           tasks: List[Dict],
                           results: List[Dict],
                           n_uavs: int = 5) -> Dict[str, float]:
        """
        从任务和结果直接计算竞争比
        
        Args:
            tasks: 任务列表
            results: 执行结果列表
            n_uavs: UAV数量
            
        Returns:
            竞争比信息字典
        """
        # 构建投标
        bids = []
        for i, (task, result) in enumerate(zip(tasks, results)):
            if result.get('success', False):
                for uav_id in range(n_uavs):
                    # 为每个UAV生成候选投标
                    utility = result.get('utility', task.get('priority', 0.5))
                    f_edge = result.get('compute_edge', task.get('compute', 1e9))
                    f_cloud = result.get('compute_cloud', 0)
                    energy = result.get('energy', 100)
                    
                    priority = task.get('priority', 0.5)
                    if priority >= 0.7:
                        priority_class = 'high'
                    elif priority <= 0.3:
                        priority_class = 'low'
                    else:
                        priority_class = 'medium'
                    
                    bids.append(BidInfo(
                        user_id=i,
                        layer=result.get('split_layer', 0),
                        uav_id=result.get('uav_id', uav_id),
                        utility=utility,
                        f_edge=f_edge,
                        f_cloud=f_cloud,
                        energy=energy,
                        priority_class=priority_class
                    ))
        
        # 计算在线社会福利
        sw_online = sum(r.get('utility', 0) for r in results if r.get('success', False))
        
        return self.compute(bids, sw_online)


def test_offline_optimal():
    """测试离线最优求解器"""
    print("=" * 60)
    print("测试离线最优求解器")
    print("=" * 60)
    
    # 创建测试投标
    bids = [
        BidInfo(0, 5, 0, 0.8, 2e9, 1e9, 100, 'high'),
        BidInfo(0, 10, 1, 0.7, 3e9, 0.5e9, 80, 'high'),
        BidInfo(1, 5, 0, 0.6, 2e9, 1e9, 90, 'medium'),
        BidInfo(1, 8, 2, 0.65, 2.5e9, 0.8e9, 85, 'medium'),
        BidInfo(2, 3, 1, 0.5, 1.5e9, 1.5e9, 120, 'low'),
        BidInfo(2, 7, 0, 0.55, 2e9, 1e9, 110, 'low'),
        BidInfo(3, 10, 2, 0.75, 3e9, 0.3e9, 70, 'high'),
        BidInfo(4, 5, 1, 0.45, 2e9, 1e9, 95, 'medium'),
    ]
    
    # 设置资源
    solver = OfflineOptimalSolver(n_uavs=3, n_users=5)
    uav_resources = [
        {'uav_id': 0, 'f_max': 10e9, 'E_max': 500e3},
        {'uav_id': 1, 'f_max': 10e9, 'E_max': 500e3},
        {'uav_id': 2, 'f_max': 10e9, 'E_max': 500e3},
    ]
    cloud_resources = {'f_cloud': 50e9}
    solver.set_resources(uav_resources, cloud_resources)
    
    # LP松弛求解
    sw_lp, alloc_lp = solver.solve(bids, use_lp_relaxation=True)
    print(f"\nLP松弛最优解: SW* = {sw_lp:.4f}")
    print(f"分配方案数: {len(alloc_lp)}")
    
    # 贪心求解对比
    sw_greedy, _ = solver._solve_greedy(bids)
    print(f"贪心解: SW = {sw_greedy:.4f}")
    print(f"LP优化增益: {(sw_lp - sw_greedy) / sw_greedy * 100:.2f}%")
    
    # 测试竞争比计算
    print("\n" + "-" * 40)
    print("测试竞争比计算")
    
    calculator = CompetitiveRatioCalculator()
    calculator.set_resources(uav_resources, cloud_resources)
    
    # 假设在线算法达到的社会福利
    sw_online = 2.0  # 假设值
    
    result = calculator.compute(bids, sw_online)
    print(f"\n离线最优: SW* = {result['sw_optimal']:.4f}")
    print(f"在线算法: SW = {result['sw_online']:.4f}")
    print(f"实际竞争比: ρ = {result['actual_ratio']:.4f}")
    print(f"间隙百分比: {result['gap_percentage']:.2f}%")
    print(f"理论上界: {result['theoretical_bound']:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")


if __name__ == "__main__":
    test_offline_optimal()
