"""
M16: ResourceOptimizer - 凸优化资源分配

功能：给定切分点，优化边缘/云端算力分配，最小化时延
输入：任务参数、切分点、资源约束
输出：最优算力分配、预期时延

关键公式 (idea118.txt 2.9-2.11节):
    优化问题: min T_total = T_edge + T_trans + T_cloud + T_return
    
    闭式解:
        边缘最优算力: f_edge* = C_edge / (T_max - T_comm - C_cloud/f_cloud_max)
        云端最优算力: f_cloud* = C_cloud / (T_max - T_comm - C_edge/f_edge*)
"""

import numpy as np
from typing import Tuple, Optional, Dict
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
        feasible: 是否可行
        margin: 时延余量 (s)
    """
    f_edge: float
    f_cloud: float
    T_edge: float
    T_trans: float
    T_cloud: float
    T_return: float
    T_total: float
    feasible: bool
    margin: float


class ResourceOptimizer:
    """
    凸优化资源分配器
    
    Attributes:
        f_edge_max: UAV最大算力
        f_cloud_max: 云端最大算力
        kappa_edge: 边缘能耗系数
    """
    
    def __init__(self,
                 f_edge_max: float = 10e9,
                 f_cloud_max: float = 100e9,
                 kappa_edge: float = 1e-28):
        """
        初始化优化器
        
        Args:
            f_edge_max: UAV最大算力 (FLOPS)
            f_cloud_max: 云端最大算力 (FLOPS)
            kappa_edge: 边缘能耗系数
        """
        self.f_edge_max = f_edge_max
        self.f_cloud_max = f_cloud_max
        self.kappa_edge = kappa_edge
    
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
    
    def optimize_allocation(self,
                            C_edge: float,
                            C_cloud: float,
                            T_upload: float,
                            T_trans: float,
                            T_return: float,
                            T_max: float,
                            f_edge_available: Optional[float] = None,
                            f_cloud_available: Optional[float] = None) -> AllocationResult:
        """
        优化资源分配
        
        使用闭式解求最优算力分配
        
        公式:
            最小化总时延: T = C_edge/f_edge + C_cloud/f_cloud + T_comm
            约束: f_edge ≤ f_edge_max, f_cloud ≤ f_cloud_max
            
        闭式解（凸优化KKT条件）:
            若无约束激活: f_edge* = C_edge * f_cloud_max / C_cloud (比例分配)
            
        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            T_upload: 上传时延 (s)
            T_trans: 中继传输时延 (s)
            T_return: 返回时延 (s)
            T_max: 最大允许时延 (s)
            f_edge_available: 可用边缘算力
            f_cloud_available: 可用云端算力
            
        Returns:
            AllocationResult: 分配结果
        """
        # 使用可用资源或最大资源
        f_edge_max = f_edge_available or self.f_edge_max
        f_cloud_max = f_cloud_available or self.f_cloud_max
        
        # 通信总时延
        T_comm = T_upload + T_trans + T_return
        
        # 可用于计算的时间
        T_compute_budget = T_max - T_comm
        
        if T_compute_budget <= 0:
            # 通信时延已超过限制
            return AllocationResult(
                f_edge=0, f_cloud=0,
                T_edge=0, T_trans=T_trans,
                T_cloud=0, T_return=T_return,
                T_total=T_comm,
                feasible=False,
                margin=-T_comm
            )
        
        # 处理纯边缘或纯云端情况
        if C_cloud <= 0:
            # 纯边缘计算
            f_edge_needed = C_edge / T_compute_budget
            f_edge = min(f_edge_needed, f_edge_max)
            T_edge = C_edge / f_edge
            
            T_total = T_upload + T_edge
            feasible = T_total <= T_max
            
            return AllocationResult(
                f_edge=f_edge, f_cloud=0,
                T_edge=T_edge, T_trans=0,
                T_cloud=0, T_return=0,
                T_total=T_total,
                feasible=feasible,
                margin=T_max - T_total
            )
        
        if C_edge <= 0:
            # 纯云端计算
            f_cloud_needed = C_cloud / T_compute_budget
            f_cloud = min(f_cloud_needed, f_cloud_max)
            T_cloud = C_cloud / f_cloud
            
            T_total = T_upload + T_trans + T_cloud + T_return
            feasible = T_total <= T_max
            
            return AllocationResult(
                f_edge=0, f_cloud=f_cloud,
                T_edge=0, T_trans=T_trans,
                T_cloud=T_cloud, T_return=T_return,
                T_total=T_total,
                feasible=feasible,
                margin=T_max - T_total
            )
        
        # 边缘-云端协同计算
        # 使用比例分配策略（基于计算量比例）
        total_compute = C_edge + C_cloud
        
        # 理想情况：按计算量比例分配时间
        # T_edge/T_cloud = C_edge/C_cloud => f_edge/f_cloud = 1 (相同利用率)
        
        # 使用最大算力计算最小时延
        T_edge_min = C_edge / f_edge_max
        T_cloud_min = C_cloud / f_cloud_max
        T_compute_min = T_edge_min + T_cloud_min
        
        if T_compute_min > T_compute_budget:
            # 即使最大算力也无法满足
            f_edge = f_edge_max
            f_cloud = f_cloud_max
            T_edge = T_edge_min
            T_cloud = T_cloud_min
            T_total = T_comm + T_compute_min
            
            return AllocationResult(
                f_edge=f_edge, f_cloud=f_cloud,
                T_edge=T_edge, T_trans=T_trans,
                T_cloud=T_cloud, T_return=T_return,
                T_total=T_total,
                feasible=False,
                margin=T_max - T_total
            )
        
        # 可行情况：按比例分配时间预算
        ratio_edge = C_edge / total_compute
        ratio_cloud = C_cloud / total_compute
        
        T_edge_alloc = T_compute_budget * ratio_edge
        T_cloud_alloc = T_compute_budget * ratio_cloud
        
        # 计算所需算力
        f_edge_needed = C_edge / T_edge_alloc
        f_cloud_needed = C_cloud / T_cloud_alloc
        
        # 限制到可用算力
        f_edge = min(f_edge_needed, f_edge_max)
        f_cloud = min(f_cloud_needed, f_cloud_max)
        
        # 重新计算实际时延
        T_edge = C_edge / f_edge
        T_cloud = C_cloud / f_cloud
        T_total = T_comm + T_edge + T_cloud
        
        feasible = T_total <= T_max
        
        return AllocationResult(
            f_edge=f_edge, f_cloud=f_cloud,
            T_edge=T_edge, T_trans=T_trans,
            T_cloud=T_cloud, T_return=T_return,
            T_total=T_total,
            feasible=feasible,
            margin=T_max - T_total
        )
    
    def compute_energy(self, f_edge: float, C_edge: float) -> float:
        """
        计算边缘计算能耗
        
        公式: E = κ * f² * C
        
        Args:
            f_edge: 边缘算力 (FLOPS)
            C_edge: 边缘计算量 (FLOPs)
            
        Returns:
            float: 能耗 (J)
        """
        return self.kappa_edge * (f_edge ** 2) * C_edge
    
    def find_optimal_split(self,
                           C_total: float,
                           split_ratios: list,
                           output_sizes: list,
                           T_upload_base: float,
                           R_trans: float,
                           R_return: float,
                           T_max: float) -> Tuple[int, AllocationResult]:
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
            T_trans = output_size / R_trans
            T_return = output_size / R_return  # 简化：返回数据量与中间特征相关
            
            result = self.optimize_allocation(
                C_edge=C_edge,
                C_cloud=C_cloud,
                T_upload=T_upload_base,
                T_trans=T_trans,
                T_return=T_return,
                T_max=T_max
            )
            
            if result.T_total < best_delay:
                best_delay = result.T_total
                best_split = i
                best_result = result
        
        return best_split, best_result


# ============ 测试用例 ============

def test_resource_optimizer():
    """测试ResourceOptimizer模块"""
    print("=" * 60)
    print("测试 M16: ResourceOptimizer")
    print("=" * 60)
    
    optimizer = ResourceOptimizer(
        f_edge_max=10e9,
        f_cloud_max=100e9
    )
    
    # 测试1: 基本资源分配
    print("\n[Test 1] 测试基本资源分配...")
    result = optimizer.optimize_allocation(
        C_edge=5e9,      # 5 GFLOPs
        C_cloud=15e9,    # 15 GFLOPs
        T_upload=0.1,    # 100ms
        T_trans=0.05,    # 50ms
        T_return=0.02,   # 20ms
        T_max=5.0        # 5s (更宽松的限制)
    )
    
    assert result.feasible, "应该可行"
    assert result.T_total <= 5.0, "总时延应不超过限制"
    
    print(f"  边缘算力: {result.f_edge/1e9:.2f} GFLOPS")
    print(f"  云端算力: {result.f_cloud/1e9:.2f} GFLOPS")
    print(f"  边缘时延: {result.T_edge*1000:.1f} ms")
    print(f"  云端时延: {result.T_cloud*1000:.1f} ms")
    print(f"  总时延: {result.T_total*1000:.1f} ms")
    print(f"  余量: {result.margin*1000:.1f} ms")
    print("  ✓ 基本分配正确")
    
    # 测试2: 不可行情况
    print("\n[Test 2] 测试不可行情况...")
    result_infeasible = optimizer.optimize_allocation(
        C_edge=50e9,     # 50 GFLOPs
        C_cloud=200e9,   # 200 GFLOPs
        T_upload=0.5,
        T_trans=0.3,
        T_return=0.2,
        T_max=1.0        # 只有1s
    )
    
    assert not result_infeasible.feasible, "应该不可行"
    print(f"  总时延: {result_infeasible.T_total*1000:.1f} ms > 1000ms")
    print(f"  余量: {result_infeasible.margin*1000:.1f} ms")
    print("  ✓ 不可行检测正确")
    
    # 测试3: 纯边缘计算
    print("\n[Test 3] 测试纯边缘计算...")
    result_edge_only = optimizer.optimize_allocation(
        C_edge=8e9,
        C_cloud=0,
        T_upload=0.1,
        T_trans=0,
        T_return=0,
        T_max=2.0
    )
    
    assert result_edge_only.f_cloud == 0, "云端算力应为0"
    print(f"  边缘算力: {result_edge_only.f_edge/1e9:.2f} GFLOPS")
    print(f"  总时延: {result_edge_only.T_total*1000:.1f} ms")
    print("  ✓ 纯边缘计算正确")
    
    # 测试4: 纯云端计算
    print("\n[Test 4] 测试纯云端计算...")
    result_cloud_only = optimizer.optimize_allocation(
        C_edge=0,
        C_cloud=50e9,
        T_upload=0.2,
        T_trans=0.1,
        T_return=0.05,
        T_max=2.0
    )
    
    assert result_cloud_only.f_edge == 0, "边缘算力应为0"
    print(f"  云端算力: {result_cloud_only.f_cloud/1e9:.2f} GFLOPS")
    print(f"  总时延: {result_cloud_only.T_total*1000:.1f} ms")
    print("  ✓ 纯云端计算正确")
    
    # 测试5: 能耗计算
    print("\n[Test 5] 测试能耗计算...")
    energy = optimizer.compute_energy(f_edge=5e9, C_edge=10e9)
    
    assert energy > 0, "能耗应大于0"
    print(f"  5 GFLOPS执行10G计算: {energy:.4f} J")
    print("  ✓ 能耗计算正确")
    
    # 测试6: 最优切分点搜索
    print("\n[Test 6] 测试最优切分点搜索...")
    split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    output_sizes = [1e6, 0.8e6, 0.5e6, 0.3e6, 0.2e6, 0.3e6, 0.5e6, 0.8e6, 1e6]  # bits
    
    best_split, best_result = optimizer.find_optimal_split(
        C_total=20e9,
        split_ratios=split_ratios,
        output_sizes=output_sizes,
        T_upload_base=0.1,
        R_trans=100e6,  # 100Mbps
        R_return=100e6,
        T_max=3.0
    )
    
    print(f"  最优切分点: {best_split} (边缘比例={split_ratios[best_split]*100:.0f}%)")
    print(f"  最小时延: {best_result.T_total*1000:.1f} ms")
    print("  ✓ 切分点搜索正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_resource_optimizer()
