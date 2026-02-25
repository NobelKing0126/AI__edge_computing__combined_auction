"""
M18: BidGenerator - 投标生成器

功能：为每个任务生成K个候选投标
输入：任务信息、UAV状态、DNN模型信息
输出：投标集合

流程 (idea118.txt 2.13节):
    1. 解析DNN模型，获取切分点
    2. 对每个切分点：
       - 计算边缘/云端计算量
       - 优化资源分配
       - 计算自由能风险
       - 确定Checkpoint策略
       - 计算效用
    3. 选择Top-K投标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.phase2.resource_optimizer import ResourceOptimizer, AllocationResult
from algorithms.phase2.free_energy import FreeEnergyCalculator, FreeEnergyResult, RiskLevel
from config.system_config import SystemConfig, BiddingConfig, ExecutionConfig
from config.constants import RESOURCE, NUMERICAL, COMMUNICATION


@dataclass
class CandidateBid:
    """
    候选投标
    
    Attributes:
        task_id: 任务ID
        user_id: 用户ID
        uav_id: 目标UAV ID
        split_layer: 切分层
        f_edge: 边缘算力
        f_cloud: 云端算力
        T_predict: 预测时延
        E_predict: 预测能耗
        utility: 效用得分
        checkpoint_layer: Checkpoint层（None表示无）
        free_energy: 自由能结果
        allocation: 资源分配结果
    """
    task_id: int
    user_id: int
    uav_id: int
    split_layer: int
    f_edge: float
    f_cloud: float
    T_predict: float
    E_predict: float
    utility: float
    checkpoint_layer: Optional[int]
    free_energy: FreeEnergyResult
    allocation: AllocationResult


@dataclass
class TaskInfo:
    """
    任务信息
    
    Attributes:
        task_id: 任务ID
        user_id: 用户ID
        data_size: 输入数据大小 (bits)
        deadline: 截止时间 (s)
        model_name: DNN模型名称
        priority_score: 优先级得分
    """
    task_id: int
    user_id: int
    data_size: float
    deadline: float
    model_name: str = "vgg16"
    priority_score: float = 0.5


@dataclass
class UAVInfo:
    """
    UAV信息
    
    Attributes:
        uav_id: UAV ID
        position: 位置 (x, y, z)
        E_remain: 剩余能量 (J)
        E_max: 最大能量 (J)
        f_available: 可用算力 (FLOPS)
        f_max: 最大算力 (FLOPS)
        health_score: 健康度
    """
    uav_id: int
    position: Tuple[float, float, float]
    E_remain: float
    E_max: float
    f_available: float
    f_max: float
    health_score: float = 0.9


@dataclass
class ModelSplitInfo:
    """
    模型切分信息
    
    Attributes:
        total_layers: 总层数
        total_flops: 总计算量 (FLOPs)
        split_points: 可切分点列表
        edge_ratios: 各切分点边缘计算比例
        output_sizes: 各切分点输出数据量 (bits)
    """
    total_layers: int
    total_flops: float
    split_points: List[int]
    edge_ratios: List[float]
    output_sizes: List[float]
    
    @classmethod
    def from_dnn_model_spec(cls, model_spec, include_all_layers: bool = False) -> 'ModelSplitInfo':
        """
        从DNNModelSpec创建ModelSplitInfo，使用精确的每层数据
        
        Args:
            model_spec: DNNModelSpec对象
            include_all_layers: 是否包含所有层作为切分点
            
        Returns:
            ModelSplitInfo: 模型切分信息
        """
        n_layers = model_spec.layers if hasattr(model_spec, 'layers') else 10
        total_flops = model_spec.total_flops if hasattr(model_spec, 'total_flops') else 10e9
        
        if include_all_layers:
            # 所有层作为切分点
            split_points = list(range(n_layers + 1))
        else:
            # 典型切分点 (0%, 20%, 40%, 60%, 80%, 100%)
            split_points = [int(n_layers * r) for r in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
            split_points = list(set(split_points))  # 去重
            split_points.sort()
        
        edge_ratios = []
        output_sizes = []
        
        for split_layer in split_points:
            # 使用精确的计算量比例
            if hasattr(model_spec, 'get_flops_ratio_at_layer'):
                edge_ratio = model_spec.get_flops_ratio_at_layer(split_layer)
            else:
                edge_ratio = split_layer / n_layers
            edge_ratios.append(edge_ratio)
            
            # 使用精确的特征大小
            if split_layer == 0:
                output_size = model_spec.input_size_bytes * 8 if hasattr(model_spec, 'input_size_bytes') else 150000 * 8
            elif split_layer >= n_layers:
                output_size = 1000 * 8  # 结果数据约1KB
            elif hasattr(model_spec, 'get_output_size_at_layer'):
                output_size = model_spec.get_output_size_at_layer(split_layer - 1) * 8  # bytes -> bits
            elif hasattr(model_spec, 'typical_feature_sizes') and model_spec.typical_feature_sizes:
                idx = int(split_layer / n_layers * len(model_spec.typical_feature_sizes))
                idx = min(idx, len(model_spec.typical_feature_sizes) - 1)
                output_size = model_spec.typical_feature_sizes[idx] * 8
            else:
                output_size = 150000 * (1 - split_layer / n_layers) * 0.5 * 8
            
            output_sizes.append(output_size)
        
        return cls(
            total_layers=n_layers,
            total_flops=total_flops,
            split_points=split_points,
            edge_ratios=edge_ratios,
            output_sizes=output_sizes
        )


class BidGenerator:
    """
    投标生成器
    
    Attributes:
        resource_optimizer: 资源优化器
        free_energy_calc: 自由能计算器
        config: 投标配置
        exec_config: 执行配置
    """
    
    def __init__(self,
                 f_edge_max: float = 10e9,
                 f_cloud_max: float = 100e9,
                 config: Optional[BiddingConfig] = None,
                 exec_config: Optional[ExecutionConfig] = None):
        """
        初始化投标生成器
        
        Args:
            f_edge_max: UAV最大算力
            f_cloud_max: 云端最大算力
            config: 投标配置
            exec_config: 执行配置（用于获取能量预算比例等参数）
        """
        self.config = config or BiddingConfig()
        self.exec_config = exec_config or ExecutionConfig()
        
        self.resource_optimizer = ResourceOptimizer(
            f_edge_max=f_edge_max,
            f_cloud_max=f_cloud_max
        )
        
        self.free_energy_calc = FreeEnergyCalculator(
            F_threshold=self.config.F_threshold,
            F_max=self.config.F_max,
            scale_factor=self.exec_config.free_energy_scale
        )
    
    def compute_communication_delays(self,
                                      data_size: float,
                                      output_size: float,
                                      R_upload: float,
                                      R_trans: float,
                                      R_return: float) -> Tuple[float, float, float]:
        """
        计算通信时延
        
        Args:
            data_size: 输入数据大小 (bits)
            output_size: 中间数据大小 (bits)
            R_upload: 上传速率 (bps)
            R_trans: 中继传输速率 (bps)
            R_return: 返回速率 (bps)
            
        Returns:
            Tuple: (上传时延, 中继时延, 返回时延)
        """
        T_upload = data_size / max(R_upload, NUMERICAL.EPSILON)
        T_trans = output_size / max(R_trans, NUMERICAL.EPSILON)
        # 返回数据大小使用常量配置的比例
        T_return = output_size * COMMUNICATION.RESULT_SIZE_RATIO / max(R_return, NUMERICAL.EPSILON)
        
        return T_upload, T_trans, T_return
    
    def compute_utility(self,
                        T_predict: float,
                        T_max: float,
                        E_predict: float,
                        E_budget: float,
                        free_energy: float) -> float:
        """
        计算投标效用
        
        公式 (idea118.txt 2.13节):
            η = β₁*U_time + β₂*U_energy + β₃*U_reliability
            
            U_time = max(0, 1 - T_predict/T_max)
            U_energy = max(0, 1 - E_predict/E_budget)
            U_reliability = max(0, 1 - F/F_max)
        
        Args:
            T_predict: 预测时延
            T_max: 最大时延
            E_predict: 预测能耗
            E_budget: 能量预算
            free_energy: 自由能
            
        Returns:
            float: 效用值
        """
        # 时延效用（使用常量保护值）
        U_time = max(0, 1 - T_predict / max(T_max, NUMERICAL.EPSILON))
        
        # 能效效用
        U_energy = max(0, 1 - E_predict / max(E_budget, NUMERICAL.EPSILON))
        
        # 可靠性效用
        U_reliability = max(0, 1 - free_energy / max(self.config.F_max, NUMERICAL.EPSILON))
        
        # 加权求和
        utility = (
            self.config.beta_time * U_time +
            self.config.beta_energy * U_energy +
            self.config.beta_reliability * U_reliability
        )
        
        return utility
    
    def determine_checkpoint(self,
                             free_energy: FreeEnergyResult,
                             split_layer: int,
                             total_layers: int) -> Optional[int]:
        """
        确定Checkpoint位置
        
        Args:
            free_energy: 自由能结果
            split_layer: 切分层
            total_layers: 总层数
            
        Returns:
            Optional[int]: Checkpoint层，None表示无
        """
        if not free_energy.requires_checkpoint:
            return None
        
        # 在边缘部分的中间位置设置Checkpoint
        if split_layer > 1:
            return split_layer // 2
        
        return None
    
    def generate_bids_for_task(self,
                               task: TaskInfo,
                               uav: UAVInfo,
                               model_info: ModelSplitInfo,
                               R_upload: float = 10e6,  # 10 Mbps
                               R_trans: float = 100e6,  # 100 Mbps
                               R_return: float = 100e6,
                               top_k: Optional[int] = None) -> List[CandidateBid]:
        """
        为单个任务生成候选投标
        
        Args:
            task: 任务信息
            uav: UAV信息
            model_info: 模型切分信息
            R_upload: 上传速率
            R_trans: 中继传输速率
            R_return: 返回速率
            top_k: 返回前K个投标
            
        Returns:
            List[CandidateBid]: 候选投标列表
        """
        if top_k is None:
            top_k = self.config.top_k
        
        candidates = []
        
        # 遍历所有切分点
        for i, (split_point, edge_ratio, output_size) in enumerate(
            zip(model_info.split_points, 
                model_info.edge_ratios, 
                model_info.output_sizes)):
            
            # 计算边缘/云端计算量
            C_edge = model_info.total_flops * edge_ratio
            C_cloud = model_info.total_flops * (1 - edge_ratio)
            
            # 计算通信时延
            T_upload, T_trans, T_return = self.compute_communication_delays(
                task.data_size, output_size, R_upload, R_trans, R_return
            )
            
            # 优化资源分配
            allocation = self.resource_optimizer.optimize_allocation(
                C_edge=C_edge,
                C_cloud=C_cloud,
                T_upload=T_upload,
                T_trans=T_trans,
                T_return=T_return,
                T_max=task.deadline,
                f_edge_available=uav.f_available
            )
            
            if not allocation.feasible:
                continue
            
            # 计算能耗
            E_predict = self.resource_optimizer.compute_energy(
                allocation.f_edge, C_edge
            )
            
            # 计算自由能（使用配置的信道质量默认值）
            free_energy = self.free_energy_calc.compute_free_energy(
                E_remain=uav.E_remain,
                E_required=E_predict,
                E_max=uav.E_max,
                T_max=task.deadline,
                T_predict=allocation.T_total,
                health_score=uav.health_score,
                channel_quality=self.exec_config.default_channel_quality
            )
            
            if free_energy.risk_level == RiskLevel.CRITICAL:
                continue
            
            # 确定Checkpoint
            checkpoint_layer = self.determine_checkpoint(
                free_energy, split_point, model_info.total_layers
            )
            
            # 计算效用（使用配置的能量预算比例）
            utility = self.compute_utility(
                T_predict=allocation.T_total,
                T_max=task.deadline,
                E_predict=E_predict,
                E_budget=uav.E_remain * self.exec_config.energy_budget_ratio,
                free_energy=free_energy.F_total
            )
            
            candidates.append(CandidateBid(
                task_id=task.task_id,
                user_id=task.user_id,
                uav_id=uav.uav_id,
                split_layer=split_point,
                f_edge=allocation.f_edge,
                f_cloud=allocation.f_cloud,
                T_predict=allocation.T_total,
                E_predict=E_predict,
                utility=utility,
                checkpoint_layer=checkpoint_layer,
                free_energy=free_energy,
                allocation=allocation
            ))
        
        # 按效用排序，返回Top-K
        candidates.sort(key=lambda b: b.utility, reverse=True)
        
        return candidates[:top_k]
    
    def generate_bids_batch(self,
                            tasks: List[TaskInfo],
                            uavs: List[UAVInfo],
                            model_info: ModelSplitInfo,
                            user_uav_mapping: Dict[int, int]) -> Dict[int, List[CandidateBid]]:
        """
        批量生成投标
        
        Args:
            tasks: 任务列表
            uavs: UAV列表
            model_info: 模型信息
            user_uav_mapping: 用户->UAV映射
            
        Returns:
            Dict: {task_id: [候选投标]}
        """
        uav_dict = {u.uav_id: u for u in uavs}
        result = {}
        
        for task in tasks:
            uav_id = user_uav_mapping.get(task.user_id)
            if uav_id is None or uav_id not in uav_dict:
                continue
            
            uav = uav_dict[uav_id]
            bids = self.generate_bids_for_task(task, uav, model_info)
            
            if bids:
                result[task.task_id] = bids
        
        return result


# ============ 测试用例 ============

def test_bid_generator():
    """测试BidGenerator模块"""
    print("=" * 60)
    print("测试 M18: BidGenerator")
    print("=" * 60)
    
    generator = BidGenerator()
    
    # 创建测试数据
    task = TaskInfo(
        task_id=0,
        user_id=0,
        data_size=1e6 * 8,  # 1MB (减小数据量)
        deadline=5.0,       # 5s (增加时间预算)
        model_name="vgg16",
        priority_score=0.7
    )
    
    uav = UAVInfo(
        uav_id=0,
        position=(500, 500, 100),
        E_remain=400e3,
        E_max=500e3,
        f_available=8e9,
        f_max=10e9,
        health_score=0.9
    )
    
    # 模拟VGG16切分信息 (缩小规模用于测试)
    model_info = ModelSplitInfo(
        total_layers=16,
        total_flops=5e9,  # 5 GFLOPs (简化)
        split_points=[2, 4, 7, 10, 13, 16],
        edge_ratios=[0.1, 0.2, 0.35, 0.5, 0.7, 1.0],
        output_sizes=[0.5e6*8, 0.3e6*8, 0.2e6*8, 0.15e6*8, 0.1e6*8, 0.05e6*8]  # bits
    )
    
    # 测试1: 单任务投标生成
    print("\n[Test 1] 测试单任务投标生成...")
    # 使用更高的上传速率
    bids = generator.generate_bids_for_task(
        task, uav, model_info,
        R_upload=10e6,  # 10 Mbps
        R_trans=100e6,
        R_return=100e6
    )
    
    assert len(bids) > 0, "应生成至少一个投标"
    print(f"  生成了 {len(bids)} 个候选投标")
    
    for i, bid in enumerate(bids[:3]):
        print(f"  投标{i+1}: 切分层={bid.split_layer}, "
              f"时延={bid.T_predict*1000:.0f}ms, "
              f"效用={bid.utility:.3f}")
    print("  ✓ 单任务投标生成正确")
    
    # 测试2: 验证投标按效用排序
    print("\n[Test 2] 验证投标排序...")
    utilities = [b.utility for b in bids]
    assert utilities == sorted(utilities, reverse=True), "应按效用降序排序"
    print(f"  效用值: {[f'{u:.3f}' for u in utilities[:5]]}")
    print("  ✓ 投标排序正确")
    
    # 测试3: 批量投标生成
    print("\n[Test 3] 测试批量投标生成...")
    
    tasks = [
        TaskInfo(task_id=i, user_id=i, data_size=1e6*8, deadline=5.0)
        for i in range(5)
    ]
    
    uavs = [
        UAVInfo(uav_id=i, position=(i*500, 500, 100),
               E_remain=400e3, E_max=500e3,
               f_available=8e9, f_max=10e9)
        for i in range(3)
    ]
    
    user_uav_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    
    batch_bids = generator.generate_bids_batch(
        tasks, uavs, model_info, user_uav_mapping
    )
    
    assert len(batch_bids) == 5, "应有5个任务的投标"
    print(f"  为 {len(batch_bids)} 个任务生成了投标")
    
    for task_id, bids in batch_bids.items():
        print(f"    任务{task_id}: {len(bids)} 个候选投标")
    print("  ✓ 批量投标生成正确")
    
    # 测试4: 效用计算
    print("\n[Test 4] 测试效用计算...")
    
    utility = generator.compute_utility(
        T_predict=2.0, T_max=3.0,
        E_predict=50e3, E_budget=100e3,
        free_energy=20.0
    )
    
    assert 0 <= utility <= 1, "效用应在[0,1]范围内"
    print(f"  效用值: {utility:.4f}")
    print("  ✓ 效用计算正确")
    
    # 测试5: Checkpoint决策
    print("\n[Test 5] 测试Checkpoint决策...")
    
    from algorithms.phase2.free_energy import FreeEnergyResult
    
    fe_high = FreeEnergyResult(
        F_total=40.0, F_energy=15.0, F_time=15.0, F_reliability=10.0,
        risk_level=RiskLevel.MEDIUM, requires_checkpoint=True, details={}
    )
    
    cp_layer = generator.determine_checkpoint(fe_high, split_layer=8, total_layers=16)
    assert cp_layer is not None, "高风险时应设置Checkpoint"
    print(f"  切分层8, Checkpoint层: {cp_layer}")
    print("  ✓ Checkpoint决策正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_bid_generator()
