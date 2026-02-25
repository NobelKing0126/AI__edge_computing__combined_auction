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
        utility_stage2: 阶段2基础效用 (idea118.txt 2.13节)
        utility_final: 最终效用 (含指数衰减融合, idea118.txt 3.4节)
        utility: 效用得分 (向后兼容，等于utility_final)
        priority_score: 任务优先级得分
        checkpoint_layer: Checkpoint层（None表示无）
        free_energy: 自由能结果
        allocation: 资源分配结果
        D_trans: 传输数据量 (bits)
        C_edge: 边缘计算量 (FLOPs)
        C_cloud: 云端计算量 (FLOPs)
    """
    task_id: int
    user_id: int
    uav_id: int
    split_layer: int
    f_edge: float
    f_cloud: float
    T_predict: float
    E_predict: float
    utility_stage2: float
    utility_final: float
    utility: float  # 向后兼容
    priority_score: float = 0.5
    checkpoint_layer: Optional[int] = None
    free_energy: FreeEnergyResult = None
    allocation: AllocationResult = None
    D_trans: float = 0.0
    C_edge: float = 0.0
    C_cloud: float = 0.0


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
    
    def compute_utility_stage2(self,
                               T_predict: float,
                               T_max: float,
                               E_predict: float,
                               E_budget: float) -> float:
        """
        计算阶段2基础效用 (不含自由能修正)

        公式 (idea118.txt 2.13节):
            η_stage2 = β₁*U_time + β₂*U_energy

            U_time = max(0, 1 - T_predict/T_max)
            U_energy = max(0, 1 - E_predict/E_budget)

        注意: 可靠性效用不再作为线性项，而是通过指数衰减融合

        Args:
            T_predict: 预测时延
            T_max: 最大时延
            E_predict: 预测能耗
            E_budget: 能量预算

        Returns:
            float: 阶段2基础效用值
        """
        # 时延效用
        U_time = max(0, 1 - T_predict / max(T_max, NUMERICAL.EPSILON))

        # 能效效用
        U_energy = max(0, 1 - E_predict / max(E_budget, NUMERICAL.EPSILON))

        # 加权求和 (重新归一化权重)
        total_weight = self.config.beta_time + self.config.beta_energy
        utility = (
            self.config.beta_time * U_time +
            self.config.beta_energy * U_energy
        ) / max(total_weight, NUMERICAL.EPSILON)

        return utility

    def compute_final_utility(self,
                              utility_stage2: float,
                              priority_score: float,
                              free_energy: float) -> float:
        """
        计算最终效用 (指数衰减融合)

        公式 (idea118.txt 3.4节):
            η_final = ω_i * η_stage2 * exp(-F̃/F_threshold)

        指数衰减特性 (idea118.txt 3.4.2节):
            F̃=0  -> exp(0) = 1.0 (无惩罚)
            F̃=15 -> exp(-0.5) ≈ 0.61 (中风险，效用衰减39%)
            F̃=30 -> exp(-1) ≈ 0.37 (高风险，效用衰减63%)
            F̃=60 -> exp(-2) ≈ 0.14 (超高风险，效用衰减86%)

        Args:
            utility_stage2: 阶段2基础效用
            priority_score: 任务优先级得分 ω_i
            free_energy: 自由能 F̃

        Returns:
            float: 最终效用值
        """
        # 指数衰减因子
        decay_factor = np.exp(-free_energy / max(self.config.F_threshold, NUMERICAL.EPSILON))

        # 优先级加权 + 指数衰减
        utility_final = priority_score * utility_stage2 * decay_factor

        return utility_final

    def compute_utility(self,
                        T_predict: float,
                        T_max: float,
                        E_predict: float,
                        E_budget: float,
                        free_energy: float,
                        priority_score: float = 0.5) -> Tuple[float, float]:
        """
        计算投标效用 (完整流程)

        整合阶段2基础效用和指数衰减融合

        Args:
            T_predict: 预测时延
            T_max: 最大时延
            E_predict: 预测能耗
            E_budget: 能量预算
            free_energy: 自由能
            priority_score: 任务优先级得分

        Returns:
            Tuple[float, float]: (utility_stage2, utility_final)
        """
        # 阶段2基础效用
        utility_stage2 = self.compute_utility_stage2(
            T_predict=T_predict,
            T_max=T_max,
            E_predict=E_predict,
            E_budget=E_budget
        )

        # 最终效用 (指数衰减融合)
        utility_final = self.compute_final_utility(
            utility_stage2=utility_stage2,
            priority_score=priority_score,
            free_energy=free_energy
        )

        return utility_stage2, utility_final
    
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

        流程 (idea118.txt 2.13节 + 3.4节):
            1. 遍历切分点，计算边缘/云端计算量
            2. 优化资源分配 (凸优化)
            3. 计算自由能 F̃ = F̃_trans + F̃_comp + F̃_energy
            4. 计算阶段2基础效用 η_stage2
            5. 计算最终效用 η_final = ω * η_stage2 * exp(-F̃/F_threshold)
            6. 返回Top-K投标

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

            # 传输数据量 = 切分层输出大小
            D_trans = output_size

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

            # 计算自由能 (idea118.txt 2.9节公式)
            # F̃ = F̃_trans + F̃_comp + F̃_energy
            free_energy = self.free_energy_calc.compute_free_energy(
                D_trans=D_trans,
                R_rate=R_trans,  # 使用中继传输速率
                C_edge=C_edge,
                C_cloud=C_cloud,
                f_edge=allocation.f_edge,
                f_cloud=allocation.f_cloud,
                E_remain=uav.E_remain,
                E_required=E_predict,
                E_max=uav.E_max
            )

            if free_energy.risk_level == RiskLevel.CRITICAL:
                continue

            # 确定Checkpoint
            checkpoint_layer = self.determine_checkpoint(
                free_energy, split_point, model_info.total_layers
            )

            # 计算效用 (idea118.txt 3.4节指数衰减融合)
            # η_final = ω_i * η_stage2 * exp(-F̃/F_threshold)
            utility_stage2, utility_final = self.compute_utility(
                T_predict=allocation.T_total,
                T_max=task.deadline,
                E_predict=E_predict,
                E_budget=uav.E_remain * self.exec_config.energy_budget_ratio,
                free_energy=free_energy.F_total,
                priority_score=task.priority_score
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
                utility_stage2=utility_stage2,
                utility_final=utility_final,
                utility=utility_final,  # 向后兼容
                priority_score=task.priority_score,
                checkpoint_layer=checkpoint_layer,
                free_energy=free_energy,
                allocation=allocation,
                D_trans=D_trans,
                C_edge=C_edge,
                C_cloud=C_cloud
            ))

        # 按最终效用排序，返回Top-K
        candidates.sort(key=lambda b: b.utility_final, reverse=True)

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
    print("测试 M18: BidGenerator (idea118.txt 3.4节指数衰减融合)")
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
              f"效用_stage2={bid.utility_stage2:.3f}, "
              f"效用_final={bid.utility_final:.3f}, "
              f"自由能={bid.free_energy.F_total:.2f}")
    print("  ✓ 单任务投标生成正确")

    # 测试2: 验证投标按效用排序
    print("\n[Test 2] 验证投标排序...")
    utilities = [b.utility_final for b in bids]
    assert utilities == sorted(utilities, reverse=True), "应按最终效用降序排序"
    print(f"  最终效用值: {[f'{u:.3f}' for u in utilities[:5]]}")
    print("  ✓ 投标排序正确")

    # 测试3: 批量投标生成
    print("\n[Test 3] 测试批量投标生成...")

    tasks = [
        TaskInfo(task_id=i, user_id=i, data_size=1e6*8, deadline=5.0, priority_score=0.5+i*0.1)
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

    # 测试4: 效用计算 (阶段2基础效用)
    print("\n[Test 4] 测试效用计算...")

    utility_stage2, utility_final = generator.compute_utility(
        T_predict=2.0, T_max=3.0,
        E_predict=50e3, E_budget=100e3,
        free_energy=20.0,
        priority_score=0.7
    )

    print(f"  效用_stage2: {utility_stage2:.4f}")
    print(f"  效用_final: {utility_final:.4f}")
    assert 0 <= utility_stage2 <= 1, "阶段2效用应在[0,1]范围内"
    assert 0 <= utility_final <= 1, "最终效用应在[0,1]范围内"
    assert utility_final < utility_stage2 * 0.7, "指数衰减应降低最终效用"
    print("  ✓ 效用计算正确")

    # 测试5: 指数衰减特性验证
    print("\n[Test 5] 测试指数衰减特性...")

    # 计算基准效用 (F=0时)
    u_stage2_base, u_final_0 = generator.compute_utility(1.0, 2.0, 50e3, 100e3, 0.0, 1.0)

    # F=0 -> exp(0) = 1.0 (无衰减)
    # 预期: utility_final = priority * utility_stage2 * exp(0) = 1.0 * u_stage2_base * 1.0
    expected_0 = 1.0 * u_stage2_base * np.exp(0)  # ω * η_stage2 * exp(0)
    assert abs(u_final_0 - expected_0) < 0.01, f"F=0时应无衰减, got {u_final_0}, expected {expected_0}"
    print(f"  F=0: utility_final={u_final_0:.4f} (预期≈{expected_0:.4f})")

    # F=30 (threshold) -> exp(-1) ≈ 0.368
    _, u_final_30 = generator.compute_utility(1.0, 2.0, 50e3, 100e3, 30.0, 1.0)
    expected_30 = 1.0 * u_stage2_base * np.exp(-1)
    assert abs(u_final_30 - expected_30) < 0.01, f"F=threshold时应衰减63%, got {u_final_30}, expected {expected_30}"
    print(f"  F=30 (threshold): utility_final={u_final_30:.4f} (预期≈{expected_30:.4f})")

    # F=60 -> exp(-2) ≈ 0.135
    _, u_final_60 = generator.compute_utility(1.0, 2.0, 50e3, 100e3, 60.0, 1.0)
    expected_60 = 1.0 * u_stage2_base * np.exp(-2)
    assert abs(u_final_60 - expected_60) < 0.01, f"F=2*threshold时应衰减87%, got {u_final_60}, expected {expected_60}"
    print(f"  F=60 (2xthreshold): utility_final={u_final_60:.4f} (预期≈{expected_60:.4f})")

    # 验证衰减比例
    ratio_30 = u_final_30 / u_final_0
    ratio_60 = u_final_60 / u_final_0
    assert abs(ratio_30 - np.exp(-1)) < 0.01, f"F=30时的衰减比例应为exp(-1)≈0.368, got {ratio_30}"
    assert abs(ratio_60 - np.exp(-2)) < 0.01, f"F=60时的衰减比例应为exp(-2)≈0.135, got {ratio_60}"
    print(f"  衰减比例验证: F=30时={ratio_30:.3f} (exp(-1)={np.exp(-1):.3f}), F=60时={ratio_60:.3f} (exp(-2)={np.exp(-2):.3f})")
    print("  ✓ 指数衰减特性正确")

    # 测试6: Checkpoint决策
    print("\n[Test 6] 测试Checkpoint决策...")

    from algorithms.phase2.free_energy import FreeEnergyResult

    fe_high = FreeEnergyResult(
        F_total=40.0, F_trans=10.0, F_comp=15.0, F_energy=15.0,
        F_time=10.0, F_reliability=10.0,
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
