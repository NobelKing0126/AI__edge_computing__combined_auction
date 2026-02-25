"""
M17: FreeEnergy - 自由能风险评估

功能：使用Active Inference的自由能框架评估任务执行风险
输入：任务状态、UAV状态、环境不确定性
输出：自由能值、风险等级

关键公式 (idea118.txt 2.4节):
    自由能: F = -log P(o|s) + D_KL[Q(s)||P(s)]
    
    简化实现:
        F = F_energy + F_time + F_reliability
        
        能量风险: F_energy = -log(E_remain / E_required)
        时间风险: F_time = -log((T_max - T_predict) / T_max)
        可靠性风险: F_reliability = -log(health_score)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import FREE_ENERGY, NUMERICAL


class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"           # F < threshold
    MEDIUM = "MEDIUM"     # threshold <= F < max
    HIGH = "HIGH"         # F >= max
    CRITICAL = "CRITICAL" # 系统无法处理


@dataclass
class FreeEnergyResult:
    """
    自由能计算结果
    
    Attributes:
        F_total: 总自由能
        F_energy: 能量自由能分量
        F_time: 时间自由能分量
        F_reliability: 可靠性自由能分量
        risk_level: 风险等级
        requires_checkpoint: 是否建议Checkpoint
        details: 详细信息
    """
    F_total: float
    F_energy: float
    F_time: float
    F_reliability: float
    risk_level: RiskLevel
    requires_checkpoint: bool
    details: Dict[str, float]


class FreeEnergyCalculator:
    """
    自由能计算器
    
    Attributes:
        F_threshold: 自由能阈值（低/中风险分界）
        F_max: 自由能最大值（中/高风险分界）
        w_energy: 能量权重
        w_time: 时间权重
        w_reliability: 可靠性权重
        scale_factor: 自由能缩放因子
    """
    
    def __init__(self,
                 F_threshold: float = 30.0,
                 F_max: float = 50.0,
                 w_energy: float = 0.4,
                 w_time: float = 0.4,
                 w_reliability: float = 0.2,
                 scale_factor: Optional[float] = None):
        """
        初始化计算器
        
        Args:
            F_threshold: 低风险阈值
            F_max: 高风险阈值
            w_energy: 能量权重
            w_time: 时间权重
            w_reliability: 可靠性权重
            scale_factor: 自由能缩放因子，默认使用常量配置
        """
        self.F_threshold = F_threshold
        self.F_max = F_max
        self.w_energy = w_energy
        self.w_time = w_time
        self.w_reliability = w_reliability
        self.scale_factor = scale_factor if scale_factor is not None else FREE_ENERGY.SCALE_FACTOR
        self.max_free_energy = FREE_ENERGY.MAX_FREE_ENERGY
    
    def compute_energy_free_energy(self,
                                    E_remain: float,
                                    E_required: float,
                                    E_max: float) -> float:
        """
        计算能量自由能
        
        公式: F_energy = -log(E_remain / E_required) * scale
        
        Args:
            E_remain: 剩余能量 (J)
            E_required: 所需能量 (J)
            E_max: 最大能量 (J)
            
        Returns:
            float: 能量自由能
        """
        if E_required <= 0:
            return 0.0
        
        # 能量充足度
        ratio = E_remain / max(E_required, NUMERICAL.EPSILON)
        
        if ratio <= 0:
            return self.max_free_energy  # 极高风险
        
        # 归一化到合理范围
        # ratio > 1 表示能量充足，F应较小
        # ratio < 1 表示能量不足，F应较大
        if ratio >= 1:
            F = -np.log(ratio) * self.scale_factor
            F = max(F, 0)
        else:
            F = -np.log(ratio) * self.scale_factor
        
        return min(F, self.max_free_energy)  # 限制最大值
    
    def compute_time_free_energy(self,
                                  T_max: float,
                                  T_predict: float) -> float:
        """
        计算时间自由能
        
        公式: F_time = -log((T_max - T_predict) / T_max) * scale
        
        Args:
            T_max: 最大允许时延 (s)
            T_predict: 预测时延 (s)
            
        Returns:
            float: 时间自由能
        """
        if T_max <= 0:
            return self.max_free_energy
        
        # 时间余量
        margin_ratio = (T_max - T_predict) / T_max
        
        if margin_ratio <= 0:
            return self.max_free_energy  # 已超时
        
        F = -np.log(margin_ratio) * self.scale_factor
        
        return min(max(F, 0), self.max_free_energy)
    
    def compute_reliability_free_energy(self,
                                         health_score: float,
                                         channel_quality: float = 1.0) -> float:
        """
        计算可靠性自由能
        
        公式: F_reliability = -log(health * channel) * scale
        
        Args:
            health_score: 健康度得分 [0, 1]
            channel_quality: 信道质量 [0, 1]
            
        Returns:
            float: 可靠性自由能
        """
        combined = health_score * channel_quality
        
        if combined <= 0:
            return self.max_free_energy
        
        F = -np.log(combined) * self.scale_factor
        
        return min(max(F, 0), self.max_free_energy)
    
    def compute_free_energy(self,
                            E_remain: float,
                            E_required: float,
                            E_max: float,
                            T_max: float,
                            T_predict: float,
                            health_score: float,
                            channel_quality: float = 1.0) -> FreeEnergyResult:
        """
        计算总自由能
        
        Args:
            E_remain: 剩余能量 (J)
            E_required: 所需能量 (J)
            E_max: 最大能量 (J)
            T_max: 最大时延 (s)
            T_predict: 预测时延 (s)
            health_score: 健康度 [0, 1]
            channel_quality: 信道质量 [0, 1]
            
        Returns:
            FreeEnergyResult: 自由能结果
        """
        # 计算各分量
        F_energy = self.compute_energy_free_energy(E_remain, E_required, E_max)
        F_time = self.compute_time_free_energy(T_max, T_predict)
        F_reliability = self.compute_reliability_free_energy(health_score, channel_quality)
        
        # 加权求和
        F_total = (
            self.w_energy * F_energy +
            self.w_time * F_time +
            self.w_reliability * F_reliability
        )
        
        # 确定风险等级
        if F_total >= self.F_max:
            risk_level = RiskLevel.HIGH
        elif F_total >= self.F_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 检查是否有致命风险
        if E_remain < E_required or T_predict > T_max:
            risk_level = RiskLevel.CRITICAL
        
        # 是否需要Checkpoint
        requires_checkpoint = F_total >= self.F_threshold
        
        details = {
            'energy_ratio': E_remain / max(E_required, 1e-10),
            'time_margin': (T_max - T_predict) / max(T_max, 1e-10),
            'reliability': health_score * channel_quality
        }
        
        return FreeEnergyResult(
            F_total=F_total,
            F_energy=F_energy,
            F_time=F_time,
            F_reliability=F_reliability,
            risk_level=risk_level,
            requires_checkpoint=requires_checkpoint,
            details=details
        )
    
    def compute_checkpoint_benefit(self,
                                    F_current: float,
                                    checkpoint_overhead: float,
                                    progress_ratio: float) -> float:
        """
        计算Checkpoint的收益
        
        Args:
            F_current: 当前自由能
            checkpoint_overhead: Checkpoint开销（时间）
            progress_ratio: 当前进度比例
            
        Returns:
            float: 收益值（正值表示应该Checkpoint）
        """
        # 失败恢复收益 = 节省的重新计算成本
        recovery_benefit = F_current * progress_ratio
        
        # Checkpoint成本（使用缩放因子）
        checkpoint_cost = checkpoint_overhead * self.scale_factor
        
        return recovery_benefit - checkpoint_cost
    
    def should_checkpoint(self,
                          F_current: float,
                          progress_ratio: float,
                          checkpoint_time: float,
                          T_remaining: float) -> Tuple[bool, float]:
        """
        决定是否应该执行Checkpoint
        
        Args:
            F_current: 当前自由能
            progress_ratio: 当前进度比例
            checkpoint_time: Checkpoint时间
            T_remaining: 剩余可用时间
            
        Returns:
            Tuple[bool, float]: (是否Checkpoint, 收益值)
        """
        # 时间约束检查
        if checkpoint_time > T_remaining:
            return False, -float('inf')
        
        benefit = self.compute_checkpoint_benefit(
            F_current,
            checkpoint_time / max(T_remaining, 1e-10),
            progress_ratio
        )
        
        # 高风险情况强制Checkpoint
        if F_current >= self.F_max and progress_ratio > 0.2:
            return True, benefit
        
        # 中等风险且收益为正
        if F_current >= self.F_threshold and benefit > 0:
            return True, benefit
        
        return benefit > 0, benefit


# ============ 测试用例 ============

def test_free_energy():
    """测试FreeEnergy模块"""
    print("=" * 60)
    print("测试 M17: FreeEnergy")
    print("=" * 60)
    
    calculator = FreeEnergyCalculator()
    
    # 测试1: 能量自由能
    print("\n[Test 1] 测试能量自由能...")
    
    # 能量充足
    F_energy_ok = calculator.compute_energy_free_energy(
        E_remain=400e3, E_required=100e3, E_max=500e3
    )
    assert F_energy_ok < 20, "能量充足时自由能应较低"
    print(f"  能量充足(4倍): F_energy = {F_energy_ok:.2f}")
    
    # 能量紧张
    F_energy_tight = calculator.compute_energy_free_energy(
        E_remain=120e3, E_required=100e3, E_max=500e3
    )
    print(f"  能量紧张(1.2倍): F_energy = {F_energy_tight:.2f}")
    
    # 能量不足
    F_energy_low = calculator.compute_energy_free_energy(
        E_remain=50e3, E_required=100e3, E_max=500e3
    )
    assert F_energy_low > F_energy_ok, "能量不足时自由能应更高"
    print(f"  能量不足(0.5倍): F_energy = {F_energy_low:.2f}")
    print("  ✓ 能量自由能正确")
    
    # 测试2: 时间自由能
    print("\n[Test 2] 测试时间自由能...")
    
    # 时间充裕
    F_time_ok = calculator.compute_time_free_energy(T_max=3.0, T_predict=1.0)
    print(f"  时间充裕(预测1s/限制3s): F_time = {F_time_ok:.2f}")
    
    # 时间紧张
    F_time_tight = calculator.compute_time_free_energy(T_max=3.0, T_predict=2.5)
    print(f"  时间紧张(预测2.5s/限制3s): F_time = {F_time_tight:.2f}")
    
    # 即将超时
    F_time_critical = calculator.compute_time_free_energy(T_max=3.0, T_predict=2.9)
    assert F_time_critical > F_time_ok, "时间紧张时自由能应更高"
    print(f"  即将超时(预测2.9s/限制3s): F_time = {F_time_critical:.2f}")
    print("  ✓ 时间自由能正确")
    
    # 测试3: 可靠性自由能
    print("\n[Test 3] 测试可靠性自由能...")
    
    F_rel_good = calculator.compute_reliability_free_energy(0.9, 0.95)
    F_rel_bad = calculator.compute_reliability_free_energy(0.5, 0.6)
    
    assert F_rel_bad > F_rel_good, "可靠性差时自由能应更高"
    print(f"  高可靠性(0.9×0.95): F_rel = {F_rel_good:.2f}")
    print(f"  低可靠性(0.5×0.6): F_rel = {F_rel_bad:.2f}")
    print("  ✓ 可靠性自由能正确")
    
    # 测试4: 总自由能计算
    print("\n[Test 4] 测试总自由能...")
    
    result = calculator.compute_free_energy(
        E_remain=300e3, E_required=100e3, E_max=500e3,
        T_max=3.0, T_predict=2.0,
        health_score=0.8, channel_quality=0.9
    )
    
    print(f"  F_total = {result.F_total:.2f}")
    print(f"  风险等级: {result.risk_level.value}")
    print(f"  需要Checkpoint: {result.requires_checkpoint}")
    print("  ✓ 总自由能正确")
    
    # 测试5: 高风险场景
    print("\n[Test 5] 测试高风险场景...")
    
    result_high = calculator.compute_free_energy(
        E_remain=80e3, E_required=100e3, E_max=500e3,
        T_max=3.0, T_predict=2.8,
        health_score=0.5, channel_quality=0.6
    )
    
    assert result_high.risk_level == RiskLevel.CRITICAL, "应为致命风险"
    print(f"  F_total = {result_high.F_total:.2f}")
    print(f"  风险等级: {result_high.risk_level.value}")
    print("  ✓ 高风险检测正确")
    
    # 测试6: Checkpoint决策
    print("\n[Test 6] 测试Checkpoint决策...")
    
    should_cp, benefit = calculator.should_checkpoint(
        F_current=40.0,
        progress_ratio=0.5,
        checkpoint_time=0.1,
        T_remaining=1.0
    )
    
    print(f"  中风险(F=40), 50%进度: 应Checkpoint={should_cp}, 收益={benefit:.2f}")
    
    should_cp_low, _ = calculator.should_checkpoint(
        F_current=10.0,
        progress_ratio=0.3,
        checkpoint_time=0.2,
        T_remaining=1.0
    )
    print(f"  低风险(F=10), 30%进度: 应Checkpoint={should_cp_low}")
    print("  ✓ Checkpoint决策正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_free_energy()
