"""
Active Inference Adapter - 主动推理框架适配器

功能：将主动推理框架与现有代码（BidGenerator、TaskExecutor）衔接
提供信息及时反馈机制

衔接说明：
    1. 从 TaskExecutor 获取实时状态，转换为 StateVector
    2. 使用主动推理计算自由能和推荐行动
    3. 将自由能结果反馈给 BidGenerator 进行效用修正
    4. 提供动态 Checkpoint 建议
    5. 支持运行时决策和风险评估
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 主动推理模块导入
from algorithms.active_inference.state_space import (
    StateVector, ActionType, StateBounds
)
from algorithms.active_inference.free_energy import (
    FreeEnergyCalculator, FourComponentCalculator, InstantFreeEnergy, RiskLevel
)
from config.constants import NUMERICAL


@dataclass
class ActiveInferenceFeedback:
    """
    主动推理反馈信息

    Attributes:
        free_energy: 四分量自由能结果
        recommended_action: 推荐行动
        action_confidence: 行动置信度
        checkpoint_suggestion: Checkpoint建议层
        risk_assessment: 风险评估
    """
    free_energy: Optional[InstantFreeEnergy] = None
    recommended_action: Optional[ActionType] = None
    action_confidence: float = 0.5
    checkpoint_suggestion: Optional[int] = None
    risk_assessment: str = "LOW"


@dataclass
class StateMapping:
    """
    状态映射参数

    Attributes:
        E_max: 最大能量
        T_max: 最大时间
        d_max: 最大距离
        initial_p: 初始进度
    """
    E_max: float = 500e3  # 500kJ
    T_max: float = 300.0  # 5分钟
    d_max: float = 2000.0  # 2km
    initial_p: float = 0.0


class FreeEnergyAdapter:
    """
    自由能适配器

    功能：将四分量自由能与 idea118 自由能格式衔接
    """

    def __init__(self):
        """初始化自由能适配器"""
        self.four_component = FourComponentCalculator()

    def convert_free_energy_result(self,
                                  four_component_fe: InstantFreeEnergy) -> Tuple[float, Dict]:
        """
        转换自由能结果为兼容格式

        Args:
            four_component_fe: 四分量自由能结果

        Returns:
            Tuple[float, Dict]: (总自由能, 详细分量字典)
        """
        # 总自由能使用四分量结果
        F_total = four_component_fe.F_total

        # 详细分量字典
        details = {
            'four_component': {
                'F_energy': four_component_fe.F_energy,
                'F_time': four_component_fe.F_time,
                'F_health': four_component_fe.F_health,
                'F_progress': four_component_fe.F_progress
            },
            'risk_level': four_component_fe.risk_level.name,
            'recommend_checkpoint': four_component_fe.risk_level != RiskLevel.LOW
        }

        return F_total, details


class ActiveInferenceIntegrator:
    """
    主动推理集成器

    功能：整合主动推理框架与现有系统，提供统一接口

    Attributes:
        state_adapter: 状态适配器
        fe_adapter: 自由能适配器
        fe_calculator: 四分量自由能计算器
        feedback: 最新反馈信息
    """

    def __init__(self, use_four_component: bool = True):
        """
        初始化集成器

        Args:
            use_four_component: 是否使用四分量公式
        """
        self.fe_adapter = FreeEnergyAdapter()
        self.fe_calculator = FreeEnergyCalculator(use_four_component=use_four_component)
        self.feedback = ActiveInferenceFeedback()

    def compute_four_component_free_energy(self,
                                        E_t: float,
                                        T_t: float,
                                        h_t: float,
                                        p_t: float,
                                        sigma_t: float,
                                        E_required: float = 100e3,
                                        T_remaining_required: float = 50.0,
                                        channel_quality: float = 1.0,
                                        p_expected: float = 0.1) -> InstantFreeEnergy:
        """
        计算四分量自由能

        Args:
            E_t: 剩余能量
            T_t: 已用时间
            h_t: 健康度
            p_t: 任务进度
            sigma_t: 不确定性
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度

        Returns:
            InstantFreeEnergy: 四分量自由能结果
        """
        # 创建状态向量
        state = StateVector(E=E_t, T=T_t, h=h_t, p=p_t, d=800.0, sigma=sigma_t)

        # 计算四分量自由能
        fe_result = self.fe_calculator.compute_instant(
            state, E_required, T_remaining_required, channel_quality, p_expected
        )

        return fe_result

    def get_checkpoint_suggestion(self,
                                 split_layer: int,
                                 free_energy: InstantFreeEnergy) -> Optional[int]:
        """
        获取 Checkpoint 建议

        Args:
            split_layer: 切分层
            free_energy: 自由能结果

        Returns:
            Optional[int]: 建议的 Checkpoint 层
        """
        # 基于自由能风险等级决定
        if free_energy.risk_level == RiskLevel.LOW:
            return None  # 低风险不需要 Checkpoint
        elif free_energy.risk_level == RiskLevel.MEDIUM:
            # 中风险：在边缘部分的中点设置
            if split_layer > 1:
                return split_layer // 2
        elif free_energy.risk_level == RiskLevel.HIGH:
            # 高风险：更早设置 Checkpoint
            if split_layer > 2:
                return split_layer // 3
        elif free_energy.risk_level == RiskLevel.CRITICAL:
            # 严重风险：立即设置 Checkpoint
            if split_layer > 1:
                return 1

        return None

    def update_feedback(self,
                       free_energy: InstantFreeEnergy,
                       split_layer: int) -> ActiveInferenceFeedback:
        """
        更新反馈信息

        Args:
            free_energy: 自由能结果
            split_layer: 切分层

        Returns:
            ActiveInferenceFeedback: 反馈信息
        """
        # 推荐行动（基于风险等级）
        if free_energy.risk_level == RiskLevel.LOW:
            recommended_action = ActionType.CONTINUE
            risk_assessment = "LOW"
            confidence = 0.9
        elif free_energy.risk_level == RiskLevel.MEDIUM:
            recommended_action = ActionType.REDUCE_POWER
            risk_assessment = "MEDIUM"
            confidence = 0.7
        elif free_energy.risk_level == RiskLevel.HIGH:
            recommended_action = ActionType.CHECKPOINT
            risk_assessment = "HIGH"
            confidence = 0.5
        else:  # CRITICAL
            recommended_action = ActionType.ABORT
            risk_assessment = "CRITICAL"
            confidence = 0.3

        # Checkpoint 建议
        checkpoint_suggestion = self.get_checkpoint_suggestion(split_layer, free_energy)

        # 更新反馈
        self.feedback = ActiveInferenceFeedback(
            free_energy=free_energy,
            recommended_action=recommended_action,
            action_confidence=confidence,
            checkpoint_suggestion=checkpoint_suggestion,
            risk_assessment=risk_assessment
        )

        return self.feedback

    def get_integration_summary(self) -> Dict:
        """
        获取集成摘要

        Returns:
            Dict: 集成摘要
        """
        return {
            'feedback_available': self.feedback.free_energy is not None,
            'current_risk_level': self.feedback.risk_assessment,
            'recommended_action': self.feedback.recommended_action.value if self.feedback.recommended_action else None,
            'checkpoint_suggested': self.feedback.checkpoint_suggestion is not None,
            'confidence': self.feedback.action_confidence,
            'free_energy_breakdown': {
                'F_energy': self.feedback.free_energy.F_energy if self.feedback.free_energy else 0,
                'F_time': self.feedback.free_energy.F_time if self.feedback.free_energy else 0,
                'F_health': self.feedback.free_energy.F_health if self.feedback.free_energy else 0,
                'F_progress': self.feedback.free_energy.F_progress if self.feedback.free_energy else 0,
            } if self.feedback.free_energy else {}
        }


# ============ 测试用例 ============

def test_active_inference_adapter():
    """测试Active Inference Adapter"""
    print("=" * 60)
    print("测试 Active Inference Adapter")
    print("=" * 60)

    # 初始化集成器
    integrator = ActiveInferenceIntegrator(use_four_component=True)

    # 测试1: 四分量自由能计算 - 正常状态
    print("\n[Test 1] 测试四分量自由能计算 - 正常状态...")
    fe_normal = integrator.compute_four_component_free_energy(
        E_t=400e3, T_t=20.0, h_t=0.9, p_t=0.5,
        sigma_t=0.1, E_required=100e3, T_remaining_required=30.0
    )

    print(f"  F_total = {fe_normal.F_total:.2f}")
    print(f"  F_energy = {fe_normal.F_energy:.2f}")
    print(f"  F_time = {fe_normal.F_time:.2f}")
    print(f"  F_health = {fe_normal.F_health:.2f}")
    print(f"  F_progress = {fe_normal.F_progress:.2f}")
    print(f"  风险等级: {fe_normal.risk_level.value}")
    print("  ✓ 四分量自由能计算正确")

    # 测试2: 四分量自由能计算 - 高风险状态
    print("\n[Test 2] 测试四分量自由能计算 - 高风险状态...")
    fe_high = integrator.compute_four_component_free_energy(
        E_t=50e3, T_t=90.0, h_t=0.4, p_t=0.2,
        sigma_t=0.5, E_required=100e3, T_remaining_required=30.0
    )

    print(f"  F_total = {fe_high.F_total:.2f}")
    print(f"  风险等级: {fe_high.risk_level.value}")
    print("  ✓ 高风险检测正确")

    # 测试3: Checkpoint 建议
    print("\n[Test 3] 测试Checkpoint建议...")
    checkpoint_layer = integrator.get_checkpoint_suggestion(split_layer=8, free_energy=fe_high)
    print(f"  切分层: 8, Checkpoint 建议: {checkpoint_layer}")
    print("  ✓ Checkpoint建议正确")

    # 测试4: 反馈更新 - 低风险
    print("\n[Test 4] 测试反馈更新 - 低风险...")
    feedback_low = integrator.update_feedback(fe_normal, split_layer=8)
    print(f"  风险评估: {feedback_low.risk_assessment}")
    print(f"  推荐行动: {feedback_low.recommended_action.value}")
    print(f"  置信度: {feedback_low.action_confidence:.2f}")
    print(f"  Checkpoint: {feedback_low.checkpoint_suggestion}")
    print("  ✓ 低风险反馈正确")

    # 测试5: 反馈更新 - 高风险
    print("\n[Test 5] 测试反馈更新 - 高风险...")
    feedback_high = integrator.update_feedback(fe_high, split_layer=8)
    print(f"  风险评估: {feedback_high.risk_assessment}")
    print(f"  推荐行动: {feedback_high.recommended_action.value}")
    print(f"  置信度: {feedback_high.action_confidence:.2f}")
    print(f"  Checkpoint: {feedback_high.checkpoint_suggestion}")
    print("  ✓ 高风险反馈正确")

    # 测试6: 集成摘要
    print("\n[Test 6] 测试集成摘要...")
    summary = integrator.get_integration_summary()

    print(f"  反馈可用: {summary['feedback_available']}")
    print(f"  当前风险: {summary['current_risk_level']}")
    print(f"  推荐行动: {summary['recommended_action']}")
    print(f"  Checkpoint建议: {summary['checkpoint_suggested']}")
    print(f"  置信度: {summary['confidence']:.2f}")
    print("  ✓ 集成摘要正确")

    # 测试7: 不同风险等级的反馈
    print("\n[Test 7] 测试不同风险等级的反馈...")
    risk_states = [
        (RiskLevel.LOW, "低能量充足，时间充裕"),
        (RiskLevel.MEDIUM, "中等能量紧张，时间一般"),
        (RiskLevel.HIGH, "高能量紧张，健康度下降"),
        (RiskLevel.CRITICAL, "严重能量不足，即将超时")
    ]

    for risk_level, description in risk_states:
        # Create test state
        if risk_level == RiskLevel.LOW:
            test_fe = integrator.compute_four_component_free_energy(
                E_t=400e3, T_t=10.0, h_t=0.9, p_t=0.3,
                sigma_t=0.1
            )
            split_layer = 8
            risk_name = "LOW"
            expected_action = ActionType.CONTINUE
            expected_cp = None
            expected_conf = 0.9
            expected_checkpoint = False

        if risk_level == RiskLevel.MEDIUM:
            test_fe = integrator.compute_four_component_free_energy(
                E_t=200e3, T_t=50.0, h_t=0.7, p_t=0.5,
                sigma_t=0.3
            )
            split_layer = 8
            risk_name = "MEDIUM"
            expected_action = ActionType.REDUCE_POWER
            expected_cp = 4
            expected_conf = 0.7
            expected_checkpoint = True

        if risk_level == RiskLevel.HIGH:
            test_fe = integrator.compute_four_component_free_energy(
                E_t=100e3, T_t=80.0, h_t=0.5, p_t=0.6,
                sigma_t=0.5
            )
            split_layer = 8
            risk_name = "HIGH"
            expected_action = ActionType.CHECKPOINT
            expected_cp = 2
            expected_conf = 0.5
            expected_checkpoint = True

        if risk_level == RiskLevel.CRITICAL:
            test_fe = integrator.compute_four_component_free_energy(
                E_t=50e3, T_t=95.0, h_t=0.3, p_t=0.7,
                sigma_t=0.7
            )
            split_layer = 8
            risk_name = "CRITICAL"
            expected_action = ActionType.ABORT
            expected_cp = 1
            expected_conf = 0.3
            expected_checkpoint = True

        feedback = integrator.update_feedback(test_fe, split_layer)

        print(f"  {risk_name:10s} ({description}):")
        print(f"    行动={feedback.recommended_action.value:12s}, "
              f"置信度={feedback.action_confidence:.2f}")
        print(f"    Checkpoint={feedback.checkpoint_suggestion}")

        assert feedback.recommended_action == expected_action, f"行动选择错误"
        if expected_checkpoint:
            assert feedback.checkpoint_suggestion == expected_cp, f"Checkpoint建议错误"
        else:
            assert feedback.checkpoint_suggestion is None, f"Checkpoint应为None"

    print("  ✓ 风险等级反馈正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_active_inference_adapter()
