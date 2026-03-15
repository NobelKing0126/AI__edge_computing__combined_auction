"""
M01: UnifiedPricing - 统一价格模型模块

功能：实现基于idea118.txt 2.12节的统一价格模型
- 资源稀缺性定价 (指数形式)
- 风险溢价嵌入 (自由能驱动)
- 边缘/云端算力价格
- 通信/能量资源价格

公式参考 (idea118.txt 2.12节):
    边缘算力价格: P_j^comp = c_edge^base * (exp(alpha_comp * u) - 1) * exp(gamma_F * F/F_threshold)
    云端算力价格: P^cloud = c_cloud^base * (exp(alpha_cloud * u) - 1) * exp(gamma_F * F/F_threshold)

设计原则:
- 资源充裕时价格趋近于零
- 资源稀缺时价格指数上升
- 自由能嵌入形成风险溢价
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from config.system_config import SystemConfig


@dataclass
class PricingConfig:
    """
    统一价格模型配置参数

    基于idea118.txt 2.12.7节的参数设置
    """
    # ============ 基础成本参数 ============
    # 边缘计算基础成本 (元/GFLOPS)
    c_edge_base: float = 0.01

    # 云端计算基础成本 (元/GFLOPS)
    c_cloud_base: float = 0.005

    # 信道基础成本 (元/次)
    c_channel_base: float = 0.001

    # 能量基础成本 (元/kJ)
    c_energy_base: float = 0.05

    # 存储基础成本 (元/MB)
    c_storage_base: float = 0.001

    # ============ 稀缺性指数 ============
    # 边缘算力稀缺性指数
    alpha_comp: float = 3.0

    # 云端算力稀缺性指数
    alpha_cloud: float = 3.0

    # 信道稀缺性指数
    alpha_channel: float = 3.0

    # 能量稀缺性指数
    alpha_energy: float = 3.0

    # ============ 风险溢价系数 ============
    # 自由能风险溢价系数
    gamma_F: float = 0.5

    # 自由能阈值
    F_threshold: float = 30.0

    # ============ 价格范围限制 ============
    # 最大价格倍数 (防止价格过高)
    max_price_multiplier: float = 100.0

    # 最小价格 (防止零价格)
    min_price: float = 0.0001


class UnifiedPricingModel:
    """
    统一价格模型

    实现资源稀缺性定价和风险溢价

    价格特性 (alpha=3):
    - 利用率0%: 价格=0 (资源空闲，免费)
    - 利用率50%: 价格=3.48倍基础成本 (中等负载)
    - 利用率80%: 价格=10.02倍基础成本 (高负载，价格陡增)
    - 利用率100%: 价格=19.09倍基础成本 (满载，价格极高)

    风险溢价特性 (gamma_F=0.5):
    - F/F_threshold=0: 溢价=1.0 (零风险，无溢价)
    - F/F_threshold=1.0: 溢价=1.65 (中风险，价格上浮65%)
    - F/F_threshold=2.0: 溢价=2.72 (高风险，价格翻倍)
    """

    def __init__(self, config: PricingConfig = None, system_config: SystemConfig = None):
        """
        初始化统一价格模型

        Args:
            config: 价格配置
            system_config: 系统配置
        """
        self.config = config if config else PricingConfig()
        self.system_config = system_config if system_config else SystemConfig()

    def compute_scarcity_factor(self, utilization: float, alpha: float) -> float:
        """
        计算稀缺性因子

        exp(alpha * utilization) - 1

        Args:
            utilization: 资源利用率 (0-1)
            alpha: 稀缺性指数

        Returns:
            float: 稀缺性因子
        """
        utilization = np.clip(utilization, 0.0, 1.0)
        scarcity = np.exp(alpha * utilization) - 1
        return scarcity

    def compute_risk_premium(self, free_energy: float) -> float:
        """
        计算风险溢价

        exp(gamma_F * F / F_threshold)

        Args:
            free_energy: 自由能值

        Returns:
            float: 风险溢价因子
        """
        gamma_F = self.config.gamma_F
        F_threshold = self.config.F_threshold

        ratio = free_energy / F_threshold
        premium = np.exp(gamma_F * ratio)

        # 限制最大溢价
        max_premium = np.exp(gamma_F * 2.0)  # 约2.72
        return min(premium, max_premium)

    def compute_edge_compute_price(
        self,
        utilization: float,
        free_energy: float,
        f_requested: float = 1.0
    ) -> float:
        """
        计算边缘算力价格

        P_j^comp = c_edge^base * (exp(alpha_comp * u) - 1) * exp(gamma_F * F/F_threshold)

        Args:
            utilization: UAV算力利用率 (0-1)
            free_energy: 自由能值
            f_requested: 请求的算力 (GFLOPS)

        Returns:
            float: 边缘算力价格 (元)
        """
        c_base = self.config.c_edge_base
        alpha = self.config.alpha_comp

        scarcity = self.compute_scarcity_factor(utilization, alpha)
        risk_premium = self.compute_risk_premium(free_energy)

        price = c_base * scarcity * risk_premium * f_requested

        # 价格范围限制
        price = max(price, self.config.min_price)
        price = min(price, c_base * self.config.max_price_multiplier * f_requested)

        return price

    def compute_cloud_compute_price(
        self,
        utilization: float,
        free_energy: float,
        f_requested: float = 1.0
    ) -> float:
        """
        计算云端算力价格

        P^cloud = c_cloud^base * (exp(alpha_cloud * u) - 1) * exp(gamma_F * F/F_threshold)

        Args:
            utilization: 云端算力利用率 (0-1)
            free_energy: 自由能值
            f_requested: 请求的算力 (GFLOPS)

        Returns:
            float: 云端算力价格 (元)
        """
        c_base = self.config.c_cloud_base
        alpha = self.config.alpha_cloud

        scarcity = self.compute_scarcity_factor(utilization, alpha)
        risk_premium = self.compute_risk_premium(free_energy)

        price = c_base * scarcity * risk_premium * f_requested

        # 价格范围限制
        price = max(price, self.config.min_price)
        price = min(price, c_base * self.config.max_price_multiplier * f_requested)

        return price

    def compute_channel_price(
        self,
        channel_utilization: float,
        free_energy: float
    ) -> float:
        """
        计算通信资源价格

        P_s^channel = c_channel^base * (exp(alpha_channel * u) - 1) * exp(gamma_F * F/F_threshold)

        Args:
            channel_utilization: 信道利用率 (0-1)
            free_energy: 自由能值

        Returns:
            float: 信道价格 (元)
        """
        c_base = self.config.c_channel_base
        alpha = self.config.alpha_channel

        scarcity = self.compute_scarcity_factor(channel_utilization, alpha)
        risk_premium = self.compute_risk_premium(free_energy)

        price = c_base * scarcity * risk_premium

        return max(price, self.config.min_price)

    def compute_energy_price(
        self,
        energy_utilization: float,
        free_energy: float,
        energy_consumed: float = 1.0
    ) -> float:
        """
        计算能量价格

        P_j^energy = c_energy^base * (exp(alpha_energy * u) - 1) * exp(gamma_F * F/F_threshold)

        Args:
            energy_utilization: 能量利用率 (0-1)
            free_energy: 自由能值
            energy_consumed: 消耗的能量 (kJ)

        Returns:
            float: 能量价格 (元)
        """
        c_base = self.config.c_energy_base
        alpha = self.config.alpha_energy

        scarcity = self.compute_scarcity_factor(energy_utilization, alpha)
        risk_premium = self.compute_risk_premium(free_energy)

        price = c_base * scarcity * risk_premium * energy_consumed

        return max(price, self.config.min_price)

    def compute_checkpoint_price(self, checkpoint_size: float) -> float:
        """
        计算Checkpoint价格

        P_checkpoint = c_storage * S_checkpoint

        Args:
            checkpoint_size: Checkpoint数据量 (MB)

        Returns:
            float: Checkpoint价格 (元)
        """
        return self.config.c_storage_base * checkpoint_size

    def compute_total_price(
        self,
        edge_utilization: float,
        cloud_utilization: float,
        channel_utilization: float,
        energy_utilization: float,
        free_energy: float,
        f_edge_requested: float,
        f_cloud_requested: float,
        energy_consumed: float,
        checkpoint_enabled: bool = False,
        checkpoint_size: float = 0.0
    ) -> Dict[str, float]:
        """
        计算方案总价格

        P_total = P_comp * f_edge + P_cloud * f_cloud + P_channel + P_energy * E_total + x_cp * P_checkpoint

        Args:
            edge_utilization: 边缘算力利用率
            cloud_utilization: 云端算力利用率
            channel_utilization: 信道利用率
            energy_utilization: 能量利用率
            free_energy: 自由能值
            f_edge_requested: 请求的边缘算力 (GFLOPS)
            f_cloud_requested: 请求的云端算力 (GFLOPS)
            energy_consumed: 消耗的能量 (kJ)
            checkpoint_enabled: 是否启用Checkpoint
            checkpoint_size: Checkpoint数据量 (MB)

        Returns:
            Dict[str, float]: 各项价格明细和总价
        """
        # 计算各项价格
        p_edge = self.compute_edge_compute_price(
            edge_utilization, free_energy, f_edge_requested
        )

        p_cloud = self.compute_cloud_compute_price(
            cloud_utilization, free_energy, f_cloud_requested
        )

        p_channel = self.compute_channel_price(channel_utilization, free_energy)

        p_energy = self.compute_energy_price(
            energy_utilization, free_energy, energy_consumed
        )

        p_checkpoint = 0.0
        if checkpoint_enabled and checkpoint_size > 0:
            p_checkpoint = self.compute_checkpoint_price(checkpoint_size)

        total_price = p_edge + p_cloud + p_channel + p_energy + p_checkpoint

        return {
            'edge_compute_price': p_edge,
            'cloud_compute_price': p_cloud,
            'channel_price': p_channel,
            'energy_price': p_energy,
            'checkpoint_price': p_checkpoint,
            'total_price': total_price
        }

    def get_scarcity_table(self, utilizations: List[float] = None) -> Dict[float, float]:
        """
        获取稀缺性因子表

        用于展示不同利用率下的稀缺性因子

        Args:
            utilizations: 利用率列表

        Returns:
            Dict[float, float]: {利用率: 稀缺性因子}
        """
        if utilizations is None:
            utilizations = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

        alpha = self.config.alpha_comp
        table = {}

        for u in utilizations:
            scarcity = self.compute_scarcity_factor(u, alpha)
            table[u] = scarcity

        return table

    def get_risk_premium_table(self, free_energies: List[float] = None) -> Dict[float, float]:
        """
        获取风险溢价表

        用于展示不同自由能下的风险溢价

        Args:
            free_energies: 自由能值列表

        Returns:
            Dict[float, float]: {自由能: 风险溢价}
        """
        if free_energies is None:
            free_energies = [0, 10, 20, 30, 40, 50, 60]

        table = {}

        for f in free_energies:
            premium = self.compute_risk_premium(f)
            table[f] = premium

        return table

    # ============ V2修正版方法 ============

    def compute_normalized_price(
        self,
        utilization: float,
        alpha: float = None
    ) -> float:
        """
        计算归一化价格 (修正版V2)

        P_norm = (exp(α*u) - 1) / (exp(α) - 1)

        修正要点: 价格范围归一化到[0, 1]
        - u=0: P_norm = 0 (资源空闲)
        - u=1: P_norm = 1 (资源满载)

        Args:
            utilization: 资源利用率 (0-1)
            alpha: 稀缺性指数

        Returns:
            float: 归一化价格 [0, 1]
        """
        if alpha is None:
            alpha = self.config.alpha_comp

        utilization = np.clip(utilization, 0.0, 1.0)

        numerator = np.exp(alpha * utilization) - 1
        denominator = np.exp(alpha) - 1

        if denominator <= 0:
            return 0.0

        return numerator / denominator

    def compute_resource_price(
        self,
        utilization: float,
        free_energy: float,
        c_base: float,
        alpha: float = None
    ) -> float:
        """
        计算资源价格 (修正版V2)

        P = P_norm * φ(F)

        其中:
        - P_norm = (exp(α*u) - 1) / (exp(α) - 1) ∈ [0, 1]
        - φ(F) = exp(γ_F * F / F_threshold) 为风险溢价

        Args:
            utilization: 资源利用率 (0-1)
            free_energy: 自由能值
            c_base: 基础成本
            alpha: 稀缺性指数

        Returns:
            float: 资源价格
        """
        # 归一化价格
        P_norm = self.compute_normalized_price(utilization, alpha)

        # 风险溢价
        risk_premium = self.compute_risk_premium(free_energy)

        # 最终价格
        price = c_base * P_norm * risk_premium

        # 价格范围限制
        price = max(price, self.config.min_price)
        price = min(price, c_base * self.config.max_price_multiplier)

        return price

    def compute_scheme_total_price_v2(
        self,
        edge_utilization: float,
        cloud_utilization: float,
        channel_utilization: float,
        energy_utilization: float,
        free_energy: float,
        f_edge_requested: float = 1.0,
        f_cloud_requested: float = 1.0,
        energy_consumed: float = 1.0,
        checkpoint_enabled: bool = False,
        checkpoint_size: float = 0.0,
        normalization_divisor: int = 4
    ) -> Dict[str, float]:
        """
        计算方案总价格 (修正版V2 - 归一化)

        P_total_normalized = P_total / 4

        修正要点: 除以资源类型数(4)归一化
        - 4种资源类型：边缘算力、云端算力、信道、能量

        Args:
            edge_utilization: 边缘算力利用率
            cloud_utilization: 云端算力利用率
            channel_utilization: 信道利用率
            energy_utilization: 能量利用率
            free_energy: 自由能值
            f_edge_requested: 请求的边缘算力 (GFLOPS)
            f_cloud_requested: 请求的云端算力 (GFLOPS)
            energy_consumed: 消耗的能量 (kJ)
            checkpoint_enabled: 是否启用Checkpoint
            checkpoint_size: Checkpoint数据量 (MB)
            normalization_divisor: 归一化除数（默认4种资源）

        Returns:
            Dict[str, float]: 各项价格明细和归一化总价
        """
        # 计算各项价格
        p_edge = self.compute_resource_price(
            edge_utilization, free_energy,
            self.config.c_edge_base, self.config.alpha_comp
        ) * f_edge_requested

        p_cloud = self.compute_resource_price(
            cloud_utilization, free_energy,
            self.config.c_cloud_base, self.config.alpha_cloud
        ) * f_cloud_requested

        p_channel = self.compute_resource_price(
            channel_utilization, free_energy,
            self.config.c_channel_base, self.config.alpha_channel
        )

        p_energy = self.compute_resource_price(
            energy_utilization, free_energy,
            self.config.c_energy_base, self.config.alpha_energy
        ) * energy_consumed

        p_checkpoint = 0.0
        if checkpoint_enabled and checkpoint_size > 0:
            p_checkpoint = self.compute_checkpoint_price(checkpoint_size)

        # 计算总价
        total_price = p_edge + p_cloud + p_channel + p_energy + p_checkpoint

        # 归一化总价
        total_price_normalized = total_price / normalization_divisor

        return {
            'edge_compute_price': p_edge,
            'cloud_compute_price': p_cloud,
            'channel_price': p_channel,
            'energy_price': p_energy,
            'checkpoint_price': p_checkpoint,
            'total_price': total_price,
            'total_price_normalized': total_price_normalized  # 新增归一化总价
        }


# ============ 便捷函数 ============

def compute_edge_compute_price(
    c_edge_base: float,
    alpha_comp: float,
    f_used: float,
    f_max: float,
    gamma_F: float,
    F_tilde: float,
    F_threshold: float
) -> float:
    """
    计算边缘算力价格 (便捷函数)

    P_j^comp = c_base * (exp(alpha * u) - 1) * exp(gamma_F * F/F_threshold)

    Args:
        c_edge_base: 边缘计算基础成本 (元/GFLOPS)
        alpha_comp: 稀缺性指数
        f_used: 已使用算力 (GFLOPS)
        f_max: 最大算力 (GFLOPS)
        gamma_F: 风险溢价系数
        F_tilde: 自由能值
        F_threshold: 自由能阈值

    Returns:
        float: 边缘算力价格 (元/GFLOPS)
    """
    utilization = f_used / f_max if f_max > 0 else 0
    scarcity = np.exp(alpha_comp * utilization) - 1
    risk_premium = np.exp(gamma_F * F_tilde / F_threshold)

    return c_edge_base * scarcity * risk_premium


def compute_cloud_compute_price(
    c_cloud_base: float,
    alpha_cloud: float,
    F_used: float,
    F_max: float,
    gamma_F: float,
    F_tilde: float,
    F_threshold: float
) -> float:
    """
    计算云端算力价格 (便捷函数)

    P^cloud = c_base * (exp(alpha * u) - 1) * exp(gamma_F * F/F_threshold)

    Args:
        c_cloud_base: 云端计算基础成本 (元/GFLOPS)
        alpha_cloud: 稀缺性指数
        F_used: 已使用算力 (GFLOPS)
        F_max: 最大算力 (GFLOPS)
        gamma_F: 风险溢价系数
        F_tilde: 自由能值
        F_threshold: 自由能阈值

    Returns:
        float: 云端算力价格 (元/GFLOPS)
    """
    utilization = F_used / F_max if F_max > 0 else 0
    scarcity = np.exp(alpha_cloud * utilization) - 1
    risk_premium = np.exp(gamma_F * F_tilde / F_threshold)

    return c_cloud_base * scarcity * risk_premium


# ============ 测试用例 ============

def test_unified_pricing():
    """测试UnifiedPricing模块"""
    print("=" * 60)
    print("测试 UnifiedPricing")
    print("=" * 60)

    # 创建模型
    model = UnifiedPricingModel()

    # 测试1: 稀缺性因子表 (参考idea118.txt 2.12.8节)
    print("\n[Test 1] 稀缺性因子表 (alpha=3)")
    print("-" * 40)
    scarcity_table = model.get_scarcity_table()
    print(f"  {'利用率':<10} {'稀缺性因子':<15} {'含义':<20}")
    print(f"  {'-'*10} {'-'*15} {'-'*20}")

    meanings = {
        0.0: "资源空闲，免费",
        0.2: "轻负载",
        0.4: "中等负载",
        0.5: "中等负载",
        0.6: "较高负载",
        0.8: "高负载，价格陡增",
        1.0: "满载，价格极高"
    }

    for u, scarcity in scarcity_table.items():
        meaning = meanings.get(u, "")
        print(f"  {u:<10.0%} {scarcity:<15.2f} {meaning:<20}")

    # 测试2: 风险溢价表 (参考idea118.txt 2.12.8节)
    print("\n[Test 2] 风险溢价表 (gamma_F=0.5)")
    print("-" * 40)
    premium_table = model.get_risk_premium_table()
    print(f"  {'自由能':<10} {'F/F_threshold':<15} {'风险溢价':<15} {'含义':<20}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*20}")

    F_threshold = model.config.F_threshold
    premium_meanings = {
        0: "零风险，无溢价",
        30: "中风险，价格上浮65%",
        60: "高风险，价格翻倍"
    }

    for f, premium in premium_table.items():
        ratio = f / F_threshold
        meaning = premium_meanings.get(f, "")
        print(f"  {f:<10.0f} {ratio:<15.2f} {premium:<15.2f} {meaning:<20}")

    # 测试3: 边缘算力价格
    print("\n[Test 3] 边缘算力价格")
    print("-" * 40)
    utilizations = [0.3, 0.5, 0.7, 0.9]
    free_energies = [0, 15, 30]

    print(f"  {'利用率':<10}", end="")
    for f in free_energies:
        print(f" {'F=' + str(f):<12}", end="")
    print()

    for u in utilizations:
        print(f"  {u:<10.0%}", end="")
        for f in free_energies:
            price = model.compute_edge_compute_price(u, f, 1.0)
            print(f" {price:<12.4f}", end="")
        print()

    # 测试4: 方案总价格
    print("\n[Test 4] 方案总价格计算")
    print("-" * 40)

    price_breakdown = model.compute_total_price(
        edge_utilization=0.6,
        cloud_utilization=0.4,
        channel_utilization=0.3,
        energy_utilization=0.5,
        free_energy=20.0,
        f_edge_requested=5.0,  # 5 GFLOPS
        f_cloud_requested=10.0,  # 10 GFLOPS
        energy_consumed=50.0,  # 50 kJ
        checkpoint_enabled=True,
        checkpoint_size=100.0  # 100 MB
    )

    print(f"  输入参数:")
    print(f"    边缘利用率: 60%")
    print(f"    云端利用率: 40%")
    print(f"    信道利用率: 30%")
    print(f"    能量利用率: 50%")
    print(f"    自由能: 20.0")
    print(f"    请求边缘算力: 5.0 GFLOPS")
    print(f"    请求云端算力: 10.0 GFLOPS")
    print(f"    能耗: 50.0 kJ")
    print(f"    Checkpoint: 100.0 MB")
    print()
    print(f"  价格明细:")
    for key, value in price_breakdown.items():
        print(f"    {key}: {value:.4f} 元")

    # 测试5: 价格对比
    print("\n[Test 5] 不同场景价格对比")
    print("-" * 40)

    scenarios = [
        {"name": "低负载+低风险", "util": 0.2, "fe": 5.0},
        {"name": "中负载+中风险", "util": 0.5, "fe": 20.0},
        {"name": "高负载+高风险", "util": 0.8, "fe": 40.0},
    ]

    print(f"  {'场景':<20} {'边缘价格':<12} {'云端价格':<12}")
    for s in scenarios:
        p_edge = model.compute_edge_compute_price(s["util"], s["fe"], 1.0)
        p_cloud = model.compute_cloud_compute_price(s["util"], s["fe"], 1.0)
        print(f"  {s['name']:<20} {p_edge:<12.4f} {p_cloud:<12.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_unified_pricing()
