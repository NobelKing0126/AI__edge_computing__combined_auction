"""
M22: DynamicPricing - 动态定价

功能：根据资源利用率动态调整资源价格
输入：UAV状态、利用率
输出：更新后的价格

关键公式 (idea118.txt 4.6节):
    算力价格: p_f(t+1) = p_f(t) * (1 + γ_f * (u_f - u_target))
    能量价格: p_E(t+1) = p_E(t) * (1 + γ_E * (u_E - u_target))
    
    平滑: p_new = λ*p_computed + (1-λ)*p_old
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import PRICING


@dataclass
class ResourcePrices:
    """
    资源价格
    
    Attributes:
        p_compute: 算力价格 ($/FLOPS)
        p_energy: 能量价格 ($/J)
        p_channel: 信道价格 ($/bps)
        timestamp: 时间戳
    """
    p_compute: float
    p_energy: float
    p_channel: float
    timestamp: float = 0.0


@dataclass
class UtilizationInfo:
    """
    利用率信息
    
    Attributes:
        uav_id: UAV ID
        compute_util: 算力利用率 [0, 1]
        energy_util: 能量利用率 [0, 1]
        channel_util: 信道利用率 [0, 1]
        load_count: 当前负载任务数
    """
    uav_id: int
    compute_util: float
    energy_util: float
    channel_util: float
    load_count: int


class DynamicPricingManager:
    """
    动态定价管理器
    
    Attributes:
        gamma_compute: 算力价格调整系数
        gamma_energy: 能量价格调整系数
        gamma_channel_up: 信道价格上涨系数
        gamma_channel_down: 信道价格下降系数
        util_target: 目标利用率
        smooth_lambda: 价格平滑系数
        min_interval: 最小更新间隔
    """
    
    def __init__(self,
                 gamma_compute: float = 1.0,
                 gamma_energy: float = 0.3,
                 gamma_channel_up: float = 0.2,
                 gamma_channel_down: float = 0.1,
                 util_target: float = 0.7,
                 smooth_lambda: float = 0.3,
                 min_interval: float = 1.0):
        """
        初始化定价管理器
        """
        self.gamma_compute = gamma_compute
        self.gamma_energy = gamma_energy
        self.gamma_channel_up = gamma_channel_up
        self.gamma_channel_down = gamma_channel_down
        self.util_target = util_target
        self.smooth_lambda = smooth_lambda
        self.min_interval = min_interval
        
        # 每个UAV的价格
        self.uav_prices: Dict[int, ResourcePrices] = {}
        self.price_history: Dict[int, List[ResourcePrices]] = {}
        self.last_update_time: Dict[int, float] = {}
    
    def initialize_prices(self, 
                         uav_ids: List[int],
                         base_compute: Optional[float] = None,
                         base_energy: Optional[float] = None,
                         base_channel: Optional[float] = None) -> Dict[int, ResourcePrices]:
        """
        初始化UAV价格
        
        Args:
            uav_ids: UAV ID列表
            base_compute: 基础算力价格，默认使用常量配置
            base_energy: 基础能量价格，默认使用常量配置
            base_channel: 基础信道价格，默认使用常量配置
            
        Returns:
            Dict: {uav_id: 价格}
        """
        # 使用常量作为默认值
        base_compute = base_compute if base_compute is not None else PRICING.BASE_COMPUTE_PRICE
        base_energy = base_energy if base_energy is not None else PRICING.BASE_ENERGY_PRICE
        base_channel = base_channel if base_channel is not None else PRICING.BASE_CHANNEL_PRICE
        
        for uid in uav_ids:
            self.uav_prices[uid] = ResourcePrices(
                p_compute=base_compute,
                p_energy=base_energy,
                p_channel=base_channel,
                timestamp=0.0
            )
            self.price_history[uid] = [self.uav_prices[uid]]
            self.last_update_time[uid] = 0.0
        
        return self.uav_prices.copy()
    
    def update_price(self,
                     uav_id: int,
                     utilization: UtilizationInfo,
                     current_time: float) -> Optional[ResourcePrices]:
        """
        更新单个UAV的价格
        
        Args:
            uav_id: UAV ID
            utilization: 利用率信息
            current_time: 当前时间
            
        Returns:
            Optional[ResourcePrices]: 新价格，None表示未更新
        """
        if uav_id not in self.uav_prices:
            return None
        
        # 检查更新间隔
        if current_time - self.last_update_time.get(uav_id, 0) < self.min_interval:
            return None
        
        old_prices = self.uav_prices[uav_id]
        
        # 计算新价格（使用常量配置的最小价格）
        # 算力价格
        compute_delta = utilization.compute_util - self.util_target
        new_p_compute = old_prices.p_compute * (1 + self.gamma_compute * compute_delta)
        new_p_compute = max(new_p_compute, PRICING.MIN_COMPUTE_PRICE)
        
        # 能量价格
        energy_delta = utilization.energy_util - self.util_target
        new_p_energy = old_prices.p_energy * (1 + self.gamma_energy * energy_delta)
        new_p_energy = max(new_p_energy, PRICING.MIN_ENERGY_PRICE)
        
        # 信道价格
        channel_delta = utilization.channel_util - self.util_target
        if channel_delta > 0:
            new_p_channel = old_prices.p_channel * (1 + self.gamma_channel_up * channel_delta)
        else:
            new_p_channel = old_prices.p_channel * (1 + self.gamma_channel_down * channel_delta)
        new_p_channel = max(new_p_channel, PRICING.MIN_CHANNEL_PRICE)
        
        # 平滑处理
        smoothed_prices = ResourcePrices(
            p_compute=self.smooth_lambda * new_p_compute + 
                     (1 - self.smooth_lambda) * old_prices.p_compute,
            p_energy=self.smooth_lambda * new_p_energy + 
                    (1 - self.smooth_lambda) * old_prices.p_energy,
            p_channel=self.smooth_lambda * new_p_channel + 
                     (1 - self.smooth_lambda) * old_prices.p_channel,
            timestamp=current_time
        )
        
        self.uav_prices[uav_id] = smoothed_prices
        self.price_history[uav_id].append(smoothed_prices)
        self.last_update_time[uav_id] = current_time
        
        return smoothed_prices
    
    def update_all_prices(self,
                          utilizations: List[UtilizationInfo],
                          current_time: float) -> Dict[int, ResourcePrices]:
        """
        更新所有UAV价格
        
        Args:
            utilizations: 利用率信息列表
            current_time: 当前时间
            
        Returns:
            Dict: 更新后的价格
        """
        updated = {}
        for util in utilizations:
            new_price = self.update_price(util.uav_id, util, current_time)
            if new_price:
                updated[util.uav_id] = new_price
        
        return updated
    
    def get_price(self, uav_id: int) -> Optional[ResourcePrices]:
        """获取UAV当前价格"""
        return self.uav_prices.get(uav_id)
    
    def get_all_prices(self) -> Dict[int, ResourcePrices]:
        """获取所有UAV价格"""
        return self.uav_prices.copy()
    
    def compute_cost(self,
                     uav_id: int,
                     compute_used: float,
                     energy_used: float,
                     channel_used: float) -> float:
        """
        计算资源使用成本
        
        Args:
            uav_id: UAV ID
            compute_used: 使用的算力
            energy_used: 使用的能量
            channel_used: 使用的信道
            
        Returns:
            float: 总成本
        """
        prices = self.get_price(uav_id)
        if prices is None:
            return 0.0
        
        return (prices.p_compute * compute_used +
                prices.p_energy * energy_used +
                prices.p_channel * channel_used)
    
    def get_price_stats(self, uav_id: int) -> Dict:
        """获取价格统计"""
        if uav_id not in self.price_history:
            return {}
        
        history = self.price_history[uav_id]
        
        return {
            'current': self.uav_prices[uav_id],
            'initial': history[0],
            'updates': len(history),
            'compute_change': (history[-1].p_compute / history[0].p_compute - 1) * 100,
            'energy_change': (history[-1].p_energy / history[0].p_energy - 1) * 100
        }


# ============ 测试用例 ============

def test_dynamic_pricing():
    """测试DynamicPricing模块"""
    print("=" * 60)
    print("测试 M22: DynamicPricing")
    print("=" * 60)
    
    manager = DynamicPricingManager()
    
    # 测试1: 初始化价格
    print("\n[Test 1] 测试价格初始化...")
    prices = manager.initialize_prices([0, 1, 2])
    
    assert len(prices) == 3
    assert all(p.p_compute > 0 for p in prices.values())
    print(f"  初始化了 {len(prices)} 个UAV的价格")
    print("  ✓ 价格初始化正确")
    
    # 测试2: 高利用率价格上涨
    print("\n[Test 2] 测试高利用率场景...")
    
    high_util = UtilizationInfo(
        uav_id=0,
        compute_util=0.9,  # 高于目标0.7
        energy_util=0.8,
        channel_util=0.85,
        load_count=5
    )
    
    old_price = manager.get_price(0)
    new_price = manager.update_price(0, high_util, 1.0)
    
    assert new_price is not None
    assert new_price.p_compute > old_price.p_compute
    print(f"  算力价格变化: {old_price.p_compute:.2e} -> {new_price.p_compute:.2e}")
    print("  ✓ 高利用率价格上涨正确")
    
    # 测试3: 低利用率价格下降
    print("\n[Test 3] 测试低利用率场景...")
    
    low_util = UtilizationInfo(
        uav_id=1,
        compute_util=0.3,  # 低于目标0.7
        energy_util=0.2,
        channel_util=0.25,
        load_count=1
    )
    
    old_price = manager.get_price(1)
    new_price = manager.update_price(1, low_util, 1.0)
    
    assert new_price.p_compute < old_price.p_compute
    print(f"  算力价格变化: {old_price.p_compute:.2e} -> {new_price.p_compute:.2e}")
    print("  ✓ 低利用率价格下降正确")
    
    # 测试4: 批量更新
    print("\n[Test 4] 测试批量更新...")
    
    utils = [
        UtilizationInfo(0, 0.8, 0.7, 0.6, 3),
        UtilizationInfo(1, 0.5, 0.4, 0.3, 2),
        UtilizationInfo(2, 0.9, 0.85, 0.8, 4),
    ]
    
    updated = manager.update_all_prices(utils, 3.0)
    
    print(f"  更新了 {len(updated)} 个UAV的价格")
    print("  ✓ 批量更新正确")
    
    # 测试5: 成本计算
    print("\n[Test 5] 测试成本计算...")
    
    cost = manager.compute_cost(
        uav_id=0,
        compute_used=5e9,
        energy_used=50e3,
        channel_used=10e6
    )
    
    assert cost > 0
    print(f"  资源成本: {cost:.6f}")
    print("  ✓ 成本计算正确")
    
    # 测试6: 价格统计
    print("\n[Test 6] 测试价格统计...")
    stats = manager.get_price_stats(0)
    
    print(f"  更新次数: {stats['updates']}")
    print(f"  算力价格变化: {stats['compute_change']:.1f}%")
    print("  ✓ 价格统计正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dynamic_pricing()
