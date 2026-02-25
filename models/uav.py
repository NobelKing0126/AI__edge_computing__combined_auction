"""
M04: UAV - 无人机模型

功能：定义UAV状态和资源管理
输入：UAV位置、算力、能量等参数
输出：UAV对象，支持资源分配和状态跟踪

关键属性 (idea118.txt 0.11节):
    State_j = (P_j, f_max, f_avail, E_remain, L_j, LoadedModels_j)
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple
from enum import Enum, auto
import numpy as np


class UAVStatus(Enum):
    """
    UAV健康状态
    
    参考 (idea118.txt 4.8节)
    """
    HEALTHY = auto()    # 正常运行 H >= 0.7
    WARNING = auto()    # 警告状态 0.5 <= H < 0.7
    CRITICAL = auto()   # 临界状态 0.3 <= H < 0.5
    FAILED = auto()     # 故障状态 H < 0.3


class ConnectionStatus(Enum):
    """
    UAV连接状态
    
    参考 (idea118.txt 1.7节)
    """
    DIRECT = auto()        # 直接连接
    RELAY = auto()         # 中继连接
    DISCONNECTED = auto()  # 断开连接


@dataclass
class UAV:
    """
    无人机实体
    
    Attributes:
        uav_id: UAV唯一标识
        x: x坐标 (m)
        y: y坐标 (m)
        height: 飞行高度 (m)
        f_max: 最大算力 (FLOPS)
        E_max: 电池容量 (J)
        
        # 动态状态
        f_avail: 可用算力 (FLOPS)
        E_remain: 剩余能量 (J)
        load_rate: 负载率
        loaded_models: 已加载的模型ID集合
        
        # 运行时属性
        status: 健康状态
        connection_status: 连接状态
        active_tasks: 正在执行的任务ID列表
        is_auctioneer: 是否为拍卖方
    """
    uav_id: int
    x: float  # 位置x坐标 (m)
    y: float  # 位置y坐标 (m)
    height: float = 100.0  # 飞行高度 (m)
    f_max: float = 10e9  # 最大算力 10 GFLOPS
    E_max: float = 500e3  # 电池容量 500 kJ
    
    # 动态状态
    f_avail: float = field(default=None)  # 可用算力
    E_remain: float = field(default=None)  # 剩余能量
    loaded_models: Set[int] = field(default_factory=set)  # 已加载模型
    
    # 悬停和飞行功率
    P_hover: float = 150.0  # 悬停功率 (W)
    P_fly: float = 200.0  # 飞行功率 (W)
    v_fly: float = 10.0  # 飞行速度 (m/s)
    
    # 通信功率
    P_rx: float = 0.1  # 接收功率 (W)
    P_tx: float = 0.5  # 发射功率 (W)
    
    # 模型加载限制
    M_max: int = 5  # 最大模型加载数
    
    # 运行时属性
    status: UAVStatus = UAVStatus.HEALTHY
    connection_status: ConnectionStatus = ConnectionStatus.DIRECT
    active_tasks: List[int] = field(default_factory=list)
    is_auctioneer: bool = False
    is_backup_auctioneer: bool = False
    
    # 拍卖方预留资源
    reserved_compute: float = 0.0
    reserved_energy: float = 0.0
    
    def __post_init__(self):
        """初始化默认值"""
        if self.f_avail is None:
            self.f_avail = self.f_max
        if self.E_remain is None:
            self.E_remain = self.E_max
    
    @property
    def position(self) -> Tuple[float, float]:
        """获取二维位置"""
        return (self.x, self.y)
    
    @property
    def position_3d(self) -> Tuple[float, float, float]:
        """获取三维位置"""
        return (self.x, self.y, self.height)
    
    @property
    def load_rate(self) -> float:
        """
        计算负载率
        
        公式: L_j = f_used / f_max = (f_max - f_avail) / f_max
        """
        return (self.f_max - self.f_avail) / self.f_max
    
    @property
    def energy_ratio(self) -> float:
        """
        计算能量剩余比例
        
        公式: r_energy = E_remain / E_max
        """
        return self.E_remain / self.E_max
    
    def distance_to(self, x: float, y: float) -> float:
        """
        计算到用户的三维距离
        
        公式: d_{i,j} = sqrt((x_i-x_j)² + (y_i-y_j)² + H²)
        
        Args:
            x: 用户x坐标
            y: 用户y坐标
            
        Returns:
            float: 三维距离 (m)
        """
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + self.height ** 2)
    
    def distance_to_uav(self, other: 'UAV') -> float:
        """
        计算到另一个UAV的距离
        
        Args:
            other: 另一个UAV
            
        Returns:
            float: 三维距离 (m)
        """
        return np.sqrt((self.x - other.x) ** 2 + 
                      (self.y - other.y) ** 2 + 
                      (self.height - other.height) ** 2)
    
    def can_load_model(self, model_id: int) -> bool:
        """
        检查是否可以加载新模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 是否可以加载
        """
        if model_id in self.loaded_models:
            return True  # 已加载
        return len(self.loaded_models) < self.M_max
    
    def load_model(self, model_id: int) -> bool:
        """
        加载模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 是否成功加载
        """
        if model_id in self.loaded_models:
            return True
        if len(self.loaded_models) >= self.M_max:
            return False
        self.loaded_models.add(model_id)
        return True
    
    def allocate_compute(self, compute: float) -> bool:
        """
        分配算力
        
        Args:
            compute: 请求的算力 (FLOPS)
            
        Returns:
            bool: 是否成功分配
        """
        if compute > self.f_avail:
            return False
        self.f_avail -= compute
        return True
    
    def release_compute(self, compute: float) -> None:
        """
        释放算力
        
        Args:
            compute: 释放的算力 (FLOPS)
        """
        self.f_avail = min(self.f_max, self.f_avail + compute)
    
    def consume_energy(self, energy: float) -> bool:
        """
        消耗能量
        
        Args:
            energy: 消耗的能量 (J)
            
        Returns:
            bool: 是否有足够能量
        """
        if energy > self.E_remain:
            return False
        self.E_remain -= energy
        return True
    
    def get_compute_power(self, compute: float, kappa_edge: float = 1e-28) -> float:
        """
        计算指定算力对应的功率
        
        公式: P = κ * f³
        
        Args:
            compute: 算力 (FLOPS)
            kappa_edge: 边缘能耗系数
            
        Returns:
            float: 功率 (W)
        """
        return kappa_edge * (compute ** 3)
    
    def get_available_compute_power_budget(self, 
                                           P_max: float = 300.0,
                                           time_window: float = 1.0) -> float:
        """
        获取计算功率预算
        
        公式: P_comp_budget = (P_max - P_hover) * T_window
        
        Args:
            P_max: 最大总功率 (W)
            time_window: 时间窗口 (s)
            
        Returns:
            float: 计算能量预算 (J)
        """
        return (P_max - self.P_hover) * time_window
    
    def set_as_auctioneer(self, gamma_f: float = 0.05, gamma_E: float = 0.02) -> None:
        """
        设置为拍卖方，预留资源
        
        公式 (idea118.txt 1.6节):
            f_reserved = γ_f * f_max
            E_reserved = γ_E * E_max
        
        Args:
            gamma_f: 算力预留比例
            gamma_E: 能量预留比例
        """
        self.is_auctioneer = True
        self.reserved_compute = gamma_f * self.f_max
        self.reserved_energy = gamma_E * self.E_max
        
        # 从可用资源中扣除
        self.f_avail -= self.reserved_compute
        self.E_remain -= self.reserved_energy
    
    def compute_health(self) -> float:
        """
        计算综合健康度
        
        公式 (idea118.txt 4.8节):
            H_j = w_E * H_energy + w_L * H_load + w_C * H_comm
        
        Returns:
            float: 健康度 [0, 1]
        """
        w_E, w_L, w_C = 0.5, 0.3, 0.2
        
        H_energy = self.energy_ratio
        H_load = 1.0 - self.load_rate
        H_comm = 1.0 if self.connection_status == ConnectionStatus.DIRECT else 0.5
        
        return w_E * H_energy + w_L * H_load + w_C * H_comm
    
    def update_status(self) -> None:
        """
        根据健康度更新状态
        
        状态分级 (idea118.txt 4.8节):
            H >= 0.7: HEALTHY
            0.5 <= H < 0.7: WARNING
            0.3 <= H < 0.5: CRITICAL
            H < 0.3: FAILED
        """
        health = self.compute_health()
        
        if health >= 0.7:
            self.status = UAVStatus.HEALTHY
        elif health >= 0.5:
            self.status = UAVStatus.WARNING
        elif health >= 0.3:
            self.status = UAVStatus.CRITICAL
        else:
            self.status = UAVStatus.FAILED
    
    def get_flight_energy(self, target_x: float, target_y: float) -> float:
        """
        计算飞往目标位置所需能量
        
        公式: E_fly = P_fly * distance / v_fly
        
        Args:
            target_x: 目标x坐标
            target_y: 目标y坐标
            
        Returns:
            float: 飞行能量 (J)
        """
        distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)
        return self.P_fly * distance / self.v_fly
    
    def move_to(self, target_x: float, target_y: float) -> bool:
        """
        移动到目标位置
        
        Args:
            target_x: 目标x坐标
            target_y: 目标y坐标
            
        Returns:
            bool: 是否成功移动
        """
        energy_needed = self.get_flight_energy(target_x, target_y)
        
        if energy_needed > self.E_remain:
            return False
        
        self.E_remain -= energy_needed
        self.x = target_x
        self.y = target_y
        return True
    
    def summary(self) -> str:
        """返回UAV状态摘要"""
        return f"""
UAV-{self.uav_id} 状态:
  位置: ({self.x:.1f}, {self.y:.1f}, {self.height:.1f})
  算力: {self.f_avail/1e9:.2f}/{self.f_max/1e9:.2f} GFLOPS ({(1-self.load_rate)*100:.1f}%可用)
  能量: {self.E_remain/1e3:.1f}/{self.E_max/1e3:.1f} kJ ({self.energy_ratio*100:.1f}%)
  健康度: {self.compute_health():.2f} ({self.status.name})
  已加载模型: {self.loaded_models}
  活跃任务: {len(self.active_tasks)}
  拍卖方: {'是' if self.is_auctioneer else '否'}
"""


# ============ 测试用例 ============

def test_uav_model():
    """测试UAV模块"""
    print("=" * 60)
    print("测试 M04: UAV")
    print("=" * 60)
    
    # 测试1: 创建UAV
    print("\n[Test 1] 创建UAV...")
    uav = UAV(uav_id=0, x=1000.0, y=1000.0)
    
    assert uav.f_avail == uav.f_max, "初始可用算力应等于最大算力"
    assert uav.E_remain == uav.E_max, "初始能量应等于最大能量"
    assert uav.load_rate == 0.0, "初始负载率应为0"
    print(uav.summary())
    print("  ✓ UAV创建成功")
    
    # 测试2: 距离计算
    print("\n[Test 2] 测试距离计算...")
    dist = uav.distance_to(500.0, 500.0)
    expected = np.sqrt(500**2 + 500**2 + 100**2)
    assert abs(dist - expected) < 1e-6, "距离计算错误"
    print(f"  到用户(500,500)的距离: {dist:.2f} m")
    print("  ✓ 距离计算正确")
    
    # 测试3: 算力分配
    print("\n[Test 3] 测试算力分配...")
    initial_avail = uav.f_avail
    result = uav.allocate_compute(2e9)  # 分配2 GFLOPS
    
    assert result, "应成功分配2 GFLOPS"
    assert abs(uav.f_avail - (initial_avail - 2e9)) < 1e-6, "可用算力应减少"
    assert uav.load_rate > 0, "负载率应大于0"
    print(f"  分配后负载率: {uav.load_rate*100:.1f}%")
    print("  ✓ 算力分配正确")
    
    # 测试4: 能量消耗
    print("\n[Test 4] 测试能量消耗...")
    initial_energy = uav.E_remain
    result = uav.consume_energy(10e3)  # 消耗10 kJ
    
    assert result, "应成功消耗能量"
    assert abs(uav.E_remain - (initial_energy - 10e3)) < 1e-6, "剩余能量应减少"
    print(f"  剩余能量: {uav.energy_ratio*100:.1f}%")
    print("  ✓ 能量消耗正确")
    
    # 测试5: 模型加载
    print("\n[Test 5] 测试模型加载...")
    for i in range(5):
        result = uav.load_model(i)
        assert result, f"应成功加载模型{i}"
    
    result = uav.load_model(5)
    assert not result, "超过限制后不应加载成功"
    
    result = uav.load_model(2)  # 已加载的模型
    assert result, "重复加载已有模型应成功"
    
    print(f"  已加载模型: {uav.loaded_models}")
    print("  ✓ 模型加载正确")
    
    # 测试6: 健康度计算
    print("\n[Test 6] 测试健康度计算...")
    health = uav.compute_health()
    assert 0 <= health <= 1, "健康度应在[0,1]范围"
    uav.update_status()
    print(f"  健康度: {health:.2f}")
    print(f"  状态: {uav.status.name}")
    print("  ✓ 健康度计算正确")
    
    # 测试7: 拍卖方设置
    print("\n[Test 7] 测试拍卖方设置...")
    uav2 = UAV(uav_id=1, x=500.0, y=500.0)
    uav2.set_as_auctioneer()
    
    assert uav2.is_auctioneer, "应设置为拍卖方"
    assert uav2.reserved_compute > 0, "应预留算力"
    assert uav2.f_avail < uav2.f_max, "可用算力应减少"
    print(f"  预留算力: {uav2.reserved_compute/1e9:.2f} GFLOPS")
    print(f"  预留能量: {uav2.reserved_energy/1e3:.2f} kJ")
    print("  ✓ 拍卖方设置正确")
    
    # 测试8: 飞行能耗
    print("\n[Test 8] 测试飞行能耗...")
    uav3 = UAV(uav_id=2, x=0.0, y=0.0)
    energy = uav3.get_flight_energy(100.0, 0.0)  # 飞100米
    expected_energy = uav3.P_fly * 100.0 / uav3.v_fly
    
    assert abs(energy - expected_energy) < 1e-6, "飞行能耗计算错误"
    print(f"  飞行100m能耗: {energy/1e3:.2f} kJ")
    print("  ✓ 飞行能耗计算正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_uav_model()
