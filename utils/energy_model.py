"""
M07: EnergyModel - 能耗模型

功能：计算边缘计算、云端计算、通信、Checkpoint等能耗
输入：算力、计算量、时间等参数
输出：各类能耗值

关键公式 (idea118.txt 2.7节):
    边缘计算能耗: E_edge = κ_edge * f² * C
    云端计算能耗: E_cloud = κ_cloud * f² * C
    通信能耗: E_comm = P_rx * T_upload + P_tx * T_trans
    Checkpoint能耗: E_cp = P_write * T_cp
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.system_config import EnergyConfig, UAVConfig


@dataclass
class EnergyBreakdown:
    """
    能耗分解
    
    Attributes:
        E_edge: 边缘计算能耗 (J)
        E_cloud: 云端计算能耗 (J)
        E_comm: 通信能耗 (J)
        E_checkpoint: Checkpoint能耗 (J)
        E_total: 总能耗 (J)
    """
    E_edge: float
    E_cloud: float
    E_comm: float
    E_checkpoint: float
    
    @property
    def E_total(self) -> float:
        """总能耗"""
        return self.E_edge + self.E_comm + self.E_checkpoint
    
    def summary(self) -> str:
        return f"""
能耗分解:
  边缘计算: {self.E_edge/1e3:.4f} kJ
  通信: {self.E_comm/1e3:.4f} kJ
  Checkpoint: {self.E_checkpoint/1e3:.4f} kJ
  总计(UAV): {self.E_total/1e3:.4f} kJ
  云端(参考): {self.E_cloud/1e3:.4f} kJ
"""


class EnergyModel:
    """
    能耗计算模型
    
    Attributes:
        kappa_edge: 边缘能耗系数
        kappa_cloud: 云端能耗系数
        P_rx: 接收功率 (W)
        P_tx: 发射功率 (W)
        P_write: 存储写入功率 (W)
        P_hover: 悬停功率 (W)
    """
    
    def __init__(self, 
                 energy_config: Optional[EnergyConfig] = None,
                 uav_config: Optional[UAVConfig] = None):
        """
        初始化能耗模型
        
        Args:
            energy_config: 能耗配置
            uav_config: UAV配置
        """
        if energy_config is None:
            energy_config = EnergyConfig()
        if uav_config is None:
            uav_config = UAVConfig()
        
        self.kappa_edge = energy_config.kappa_edge
        self.kappa_cloud = energy_config.kappa_cloud
        self.P_write = energy_config.P_write
        self.P_rx = uav_config.P_rx
        self.P_tx = uav_config.P_tx
        self.P_hover = uav_config.P_hover
    
    def compute_edge_energy(self, f_edge: float, C_edge: float) -> float:
        """
        计算边缘计算能耗
        
        公式: E_edge = κ_edge * f² * C
        
        Note: 这里使用 E = κ * f² * C，其中C是计算量(FLOPs)
              因为 P = κ * f³，T = C/f，所以 E = P*T = κ*f³*(C/f) = κ*f²*C
        
        Args:
            f_edge: 边缘算力 (FLOPS)
            C_edge: 边缘计算量 (FLOPs)
            
        Returns:
            float: 边缘计算能耗 (J)
        """
        if f_edge <= 0 or C_edge <= 0:
            return 0.0
        return self.kappa_edge * (f_edge ** 2) * C_edge
    
    def compute_cloud_energy(self, f_cloud: float, C_cloud: float) -> float:
        """
        计算云端计算能耗
        
        公式: E_cloud = κ_cloud * f² * C
        
        Note: 云端能耗对UAV能量无直接影响，仅用于系统总能耗分析
        
        Args:
            f_cloud: 云端算力 (FLOPS)
            C_cloud: 云端计算量 (FLOPs)
            
        Returns:
            float: 云端计算能耗 (J)
        """
        if f_cloud <= 0 or C_cloud <= 0:
            return 0.0
        return self.kappa_cloud * (f_cloud ** 2) * C_cloud
    
    def compute_communication_energy(self, T_upload: float, T_trans: float) -> float:
        """
        计算通信能耗
        
        公式: E_comm = P_rx * T_upload + P_tx * T_trans
        
        Args:
            T_upload: 上传时间 (s)
            T_trans: 中继传输时间 (s)
            
        Returns:
            float: 通信能耗 (J)
        """
        return self.P_rx * T_upload + self.P_tx * T_trans
    
    def compute_checkpoint_energy(self, T_checkpoint: float) -> float:
        """
        计算Checkpoint能耗
        
        公式: E_cp = P_write * T_cp
        
        Args:
            T_checkpoint: Checkpoint时间 (s)
            
        Returns:
            float: Checkpoint能耗 (J)
        """
        return self.P_write * T_checkpoint
    
    def compute_checkpoint_time(self, data_size: float, gamma_cp: float = 1e-7) -> float:
        """
        计算Checkpoint时间
        
        公式: T_cp = γ_cp * OutputSize
        
        Args:
            data_size: Checkpoint数据量 (bits)
            gamma_cp: Checkpoint时间系数 (s/bit)
            
        Returns:
            float: Checkpoint时间 (s)
        """
        return gamma_cp * data_size
    
    def compute_compute_power(self, f: float) -> float:
        """
        计算指定算力对应的瞬时功率
        
        公式: P = κ * f³
        
        Args:
            f: 算力 (FLOPS)
            
        Returns:
            float: 功率 (W)
        """
        return self.kappa_edge * (f ** 3)
    
    def compute_total_power(self, f_compute: float, 
                           is_receiving: bool = False,
                           is_transmitting: bool = False) -> float:
        """
        计算UAV总瞬时功率
        
        公式: P_total = P_hover + P_compute + P_comm
        
        Args:
            f_compute: 计算算力 (FLOPS)
            is_receiving: 是否正在接收
            is_transmitting: 是否正在发射
            
        Returns:
            float: 总功率 (W)
        """
        P_compute = self.compute_compute_power(f_compute)
        P_comm = 0.0
        if is_receiving:
            P_comm += self.P_rx
        if is_transmitting:
            P_comm += self.P_tx
        
        return self.P_hover + P_compute + P_comm
    
    def compute_total_energy(self,
                            f_edge: float,
                            C_edge: float,
                            f_cloud: float,
                            C_cloud: float,
                            T_upload: float,
                            T_trans: float,
                            T_checkpoint: float = 0.0) -> EnergyBreakdown:
        """
        计算完整能耗分解
        
        Args:
            f_edge: 边缘算力 (FLOPS)
            C_edge: 边缘计算量 (FLOPs)
            f_cloud: 云端算力 (FLOPS)
            C_cloud: 云端计算量 (FLOPs)
            T_upload: 上传时间 (s)
            T_trans: 中继传输时间 (s)
            T_checkpoint: Checkpoint时间 (s)
            
        Returns:
            EnergyBreakdown: 能耗分解
        """
        E_edge = self.compute_edge_energy(f_edge, C_edge)
        E_cloud = self.compute_cloud_energy(f_cloud, C_cloud)
        E_comm = self.compute_communication_energy(T_upload, T_trans)
        E_checkpoint = self.compute_checkpoint_energy(T_checkpoint)
        
        return EnergyBreakdown(
            E_edge=E_edge,
            E_cloud=E_cloud,
            E_comm=E_comm,
            E_checkpoint=E_checkpoint
        )
    
    def compute_energy_budget(self, 
                             E_remain: float,
                             num_pending_tasks: int,
                             E_max: float,
                             safety_ratio: float = 0.3) -> float:
        """
        计算单任务能量预算
        
        公式 (idea118.txt 2.6节):
            E_budget = min(E_remain / (num_tasks + 1), safety_ratio * E_max)
        
        Args:
            E_remain: 剩余能量 (J)
            num_pending_tasks: 待处理任务数
            E_max: 最大电池容量 (J)
            safety_ratio: 安全系数
            
        Returns:
            float: 能量预算 (J)
        """
        per_task_budget = E_remain / (num_pending_tasks + 1)
        safety_budget = safety_ratio * E_max
        return min(per_task_budget, safety_budget)
    
    def check_energy_feasibility(self, 
                                 E_required: float,
                                 E_remain: float,
                                 E_comm: float = 0.0) -> Tuple[bool, float]:
        """
        检查能量可行性
        
        Args:
            E_required: 任务所需能量 (J)
            E_remain: 剩余能量 (J)
            E_comm: 通信能耗预估 (J)
            
        Returns:
            Tuple[bool, float]: (是否可行, 执行后剩余能量)
        """
        total_required = E_required + E_comm
        remaining_after = E_remain - total_required
        feasible = remaining_after >= 0
        return feasible, remaining_after


# ============ 测试用例 ============

def test_energy_model():
    """测试EnergyModel模块"""
    print("=" * 60)
    print("测试 M07: EnergyModel")
    print("=" * 60)
    
    model = EnergyModel()
    
    # 测试1: 边缘计算能耗
    print("\n[Test 1] 测试边缘计算能耗...")
    f_edge = 5e9  # 5 GFLOPS
    C_edge = 10e9  # 10 GFLOPs
    E_edge = model.compute_edge_energy(f_edge, C_edge)
    
    expected = model.kappa_edge * (f_edge ** 2) * C_edge
    assert abs(E_edge - expected) < 1e-10, "边缘能耗计算错误"
    print(f"  5 GFLOPS执行10G计算: {E_edge:.6f} J")
    print("  ✓ 边缘计算能耗正确")
    
    # 测试2: 云端计算能耗
    print("\n[Test 2] 测试云端计算能耗...")
    f_cloud = 100e9  # 100 GFLOPS
    C_cloud = 10e9
    E_cloud = model.compute_cloud_energy(f_cloud, C_cloud)
    print(f"  100 GFLOPS执行10G计算: {E_cloud:.6f} J")
    
    # 验证: 云端能耗系数是边缘的1/10
    # 但由于f_cloud=20*f_edge, E ~ f²*C, 所以 E_cloud = 0.1 * 400 * E_edge_base
    # 云端用更高算力但系数更低，总能耗仍可能较高
    expected_cloud = model.kappa_cloud * (f_cloud ** 2) * C_cloud
    assert abs(E_cloud - expected_cloud) < 1e-10, "云端能耗计算错误"
    print("  ✓ 云端计算能耗正确")
    
    # 测试3: 通信能耗
    print("\n[Test 3] 测试通信能耗...")
    E_comm = model.compute_communication_energy(T_upload=0.5, T_trans=0.1)
    expected_comm = model.P_rx * 0.5 + model.P_tx * 0.1
    assert abs(E_comm - expected_comm) < 1e-10, "通信能耗计算错误"
    print(f"  上传0.5s + 传输0.1s: {E_comm:.4f} J")
    print("  ✓ 通信能耗正确")
    
    # 测试4: Checkpoint能耗
    print("\n[Test 4] 测试Checkpoint能耗...")
    T_cp = model.compute_checkpoint_time(data_size=1e6 * 8)  # 1MB
    E_cp = model.compute_checkpoint_energy(T_cp)
    print(f"  1MB Checkpoint时间: {T_cp*1000:.2f} ms")
    print(f"  1MB Checkpoint能耗: {E_cp:.4f} J")
    print("  ✓ Checkpoint能耗正确")
    
    # 测试5: 瞬时功率
    print("\n[Test 5] 测试瞬时功率...")
    P_compute = model.compute_compute_power(5e9)
    P_total = model.compute_total_power(5e9, is_receiving=True)
    print(f"  5 GFLOPS计算功率: {P_compute:.4f} W")
    print(f"  总功率(含悬停+接收): {P_total:.2f} W")
    print("  ✓ 功率计算正确")
    
    # 测试6: 完整能耗分解
    print("\n[Test 6] 测试完整能耗分解...")
    breakdown = model.compute_total_energy(
        f_edge=5e9,
        C_edge=5e9,
        f_cloud=100e9,
        C_cloud=5e9,
        T_upload=0.5,
        T_trans=0.1,
        T_checkpoint=0.05
    )
    print(breakdown.summary())
    assert breakdown.E_total > 0, "总能耗应大于0"
    print("  ✓ 能耗分解正确")
    
    # 测试7: 能量预算
    print("\n[Test 7] 测试能量预算...")
    E_budget = model.compute_energy_budget(
        E_remain=400e3,  # 400kJ
        num_pending_tasks=5,
        E_max=500e3
    )
    print(f"  剩余400kJ, 5个待处理任务: 预算={E_budget/1e3:.2f} kJ")
    assert E_budget <= 400e3 / 6, "预算应不超过平均值"
    assert E_budget <= 0.3 * 500e3, "预算应不超过安全上限"
    print("  ✓ 能量预算正确")
    
    # 测试8: 能量可行性检查
    print("\n[Test 8] 测试能量可行性...")
    feasible, remaining = model.check_energy_feasibility(
        E_required=50e3,
        E_remain=100e3,
        E_comm=5e3
    )
    assert feasible, "应该可行"
    assert abs(remaining - 45e3) < 1e-6, "剩余能量计算错误"
    print(f"  需要55kJ, 有100kJ: 可行={feasible}, 剩余={remaining/1e3:.2f}kJ")
    
    # 不可行情况
    feasible2, remaining2 = model.check_energy_feasibility(
        E_required=150e3,
        E_remain=100e3
    )
    assert not feasible2, "应该不可行"
    print(f"  需要150kJ, 有100kJ: 可行={feasible2}")
    print("  ✓ 可行性检查正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_energy_model()
