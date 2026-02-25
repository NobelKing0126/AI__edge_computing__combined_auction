"""
M05: Bid - 投标结构

功能：封装投标信息，包含切分方案、资源需求、效用值等
输入：用户任务、切分层、目标UAV等参数
输出：Bid对象

关键字段 (idea118.txt 2.15节):
    Bid = (i, m, l, j, C_edge, C_cloud, D_trans, f*, f_c*, s, l_cp, x_cp, T, E, F̃, γ, η, t)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class Bid:
    """
    投标数据结构
    
    包含用户任务的一个候选卸载方案的完整信息
    
    Attributes:
        # 基本标识
        bid_id: 投标唯一标识
        user_id: 用户ID
        model_id: DNN模型ID
        cut_layer: 切分层 l ∈ [0, L]
        target_uav: 目标UAV ID
        
        # 计算量参数
        C_edge: 边缘计算量 (FLOPs)
        C_cloud: 云端计算量 (FLOPs)
        D_trans: 传输数据量 (bits)
        
        # 资源分配 (凸优化结果)
        f_edge: 请求边缘算力 (FLOPS)
        f_cloud: 请求云端算力 (FLOPS)
        
        # 信道分配
        channel_primary: 首选信道ID
        channels_backup: 备选信道ID列表
        
        # Checkpoint配置
        checkpoint_layer: Checkpoint层位置
        checkpoint_enabled: 是否启用Checkpoint
        
        # 性能预测
        T_total: 预测总时延 (s)
        T_upload: 上传时延 (s)
        T_edge: 边缘计算时延 (s)
        T_trans: 中继传输时延 (s)
        T_cloud: 云端计算时延 (s)
        T_return: 返回时延 (s)
        
        # 能耗预测
        E_total: 总能耗 (J)
        E_edge: 边缘计算能耗 (J)
        E_comm: 通信能耗 (J)
        E_checkpoint: Checkpoint能耗 (J)
        
        # 风险评估
        free_energy: 自由能 F̃
        risk_factor: 综合风险因子 γ
        
        # 效用值
        utility_stage2: 阶段2效用 η^stage2
        utility_final: 最终效用 η^final (阶段3计算)
        
        # 时间戳
        timestamp: 生成时间
    """
    # 基本标识
    bid_id: int
    user_id: int
    model_id: int
    cut_layer: int
    target_uav: int
    
    # 计算量参数
    C_edge: float  # FLOPs
    C_cloud: float  # FLOPs
    D_trans: float  # bits
    
    # 资源分配
    f_edge: float = 0.0  # FLOPS
    f_cloud: float = 0.0  # FLOPS
    
    # 信道分配
    channel_primary: int = 0
    channels_backup: List[int] = field(default_factory=list)
    
    # Checkpoint配置
    checkpoint_layer: int = 0
    checkpoint_enabled: bool = False
    
    # 时延分量
    T_total: float = 0.0
    T_upload: float = 0.0
    T_edge: float = 0.0
    T_trans: float = 0.0
    T_cloud: float = 0.0
    T_return: float = 0.0
    T_checkpoint: float = 0.0
    
    # 能耗分量
    E_total: float = 0.0
    E_edge: float = 0.0
    E_comm: float = 0.0
    E_checkpoint: float = 0.0
    
    # 风险评估
    free_energy: float = 0.0
    risk_factor: float = 0.0
    
    # 效用值
    utility_stage2: float = 0.0
    utility_final: float = 0.0
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)
    
    def is_full_edge(self) -> bool:
        """是否为全边缘方案 (l=L)"""
        return self.C_cloud == 0.0
    
    def is_full_cloud(self) -> bool:
        """是否为全云端方案 (l=0)"""
        return self.C_edge == 0.0
    
    def is_collaborative(self) -> bool:
        """是否为边云协作方案"""
        return not self.is_full_edge() and not self.is_full_cloud()
    
    def get_mode_name(self) -> str:
        """获取卸载模式名称"""
        if self.is_full_edge():
            return "全边缘"
        elif self.is_full_cloud():
            return "全云端"
        else:
            return "边云协作"
    
    def get_compute_ratio(self) -> float:
        """
        获取边缘计算比例
        
        Returns:
            float: C_edge / C_total
        """
        total = self.C_edge + self.C_cloud
        if total == 0:
            return 0.0
        return self.C_edge / total
    
    def summary(self) -> str:
        """返回投标摘要"""
        return f"""
Bid-{self.bid_id} (User-{self.user_id} → UAV-{self.target_uav}):
  模式: {self.get_mode_name()} (切分层={self.cut_layer})
  计算: 边缘={self.C_edge/1e9:.2f}G, 云端={self.C_cloud/1e9:.2f}G ({self.get_compute_ratio()*100:.1f}%边缘)
  传输: {self.D_trans/8/1e6:.2f} MB
  时延: {self.T_total*1000:.1f} ms (上传={self.T_upload*1000:.1f}, 边缘={self.T_edge*1000:.1f}, 云端={self.T_cloud*1000:.1f})
  能耗: {self.E_total/1e3:.2f} kJ
  自由能: {self.free_energy:.2f}
  效用: stage2={self.utility_stage2:.4f}, final={self.utility_final:.4f}
  Checkpoint: {'启用 (层{})'.format(self.checkpoint_layer) if self.checkpoint_enabled else '禁用'}
"""


@dataclass
class BidSet:
    """
    用户投标集合
    
    每个用户的Top-K候选投标
    
    Attributes:
        user_id: 用户ID
        bids: 投标列表 (按效用降序排列)
        best_bid: 最佳投标 (效用最高)
    """
    user_id: int
    bids: List[Bid] = field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[Bid]:
        """获取最佳投标"""
        if not self.bids:
            return None
        return max(self.bids, key=lambda b: b.utility_stage2)
    
    def add_bid(self, bid: Bid) -> None:
        """添加投标"""
        self.bids.append(bid)
    
    def sort_by_utility(self) -> None:
        """按效用降序排序"""
        self.bids.sort(key=lambda b: b.utility_stage2, reverse=True)
    
    def filter_by_uav(self, uav_id: int) -> List[Bid]:
        """筛选指定UAV的投标"""
        return [b for b in self.bids if b.target_uav == uav_id]
    
    def filter_feasible(self, tau_max: float) -> List[Bid]:
        """筛选时延可行的投标"""
        return [b for b in self.bids if b.T_total <= tau_max]
    
    def top_k(self, k: int) -> List[Bid]:
        """
        获取Top-K投标
        
        Args:
            k: 返回数量
            
        Returns:
            List[Bid]: 效用最高的k个投标
        """
        self.sort_by_utility()
        return self.bids[:k]
    
    def summary(self) -> str:
        """返回投标集摘要"""
        if not self.bids:
            return f"User-{self.user_id}: 无投标"
        
        lines = [f"User-{self.user_id} 投标集 ({len(self.bids)}个):"]
        for i, bid in enumerate(self.bids[:5]):  # 只显示前5个
            lines.append(f"  [{i+1}] UAV-{bid.target_uav}, 切分={bid.cut_layer}, "
                        f"时延={bid.T_total*1000:.1f}ms, 效用={bid.utility_stage2:.4f}")
        if len(self.bids) > 5:
            lines.append(f"  ... 还有{len(self.bids)-5}个投标")
        return "\n".join(lines)


# ============ 测试用例 ============

def test_bid():
    """测试Bid模块"""
    print("=" * 60)
    print("测试 M05: Bid")
    print("=" * 60)
    
    # 测试1: 创建投标
    print("\n[Test 1] 创建投标...")
    bid = Bid(
        bid_id=0,
        user_id=0,
        model_id=1,
        cut_layer=10,
        target_uav=0,
        C_edge=5e9,
        C_cloud=5e9,
        D_trans=1e6 * 8,  # 1 MB
        f_edge=5e9,
        f_cloud=10e9,
        T_total=0.5,
        T_upload=0.1,
        T_edge=0.2,
        T_trans=0.05,
        T_cloud=0.1,
        T_return=0.05,
        E_total=10e3,
        free_energy=15.0,
        utility_stage2=0.75
    )
    
    assert bid.is_collaborative(), "应为边云协作模式"
    print(bid.summary())
    print("  ✓ 投标创建成功")
    
    # 测试2: 测试卸载模式判断
    print("\n[Test 2] 测试卸载模式判断...")
    
    # 全边缘
    bid_edge = Bid(bid_id=1, user_id=0, model_id=1, cut_layer=20, target_uav=0,
                   C_edge=10e9, C_cloud=0, D_trans=0)
    assert bid_edge.is_full_edge(), "应为全边缘模式"
    assert bid_edge.get_mode_name() == "全边缘", "模式名称错误"
    
    # 全云端
    bid_cloud = Bid(bid_id=2, user_id=0, model_id=1, cut_layer=0, target_uav=0,
                    C_edge=0, C_cloud=10e9, D_trans=5e6)
    assert bid_cloud.is_full_cloud(), "应为全云端模式"
    
    print("  ✓ 卸载模式判断正确")
    
    # 测试3: 计算比例
    print("\n[Test 3] 测试计算比例...")
    ratio = bid.get_compute_ratio()
    assert abs(ratio - 0.5) < 1e-6, "50%边缘计算比例"
    print(f"  边缘计算比例: {ratio*100:.1f}%")
    print("  ✓ 计算比例正确")
    
    # 测试4: 投标集
    print("\n[Test 4] 测试投标集...")
    bid_set = BidSet(user_id=0)
    
    for i in range(10):
        b = Bid(
            bid_id=i,
            user_id=0,
            model_id=1,
            cut_layer=i * 2,
            target_uav=i % 3,
            C_edge=5e9 + i * 1e9,
            C_cloud=5e9 - i * 0.5e9,
            D_trans=1e6,
            utility_stage2=0.5 + i * 0.05
        )
        bid_set.add_bid(b)
    
    assert len(bid_set.bids) == 10, "应有10个投标"
    print(bid_set.summary())
    print("  ✓ 投标集创建成功")
    
    # 测试5: Top-K选择
    print("\n[Test 5] 测试Top-K选择...")
    top3 = bid_set.top_k(3)
    
    assert len(top3) == 3, "应返回3个投标"
    assert top3[0].utility_stage2 >= top3[1].utility_stage2, "应按效用降序"
    print(f"  Top-3效用: {[b.utility_stage2 for b in top3]}")
    print("  ✓ Top-K选择正确")
    
    # 测试6: UAV筛选
    print("\n[Test 6] 测试UAV筛选...")
    uav0_bids = bid_set.filter_by_uav(0)
    
    assert all(b.target_uav == 0 for b in uav0_bids), "筛选结果应都是UAV-0"
    print(f"  UAV-0的投标数: {len(uav0_bids)}")
    print("  ✓ UAV筛选正确")
    
    # 测试7: 最佳投标
    print("\n[Test 7] 测试最佳投标...")
    best = bid_set.best_bid
    
    assert best is not None, "应有最佳投标"
    assert best.utility_stage2 == max(b.utility_stage2 for b in bid_set.bids), "应为最高效用"
    print(f"  最佳投标效用: {best.utility_stage2:.4f}")
    print("  ✓ 最佳投标选择正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_bid()
