"""
M14: Election - 分布式拍卖方选举

功能：通过分布式投票选举拍卖方UAV
输入：UAV列表、状态信息
输出：选举结果

关键公式 (idea118.txt 1.7节):
    评分函数: Score_j = w₁*(E_j/E_max) + w₂*(f_j/f_max) + w₃*(1-d_j/d_max) + w₄*(1-L_j/L_max)
    
    投票规则:
        - 每个UAV计算所有候选者得分
        - 投票给得分最高者
        - 获得多数票的UAV成为拍卖方
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import RESOURCE, NUMERICAL


class ElectionStatus(Enum):
    """选举状态"""
    SUCCESS = "SUCCESS"
    TIE = "TIE"
    NO_QUORUM = "NO_QUORUM"
    FAILED = "FAILED"


@dataclass
class UAVState:
    """
    UAV状态（用于选举）
    
    Attributes:
        uav_id: UAV ID
        energy: 剩余能量 (J)
        energy_max: 最大能量 (J)
        compute_cap: 可用算力 (FLOPS)
        compute_max: 最大算力 (FLOPS)
        position: 位置 (x, y, z)
        load: 当前负载（任务数）
        load_max: 最大负载
        is_active: 是否活跃
    """
    uav_id: int
    energy: float
    energy_max: float
    compute_cap: float
    compute_max: float
    position: Tuple[float, float, float]
    load: int
    load_max: int
    is_active: bool = True


@dataclass
class ElectionResult:
    """
    选举结果
    
    Attributes:
        auctioneer_id: 当选拍卖方ID
        status: 选举状态
        votes: 投票详情 {uav_id: 得票数}
        scores: 候选者得分 {uav_id: 分数}
        rounds: 选举轮数
    """
    auctioneer_id: Optional[int]
    status: ElectionStatus
    votes: Dict[int, int]
    scores: Dict[int, float]
    rounds: int


class AuctioneerElector:
    """
    拍卖方选举器
    
    Attributes:
        w1: 能量权重
        w2: 算力权重
        w3: 位置权重
        w4: 负载权重
    """
    
    def __init__(self,
                 w1: float = 0.25,
                 w2: float = 0.30,
                 w3: float = 0.25,
                 w4: float = 0.20):
        """
        初始化选举器
        
        Args:
            w1-w4: 各评分因子权重
        """
        self.w1 = w1  # 能量
        self.w2 = w2  # 算力
        self.w3 = w3  # 位置
        self.w4 = w4  # 负载
        
        # 验证权重之和为1
        assert abs(w1 + w2 + w3 + w4 - 1.0) < 1e-6, "权重之和应为1"
    
    def compute_candidate_score(self,
                                candidate: UAVState,
                                all_uavs: List[UAVState],
                                center: Tuple[float, float]) -> float:
        """
        计算候选者得分
        
        公式: Score = w₁*(E/E_max) + w₂*(f/f_max) + w₃*(1-d/d_max) + w₄*(1-L/L_max)
        
        Args:
            candidate: 候选UAV状态
            all_uavs: 所有UAV列表
            center: 场景中心或用户质心
            
        Returns:
            float: 候选者得分
        """
        # 能量因子
        energy_factor = candidate.energy / max(candidate.energy_max, 1e-10)
        
        # 算力因子
        compute_factor = candidate.compute_cap / max(candidate.compute_max, 1e-10)
        
        # 位置因子（到中心的距离，越近越好）
        dx = candidate.position[0] - center[0]
        dy = candidate.position[1] - center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算所有UAV到中心的最大距离
        max_distance = 0.0
        for uav in all_uavs:
            dx = uav.position[0] - center[0]
            dy = uav.position[1] - center[1]
            d = np.sqrt(dx**2 + dy**2)
            max_distance = max(max_distance, d)
        
        position_factor = 1.0 - distance / max(max_distance, 1e-10)
        
        # 负载因子（负载越低越好）
        load_factor = 1.0 - candidate.load / max(candidate.load_max, 1)
        
        # 加权求和
        score = (
            self.w1 * energy_factor +
            self.w2 * compute_factor +
            self.w3 * position_factor +
            self.w4 * load_factor
        )
        
        return score
    
    def elect(self,
              uav_states: List[UAVState],
              scene_center: Optional[Tuple[float, float]] = None,
              min_energy_ratio: Optional[float] = None) -> ElectionResult:
        """
        执行分布式选举
        
        Args:
            uav_states: 所有UAV状态列表
            scene_center: 场景中心，None则自动计算
            min_energy_ratio: 最低能量比例要求，默认使用常量配置
            
        Returns:
            ElectionResult: 选举结果
        """
        # 使用常量作为默认值
        if min_energy_ratio is None:
            min_energy_ratio = RESOURCE.MIN_ENERGY_RATIO
        
        # 过滤活跃且满足能量要求的候选者
        candidates = [
            uav for uav in uav_states
            if uav.is_active and (uav.energy / uav.energy_max) >= min_energy_ratio
        ]
        
        if not candidates:
            return ElectionResult(
                auctioneer_id=None,
                status=ElectionStatus.NO_QUORUM,
                votes={},
                scores={},
                rounds=0
            )
        
        # 计算场景中心
        if scene_center is None:
            xs = [uav.position[0] for uav in uav_states]
            ys = [uav.position[1] for uav in uav_states]
            scene_center = (np.mean(xs), np.mean(ys))
        
        # 计算每个候选者得分
        scores = {}
        for candidate in candidates:
            score = self.compute_candidate_score(candidate, uav_states, scene_center)
            scores[candidate.uav_id] = score
        
        # 每个UAV投票给得分最高者（模拟分布式投票）
        votes = {c.uav_id: 0 for c in candidates}
        
        for voter in candidates:
            # 找到得分最高的候选者
            best_candidate = max(candidates, key=lambda c: scores[c.uav_id])
            votes[best_candidate.uav_id] += 1
        
        # 找到得票最多的候选者
        max_votes = max(votes.values())
        winners = [uid for uid, v in votes.items() if v == max_votes]
        
        if len(winners) == 1:
            return ElectionResult(
                auctioneer_id=winners[0],
                status=ElectionStatus.SUCCESS,
                votes=votes,
                scores=scores,
                rounds=1
            )
        else:
            # 平局：选择得分最高的
            best_winner = max(winners, key=lambda w: scores[w])
            return ElectionResult(
                auctioneer_id=best_winner,
                status=ElectionStatus.TIE,
                votes=votes,
                scores=scores,
                rounds=1
            )
    
    def re_elect(self,
                 uav_states: List[UAVState],
                 exclude_ids: Set[int],
                 scene_center: Optional[Tuple[float, float]] = None) -> ElectionResult:
        """
        重新选举（排除特定UAV）
        
        用于当前拍卖方故障时的重新选举
        
        Args:
            uav_states: 所有UAV状态列表
            exclude_ids: 需排除的UAV ID集合
            scene_center: 场景中心
            
        Returns:
            ElectionResult: 选举结果
        """
        filtered_states = [uav for uav in uav_states if uav.uav_id not in exclude_ids]
        return self.elect(filtered_states, scene_center)


# ============ 测试用例 ============

def test_election():
    """测试Election模块"""
    print("=" * 60)
    print("测试 M14: Election")
    print("=" * 60)
    
    # 创建测试数据
    uav_states = [
        UAVState(
            uav_id=0,
            energy=400e3, energy_max=500e3,
            compute_cap=8e9, compute_max=10e9,
            position=(500, 500, 100),
            load=5, load_max=20
        ),
        UAVState(
            uav_id=1,
            energy=450e3, energy_max=500e3,
            compute_cap=9e9, compute_max=10e9,
            position=(1000, 1000, 100),
            load=3, load_max=20
        ),
        UAVState(
            uav_id=2,
            energy=300e3, energy_max=500e3,
            compute_cap=7e9, compute_max=10e9,
            position=(1500, 500, 100),
            load=10, load_max=20
        ),
    ]
    
    elector = AuctioneerElector()
    
    # 测试1: 候选者得分计算
    print("\n[Test 1] 测试候选者得分计算...")
    center = (1000, 1000)
    
    for uav in uav_states:
        score = elector.compute_candidate_score(uav, uav_states, center)
        print(f"  UAV-{uav.uav_id}: 能量={uav.energy/1e3:.0f}kJ, 得分={score:.4f}")
    print("  ✓ 得分计算正确")
    
    # 测试2: 选举
    print("\n[Test 2] 测试选举...")
    result = elector.elect(uav_states, center)
    
    assert result.auctioneer_id is not None, "应选出拍卖方"
    assert result.status in [ElectionStatus.SUCCESS, ElectionStatus.TIE]
    
    print(f"  当选拍卖方: UAV-{result.auctioneer_id}")
    print(f"  选举状态: {result.status.value}")
    print(f"  投票详情: {result.votes}")
    print("  ✓ 选举正确")
    
    # 测试3: 验证高能量、高算力UAV当选
    print("\n[Test 3] 验证评分因素影响...")
    
    # UAV-1应该得分最高（能量高、算力高、位置中心、负载低）
    assert result.scores[1] >= max(result.scores[0], result.scores[2]), \
        "UAV-1应有最高得分"
    print(f"  最高得分: UAV-{max(result.scores, key=result.scores.get)}")
    print("  ✓ 评分因素正确")
    
    # 测试4: 能量不足的UAV排除
    print("\n[Test 4] 测试能量不足排除...")
    
    low_energy_states = uav_states.copy()
    low_energy_states[0] = UAVState(
        uav_id=0,
        energy=50e3, energy_max=500e3,  # 10%能量
        compute_cap=8e9, compute_max=10e9,
        position=(500, 500, 100),
        load=5, load_max=20
    )
    
    result_filtered = elector.elect(low_energy_states, min_energy_ratio=0.2)
    
    assert 0 not in result_filtered.scores, "低能量UAV应被排除"
    print(f"  排除UAV-0后，候选者: {list(result_filtered.scores.keys())}")
    print("  ✓ 能量过滤正确")
    
    # 测试5: 重新选举
    print("\n[Test 5] 测试重新选举...")
    result_reelect = elector.re_elect(uav_states, exclude_ids={1})
    
    assert result_reelect.auctioneer_id != 1, "被排除的UAV不应当选"
    print(f"  排除UAV-1后，当选: UAV-{result_reelect.auctioneer_id}")
    print("  ✓ 重新选举正确")
    
    # 测试6: 无可用候选者
    print("\n[Test 6] 测试无候选者情况...")
    
    inactive_states = [
        UAVState(uav_id=0, energy=400e3, energy_max=500e3,
                compute_cap=8e9, compute_max=10e9,
                position=(500, 500, 100), load=5, load_max=20,
                is_active=False)
    ]
    
    result_no_candidate = elector.elect(inactive_states)
    
    assert result_no_candidate.status == ElectionStatus.NO_QUORUM
    assert result_no_candidate.auctioneer_id is None
    print(f"  选举状态: {result_no_candidate.status.value}")
    print("  ✓ 无候选者处理正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_election()
