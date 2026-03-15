"""
论文基线算法复现模块

用于复现相关论文的核心算法，与本项目方法进行对比

包含:
- MAPPO-Attention: Sub-THz Networks: Multi-Agent PPO with Attention Mechan

V2版本包含完整的DRL训练过程:
- MAPPO-Attention-V2: 完整MAPPO训练

设计原则:
1. 继承现有 BaselineAlgorithm 接口，完全兼容
2. 参数从配置文件读取，不硬编码
3. 复用现有基础设施（信道模型、能耗计算等)
4. 支持在线学习和预训练模型加载

注: Lyapunov-DRL 已于 2026-03-14 移除
"""

# 原始版本（简化版，无完整训练）
from .mappo_attention import MAPPOAttentionBaseline

# V2版本（完整DRL训练）
try:
    from .mappo_attention_v2 import MAPPOAttentionBaselineV2
    DRL_TRAINING_AVAILABLE = True
except ImportError:
    DRL_TRAINING_AVAILABLE = False
    MAPPOAttentionBaselineV2 = MAPPOAttentionBaseline

__all__ = [
    # 原始版本
    'MAPPOAttentionBaseline',

    # V2版本（完整训练）
    'MAPPOAttentionBaselineV2',

    # 便捷函数
    'get_paper_baselines',
    'create_paper_baseline_runner',
    'DRL_TRAINING_AVAILABLE'
]


def get_paper_baselines() -> list:
    """
    获取所有论文基线算法实例

    Returns:
        List[BaselineAlgorithm]: 论文基线算法列表

    Usage:
        baselines = get_paper_baselines()
        for baseline in baselines:
            result = baseline.run(tasks, uav_resources, cloud_resources)
    """
    return [
        MAPPOAttentionBaseline(),
    ]


def create_paper_baseline_runner():
    """
    创建包含论文基线的运行器

    Returns:
        PaperBaselineRunner 实例

    Usage:
        runner = create_paper_baseline_runner()
        results = runner.run_all(tasks, uav_resources, cloud_resources)
    """
    return PaperBaselineRunner()


class PaperBaselineRunner:
    """
    论文基线运行器

    与现有 BaselineRunner 接口兼容，可直接替换使用
    """

    def __init__(self):
        """初始化论文基线运行器"""
        self.baselines = get_paper_baselines()

    def run_all(
        self,
        tasks: list,
        uav_resources: list,
        cloud_resources: dict
    ) -> dict:
        """
        运行所有论文基线算法

        Args:
            tasks: 任务列表
            uav_resources: UAV资源列表
            cloud_resources: 云端资源

        Returns:
            Dict[str, BaselineResult]: {算法名: 结果}
        """
        results = {}

        for baseline in self.baselines:
            result = baseline.run(tasks, uav_resources, cloud_resources)
            results[baseline.name] = result

        return results

    def run_single(
        self,
        baseline_name: str,
        tasks: list,
        uav_resources: list,
        cloud_resources: dict
    ):
        """
        运行单个论文基线算法

        Args:
            baseline_name: 基线算法名称
            tasks: 任务列表
            uav_resources: UAV资源列表
            cloud_resources: 云端资源

        Returns:
            BaselineResult
        """
        name_mapping = {
            'MAPPO-Attention': 'MAPPO-Attention',
        }

        target_name = name_mapping.get(baseline_name, baseline_name)

        for baseline in self.baselines:
            if baseline.name == target_name:
                return baseline.run(tasks, uav_resources, cloud_resources)

        raise ValueError(f"未找到论文基线算法: {baseline_name}")

    def print_comparison(self, results: dict):
        """打印对比结果"""
        print("\n" + "=" * 90)
        print("Paper Baseline Comparison")
        print("=" * 90)

        print(f"\n{'Algorithm':<20} {'Success%':>10} {'AvgDelay':>12} {'Energy':>10} {'HiPrio%':>10} {'SW':>10}")
        print("-" * 90)

        for name, result in results.items():
            print(f"{name:<20} {result.success_rate*100:>9.1f}% "
                  f"{result.avg_delay*1000:>10.1f}ms "
                  f"{result.total_energy:>9.2f}J "
                  f"{result.high_priority_rate*100:>9.1f}% "
                  f"{result.social_welfare:>10.2f}")

        print("=" * 90)
