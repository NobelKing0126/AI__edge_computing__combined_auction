"""
论文基线算法集成指南

本文件展示如何将论文算法集成到 run_real_experiments_v9.py 中

使用方法:
1. 在 run_real_experiments_v9.py 顶部添加导入:
   from experiments.paper_baselines import get_paper_baselines

2. 在 RealExperimentRunnerV9.__init__ 中添加:
   self.paper_baselines = get_paper_baselines()

3. 在 run_exp1 的 baselines 列表中添加:
   baselines = [..., "Lyapunov-DRL", "MAPPO-Attention"]

4. 在 run_single_baseline 调用前添加论文算法处理逻辑

详细修改见下方代码。
"""

# ============ 修改1: 导入模块 ============
"""
在 run_real_experiments_v9.py 顶部 (约第60行) 添加:
"""

IMPORT_CODE = '''
# 导入论文基线算法
from experiments.paper_baselines import (
    get_paper_baselines,
    MAPPOAttentionBaseline
)
# Lyapunov-DRL 已移除 (2026-03-14)
'''


# ============ 修改2: RealExperimentRunnerV9 初始化 ============
"""
在 RealExperimentRunnerV9.__init__ 中 (约第260行) 添加:
"""

INIT_CODE = '''
        # 初始化论文基线算法
        # Lyapunov-DRL 已移除 (2026-03-14)
        self.paper_baselines = {
            'MAPPO-Attention': MAPPOAttentionBaseline()
        }
'''


# ============ 修改3: run_exp1 基线列表 ============
"""
在 run_exp1 方法中 (约第680行) 修改 baselines 列表:
"""

EXP1_BASELINES = '''
        baselines = [
            "Proposed", "Edge-Only", "Cloud-Only", "Greedy",
            "Fixed-Split", "Random-Auction", "No-ActiveInference",
            "Heuristic-Alloc", "No-DynPricing", "B11-FixedPrice",
            "B11a-HighFixed", "B11b-LowFixed", "B12-DelayOpt",
            "MAPPO-Attention"    # 论文算法
        ]
'''


# ============ 修改4: 添加论文算法运行逻辑 ============
"""
在 run_exp1 的 for baseline in baselines 循环中 (约第685行) 添加:
"""

RUN_PAPER_BASELINE_CODE = '''
        for baseline in baselines:
            print(f"运行 {baseline}...")

            # 检查是否是论文基线
            if baseline in self.paper_baselines:
                try:
                    result = self.paper_baselines[baseline].run(
                        task_dicts, uav_resources, cloud_resources
                    )
                    metrics = self._extract_full_metrics(result, offline_sw)
                    results[baseline] = ExperimentResult(baseline, scenario.name, metrics)
                    print(f"  SW={metrics.social_welfare:.2f}, Success={metrics.success_rate*100:.1f}%")
                except Exception as e:
                    print(f"  [Error] {e}")
                continue

            # 原有基线逻辑
            try:
                result = self.baseline_runner.run_single_baseline(
                    baseline, task_dicts, uav_resources, cloud_resources
                )
                ...
'''


# ============ 完整示例: 修改后的 run_exp1 方法片段 ============

RUN_EXP1_MODIFIED = '''
    def run_exp1(self) -> Dict:
        """实验1: 小规模基线对比 - 输出完整32项指标（使用任务队列）"""
        print("\\n" + "=" * 70)
        print("实验1: 小规模基线对比 (完整32项指标)")
        print("=" * 70)

        # ... 场景配置代码不变 ...

        results = {}

        # 运行Proposed（带价格追踪）
        print("\\n运行 Proposed...")
        proposed_result, price_tracker = self._run_with_price_tracking(
            tasks, scenario, n_batches=25
        )

        # ... 离线最优计算代码不变 ...

        # 运行所有基线（包括论文算法）
        baselines = [
            "Proposed", "Edge-Only", "Cloud-Only", "Greedy",
            "Fixed-Split", "Random-Auction", "No-ActiveInference",
            "Heuristic-Alloc", "No-DynPricing", "B11-FixedPrice",
            "B11a-HighFixed", "B11b-LowFixed", "B12-DelayOpt",
            "MAPPO-Attention"    # 论文算法
        ]

        for baseline in baselines:
            if baseline == "Proposed":
                continue  # 已在上面运行

            print(f"运行 {baseline}...")

            # 检查是否是论文基线
            if baseline in self.paper_baselines:
                try:
                    result = self.paper_baselines[baseline].run(
                        task_dicts, uav_resources, cloud_resources
                    )
                    metrics = self._extract_full_metrics(result, offline_sw)
                    results[baseline] = ExperimentResult(baseline, scenario.name, metrics)
                    print(f"  SW={metrics.social_welfare:.2f}, Success={metrics.success_rate*100:.1f}%")
                except Exception as e:
                    print(f"  [Error] {e}")
                continue

            # 原有基线逻辑
            try:
                result = self.baseline_runner.run_single_baseline(
                    baseline, task_dicts, uav_resources, cloud_resources
                )
                metrics = self._extract_full_metrics(result, offline_sw)
                results[baseline] = ExperimentResult(baseline, scenario.name, metrics)
                print(f"  SW={metrics.social_welfare:.2f}, Success={metrics.success_rate*100:.1f}%")
            except Exception as e:
                print(f"  [Error] {e}")

        # ... 后续代码不变 ...
'''


# ============ 快速集成: 一键修改脚本 ============

def create_integration_patch():
    """
    创建集成补丁

    返回需要修改的代码片段
    """
    return {
        'import': IMPORT_CODE,
        'init': INIT_CODE,
        'baselines': EXP1_BASELINES,
        'run_logic': RUN_PAPER_BASELINE_CODE,
        'full_example': RUN_EXP1_MODIFIED
    }


# ============ 独立运行示例 ============

def run_paper_baselines_standalone():
    """
    独立运行论文基线的示例

    可以直接执行此函数测试论文算法
    """
    import numpy as np
    from experiments.scenario_config import create_small_scale_config
    from experiments.task_types import MNISTTaskGenerator, tasks_to_dict_list
    from experiments.paper_baselines import get_paper_baselines

    print("=" * 60)
    print("论文基线算法独立测试")
    print("=" * 60)

    # 创建场景
    scenario = create_small_scale_config(n_uavs=5, n_users=30)

    # 生成任务
    generator = MNISTTaskGenerator(
        area_size=scenario.area_size,
        latency_ratio=scenario.latency_ratio,
        tasks_per_user=scenario.tasks_per_user,
        seed=42
    )
    tasks = generator.generate_tasks(scenario.n_users)
    task_dicts = tasks_to_dict_list(tasks)

    # 获取资源
    uav_resources = scenario.get_uav_resources()
    cloud_resources = scenario.get_cloud_resources()

    print(f"\n场景: {scenario.name}")
    print(f"任务数: {len(tasks)}")
    print(f"UAV数: {len(uav_resources)}")

    # 运行论文基线
    paper_baselines = get_paper_baselines()

    results = {}
    for baseline in paper_baselines:
        print(f"\n运行 {baseline.name}...")
        result = baseline.run(task_dicts, uav_resources, cloud_resources)
        results[baseline.name] = result

        print(f"  成功率: {result.success_rate*100:.1f}%")
        print(f"  平均延迟: {result.avg_delay*1000:.1f}ms")
        print(f"  总能耗: {result.total_energy:.2f}J")
        print(f"  社会福利: {result.social_welfare:.2f}")
        print(f"  高优先级率: {result.high_priority_rate*100:.1f}%")

    # 打印对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"\n{'算法':<20} {'成功率':>10} {'延迟(ms)':>12} {'能耗(J)':>10} {'社会福利':>10}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<20} {result.success_rate*100:>9.1f}% "
              f"{result.avg_delay*1000:>10.1f}ms "
              f"{result.total_energy:>9.2f}J "
              f"{result.social_welfare:>10.2f}")

    return results


if __name__ == "__main__":
    run_paper_baselines_standalone()
