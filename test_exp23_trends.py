"""
测试实验2和3的趋势（小规模实验）
验证用户扩展和UAV扩展的正确趋势
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_real_experiments_v9 import RealExperimentRunnerV9
from experiments.scenario_config import (
    ScenarioConfig, ScenarioType,
    create_small_scale_config, create_large_scale_config,
    _SMALL_SCALE_CLOUD, _LARGE_SCALE_CLOUD
)
from experiments.task_types import (
    MNISTTaskGenerator, tasks_to_dict_list,
    get_task_configs_for_scale
)
import numpy as np


def test_scale_specific_params():
    """测试规模特定参数是否正确配置"""
    print("=" * 60)
    print("测试规模特定参数配置")
    print("=" * 60)

    # 小规模配置
    small_config = create_small_scale_config(5, 30)
    print(f"\n小规模场景:")
    print(f"  区域: {small_config.area_size}m")
    print(f"  云端算力: {small_config.cloud_config.compute_capacity/1e9:.0f} GFLOPS")
    print(f"  单任务最大: {small_config.cloud_config.F_per_task_max/1e9:.0f} GFLOPS")
    print(f"  传播延迟: {small_config.cloud_config.T_propagation*1000:.0f} ms")
    print(f"  最大并发: {small_config.cloud_config.max_concurrent_tasks}")

    cloud_res = small_config.get_cloud_resources()
    print(f"  云端资源: F_c={cloud_res['f_cloud']/1e9:.0f}G, "
          f"F_per_task={cloud_res['F_per_task_max']/1e9:.0f}G, "
          f"T_prop={cloud_res['T_propagation']*1000:.0f}ms, "
          f"max_concurrent={cloud_res['max_concurrent_tasks']}")

    # 大规模配置
    large_config = create_large_scale_config(15, 100)
    print(f"\n大规模场景:")
    print(f"  区域: {large_config.area_size}m")
    print(f"  云端算力: {large_config.cloud_config.compute_capacity/1e9:.0f} GFLOPS")
    print(f"  单任务最大: {large_config.cloud_config.F_per_task_max/1e9:.0f} GFLOPS")
    print(f"  传播延迟: {large_config.cloud_config.T_propagation*1000:.0f} ms")
    print(f"  最大并发: {large_config.cloud_config.max_concurrent_tasks}")

    cloud_res = large_config.get_cloud_resources()
    print(f"  云端资源: F_c={cloud_res['f_cloud']/1e9:.0f}G, "
          f"F_per_task={cloud_res['F_per_task_max']/1e9:.0f}G, "
          f"T_prop={cloud_res['T_propagation']*1000:.0f}ms, "
          f"max_concurrent={cloud_res['max_concurrent_tasks']}")

    # 测试任务配置
    print("\n任务配置:")
    small_latency, small_compute = get_task_configs_for_scale(True)
    large_latency, large_compute = get_task_configs_for_scale(False)

    print(f"  小规模延迟敏感: {small_latency.min_images}-{small_latency.max_images}张, "
          f"deadline {small_latency.min_deadline}-{small_latency.max_deadline}s")
    print(f"  小规模计算密集: {small_compute.min_images}-{small_compute.max_images}张, "
          f"deadline {small_compute.min_deadline}-{small_compute.max_deadline}s")
    print(f"  大规模延迟敏感: {large_latency.min_images}-{large_latency.max_images}张, "
          f"deadline {large_latency.min_deadline}-{large_latency.max_deadline}s")
    print(f"  大规模计算密集: {large_compute.min_images}-{large_compute.max_images}张, "
          f"deadline {large_compute.min_deadline}-{large_compute.max_deadline}s")


def run_exp2_quick():
    """快速运行实验2：小规模用户扩展"""
    print("\n" + "=" * 70)
    print("实验2: 小规模用户扩展 (5 UAV固定)")
    print("=" * 70)

    runner = RealExperimentRunnerV9(seed=42)
    user_counts = [10, 20, 30, 40, 50]
    results = []

    for n_users in user_counts:
        scenario = create_small_scale_config(5, n_users)
        task_gen = runner._create_task_generator(scenario)
        # 使用不同的种子，基于用户数量
        tasks = task_gen.generate_tasks(n_users, seed=42 + n_users)
        task_dicts = tasks_to_dict_list(tasks)

        uav_resources = scenario.get_uav_resources()
        cloud_resources = scenario.get_cloud_resources()

        runner.proposed._reset_tracking(5)
        result = runner.proposed.run(task_dicts, uav_resources, cloud_resources)

        results.append({
            'n_users': n_users,
            'success_rate': result.success_rate,
            'social_welfare': result.social_welfare
        })

        print(f"  用户数={n_users}: 成功率={result.success_rate*100:.1f}%, SW={result.social_welfare:.2f}")

    # 验证趋势
    rates = [r['success_rate'] for r in results]
    is_decreasing = all(rates[i] >= rates[i+1] - 0.05 for i in range(len(rates)-1))
    total_drop = (rates[0] - rates[-1]) * 100

    print(f"\n成功率序列: {[f'{r*100:.1f}%' for r in rates]}")
    print(f"是否单调下降: {'是' if is_decreasing else '否'}")
    print(f"总下降幅度: {total_drop:.1f}%")
    print(f"目标: 下降≥20%")

    return results


def run_exp3_quick():
    """快速运行实验3：小规模UAV扩展"""
    print("\n" + "=" * 70)
    print("实验3: 小规模UAV扩展 (50用户固定)")
    print("=" * 70)

    runner = RealExperimentRunnerV9(seed=42)
    uav_counts = [3, 4, 5, 6, 7, 8]
    results = []

    for n_uavs in uav_counts:
        scenario = create_small_scale_config(n_uavs, 50)
        task_gen = runner._create_task_generator(scenario)
        # 使用不同的种子，基于UAV数量
        tasks = task_gen.generate_tasks(50, seed=42 + n_uavs * 100)
        task_dicts = tasks_to_dict_list(tasks)

        uav_resources = scenario.get_uav_resources()
        cloud_resources = scenario.get_cloud_resources()

        runner.proposed._reset_tracking(n_uavs)
        result = runner.proposed.run(task_dicts, uav_resources, cloud_resources)

        results.append({
            'n_uavs': n_uavs,
            'success_rate': result.success_rate,
            'social_welfare': result.social_welfare
        })

        print(f"  UAV数={n_uavs}: 成功率={result.success_rate*100:.1f}%, SW={result.social_welfare:.2f}")

    # 验证趋势
    rates = [r['success_rate'] for r in results]
    is_increasing = all(rates[i] <= rates[i+1] + 0.05 for i in range(len(rates)-1))
    total_rise = (rates[-1] - rates[0]) * 100

    print(f"\n成功率序列: {[f'{r*100:.1f}%' for r in rates]}")
    print(f"是否单调上升: {'是' if is_increasing else '否'}")
    print(f"总上升幅度: {total_rise:.1f}%")
    print(f"目标: 上升≥25%")

    return results


if __name__ == "__main__":
    # 测试参数配置
    test_scale_specific_params()

    # 运行实验2
    exp2_results = run_exp2_quick()

    # 运行实验3
    exp3_results = run_exp3_quick()

    print("\n" + "=" * 70)
    print("快速测试完成")
    print("=" * 70)
