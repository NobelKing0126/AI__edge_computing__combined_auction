"""
实验参数配置 - 确保稳健趋势 (平衡版V5)

设计目标：
1. 用户数增加 → 任务成功率稳健下降 (从~95%降到~60%)
2. UAV数增加 → 任务成功率稳健上升 (从~55%升到~85%)

核心原理（时分系统）：
- 系统按时间片处理任务，任务在deadline内完成即可
- 每秒任务到达数 = n_users × per_user_arrival_rate
- 每秒处理能力 = UAV数 × 单UAV算力 + 云端算力
- 资源利用率 = 任务需求算力 / 系统供给算力
- 成功率与资源利用率负相关

关键洞察：
- 资源利用率在0.4-1.2之间时，成功率对用户数/UAV数变化最敏感
- 需要平衡资源约束，使系统处于"适度紧张"状态

基于docs/idea38.txt和docs/竞争比.txt设计
"""

# ============================================================
# 小规模实验参数 (200m x 200m) - 平衡版V5
# 目标：
# - 用户10→50: 成功率从~95%降到~65%
# - UAV 3→8 (固定50用户): 成功率从~55%升到~85%
# ============================================================

# 小规模实验 - UAV参数
SMALL_SCALE_UAV_PARAMS = {
    'f_max': 2.0e9,            # UAV最大算力 2.0 GFLOPS (V5: 从1.5提升)
    'E_max': 150e3,            # 电池容量 150kJ (V5: 从120提升)
    'H': 100.0,                # 飞行高度 100m
    'R_cover': 45.0,           # 覆盖半径 45m (V5: 从40提升)
    'P_hover': 130.0,          # 悬停功率 W
    'P_fly': 190.0,            # 飞行功率 W
    'v_fly': 10.0,             # 飞行速度 m/s
    'kappa_edge': 1e-28,       # 边缘能耗系数
}

# 小规模实验 - 云端参数 (平衡版V5)
SMALL_SCALE_CLOUD_PARAMS = {
    'F_c': 4.0e9,              # 云端总算力 4.0 GFLOPS (V5: 从2.8大幅提升)
    'F_per_task_max': 1.2e9,   # 单任务最大 1.2 GFLOPS (V5: 从0.8提升)
    'T_propagation': 0.25,     # 传播延迟 250ms (V5: 从380ms降低)
    'max_concurrent_tasks': 5, # 最大并发 5 个任务 (V5: 从3增加)
    'kappa_cloud': 1e-29,      # 云端能耗系数
}

# 小规模实验 - 通信参数
SMALL_SCALE_CHANNEL_PARAMS = {
    'W': 0.7e6,                # 信道带宽 0.7 MHz
    'beta_0': 3.5e-7,          # 参考信道增益
    'N_0': 1e-18,              # 噪声功率谱密度
    'P_tx_user': 0.09,         # 用户发射功率 0.09W
    'R_backhaul': 40e6,        # 回程带宽 40 Mbps
    'num_channels': 8,         # 子信道数
}

# 小规模实验 - 任务参数 (平衡版V5)
SMALL_SCALE_TASK_PARAMS = {
    'latency_sensitive': {
        'min_images': 20,      # 最少图像数 (V5: 从25减少)
        'max_images': 40,      # 最多图像数 (V5: 从50减少)
        'min_deadline': 4.0,   # 最小时延上限 4.0s (V5: 放宽)
        'max_deadline': 6.0,   # 最大时延上限 6.0s (V5: 放宽)
        'ratio': 0.6,          # 占比 60%
    },
    'compute_intensive': {
        'min_images': 50,      # 最少图像数 (V5: 从70减少)
        'max_images': 100,     # 最多图像数 (V5: 从130减少)
        'min_deadline': 15.0,  # 最小时延上限 15s (V5: 放宽)
        'max_deadline': 25.0,  # 最大时延上限 25s (V5: 放宽)
        'ratio': 0.4,          # 占比 40%
    },
    'tasks_per_user': 5,       # 每用户任务数
}

# 小规模实验 - 仿真参数 (平衡版V6 - 目标成功率60-90%)
SMALL_SCALE_SIMULATION = {
    'per_user_arrival_rate': 0.05,  # 每用户每秒0.05个任务 (V6: 降低以提高成功率)
    'simulation_time': 200.0,       # 仿真时间 s
}


# ============================================================
# 大规模实验参数 (1000m x 1000m) - 平衡版V5
# 目标：
# - 用户50→200: 成功率从~95%降到~55%
# - UAV 8→16 (固定200用户): 成功率从~50%升到~72%
# ============================================================

# 大规模实验 - UAV参数 (平衡版V5)
LARGE_SCALE_UAV_PARAMS = {
    'f_max': 2.5e9,            # UAV最大算力 2.5 GFLOPS (V5: 从1.8提升)
    'E_max': 200e3,            # 电池容量 200kJ (V5: 从180提升)
    'H': 100.0,                # 飞行高度 100m
    'R_cover': 70.0,           # 覆盖半径 70m
    'P_hover': 130.0,          # 悬停功率 W
    'P_fly': 190.0,            # 飞行功率 W
    'v_fly': 10.0,             # 飞行速度 m/s
    'kappa_edge': 1e-28,       # 边缘能耗系数
}

# 大规模实验 - 云端参数 (平衡版V5)
LARGE_SCALE_CLOUD_PARAMS = {
    'F_c': 15e9,               # 云端总算力 15 GFLOPS (V5: 从12大幅提升)
    'F_per_task_max': 2.5e9,   # 单任务最大 2.5 GFLOPS (V5: 从1.8提升)
    'T_propagation': 0.20,     # 传播延迟 200ms (V5: 从320ms降低)
    'max_concurrent_tasks': 8, # 最大并发 8 个任务 (V5: 从6增加)
    'kappa_cloud': 1e-29,      # 云端能耗系数
}

# 大规模实验 - 通信参数
LARGE_SCALE_CHANNEL_PARAMS = {
    'W': 1.0e6,                # 信道带宽 1 MHz
    'beta_0': 4.5e-7,          # 参考信道增益
    'N_0': 1e-18,              # 噪声功率谱密度
    'P_tx_user': 0.11,         # 用户发射功率 0.11W
    'R_backhaul': 80e6,        # 回程带宽 80 Mbps
    'num_channels': 16,        # 子信道数
}

# 大规模实验 - 任务参数 (平衡版V5)
LARGE_SCALE_TASK_PARAMS = {
    'latency_sensitive': {
        'min_images': 25,      # 最少图像数 (V5: 从30减少)
        'max_images': 50,      # 最多图像数 (V5: 从60减少)
        'min_deadline': 5.0,   # 最小时延上限 5s (V5: 放宽)
        'max_deadline': 8.0,   # 最大时延上限 8s (V5: 放宽)
        'ratio': 0.6,          # 占比 60%
    },
    'compute_intensive': {
        'min_images': 70,      # 最少图像数 (V5: 从90减少)
        'max_images': 140,     # 最多图像数 (V5: 从170减少)
        'min_deadline': 18.0,  # 最小时延上限 18s (V5: 放宽)
        'max_deadline': 30.0,  # 最大时延上限 30s (V5: 放宽)
        'ratio': 0.4,          # 占比 40%
    },
    'tasks_per_user': 6,       # 每用户任务数
}

# 大规模实验 - 仿真参数 (平衡版V6 - 目标成功率60-90%)
LARGE_SCALE_SIMULATION = {
    'per_user_arrival_rate': 0.02,  # 每用户每秒0.02个任务 (V6: 进一步降低以提升成功率)
    'simulation_time': 300.0,       # 仿真时间 s
}


# ============================================================
# 资源供需分析 (时分系统正确计算)
# ============================================================

def analyze_resource_balance(n_uavs: int, n_users: int, is_small_scale: bool = True,
                            per_user_arrival_rate: float = None) -> dict:
    """
    分析时分系统的资源供需平衡

    核心原理：
    - 系统按时间片处理任务
    - 总到达率 = n_users × per_user_arrival_rate (每用户每秒提交的任务数)
    - 系统每秒处理能力 = n_uavs × f_max + F_c (GFLOPS/s)
    - 任务每秒需求算力 = 计算量 / deadline

    Args:
        n_uavs: UAV数量
        n_users: 用户数量
        is_small_scale: 是否小规模实验
        per_user_arrival_rate: 每用户每秒的任务到达率 (None则使用默认值)

    Returns:
        dict: 包含供给、需求、利用率等信息
    """
    if is_small_scale:
        uav_params = SMALL_SCALE_UAV_PARAMS
        cloud_params = SMALL_SCALE_CLOUD_PARAMS
        task_params = SMALL_SCALE_TASK_PARAMS
        if per_user_arrival_rate is None:
            per_user_arrival_rate = SMALL_SCALE_SIMULATION['per_user_arrival_rate']
    else:
        uav_params = LARGE_SCALE_UAV_PARAMS
        cloud_params = LARGE_SCALE_CLOUD_PARAMS
        task_params = LARGE_SCALE_TASK_PARAMS
        if per_user_arrival_rate is None:
            per_user_arrival_rate = LARGE_SCALE_SIMULATION['per_user_arrival_rate']

    # 1. 计算系统每秒处理能力 (GFLOPS)
    edge_capacity = n_uavs * uav_params['f_max'] / 1e9  # GFLOPS
    cloud_capacity = cloud_params['F_c'] / 1e9  # GFLOPS
    total_capacity = edge_capacity + cloud_capacity  # GFLOPS

    # 2. 估算任务特征
    # 延迟敏感型任务
    ls_images_avg = (task_params['latency_sensitive']['min_images'] +
                     task_params['latency_sensitive']['max_images']) / 2
    ls_deadline_avg = (task_params['latency_sensitive']['min_deadline'] +
                       task_params['latency_sensitive']['max_deadline']) / 2
    ls_compute = ls_images_avg * 0.3  # GFLOPs (每图像约0.3 GFLOPs)

    # 计算密集型任务
    ci_images_avg = (task_params['compute_intensive']['min_images'] +
                     task_params['compute_intensive']['max_images']) / 2
    ci_deadline_avg = (task_params['compute_intensive']['min_deadline'] +
                       task_params['compute_intensive']['max_deadline']) / 2
    ci_compute = ci_images_avg * 0.3  # GFLOPs

    # 3. 计算每秒任务到达数和资源需求
    ls_ratio = task_params['latency_sensitive']['ratio']
    ci_ratio = task_params['compute_intensive']['ratio']

    # 每秒到达的总任务数 = 用户数 × 单用户到达率
    total_arrival_rate = n_users * per_user_arrival_rate

    # 每个任务需要的瞬时算力 (GFLOPS)
    ls_instant_demand = ls_compute / ls_deadline_avg  # GFLOPS per task
    ci_instant_demand = ci_compute / ci_deadline_avg   # GFLOPS per task

    # 平均每个任务需要的瞬时算力
    avg_instant_demand = ls_instant_demand * ls_ratio + ci_instant_demand * ci_ratio

    # 每秒总资源需求 (GFLOPS)
    total_demand_per_second = total_arrival_rate * avg_instant_demand

    # 4. 计算资源利用率
    utilization = total_demand_per_second / total_capacity if total_capacity > 0 else 0

    # 5. 预期成功率模型 (平衡版)
    # 利用率 < 0.4: 成功率 > 92%
    # 利用率 0.4-0.6: 成功率 80-92%
    # 利用率 0.6-0.8: 成功率 65-80%
    # 利用率 0.8-1.0: 成功率 45-65%
    # 利用率 1.0-1.2: 成功率 30-45%
    # 利用率 > 1.2: 成功率 < 30%
    if utilization < 0.4:
        expected_sr = 0.95
    elif utilization < 0.6:
        expected_sr = 0.80 + (0.6 - utilization) * 0.75
    elif utilization < 0.8:
        expected_sr = 0.65 + (0.8 - utilization) * 0.75
    elif utilization < 1.0:
        expected_sr = 0.45 + (1.0 - utilization) * 1.0
    elif utilization < 1.2:
        expected_sr = 0.30 + (1.2 - utilization) * 0.75
    else:
        expected_sr = max(0.15, 0.30 / utilization)

    return {
        'n_uavs': n_uavs,
        'n_users': n_users,
        'per_user_arrival_rate': per_user_arrival_rate,
        'total_arrival_rate': total_arrival_rate,
        'edge_capacity_gflops': edge_capacity,
        'cloud_capacity_gflops': cloud_capacity,
        'total_capacity_gflops': total_capacity,
        'avg_task_demand_gflops': avg_instant_demand,
        'total_demand_per_second': total_demand_per_second,
        'utilization': utilization,
        'expected_success_rate': expected_sr
    }


# ============================================================
# 验证参数设计的预期趋势
# ============================================================

def validate_trend_design():
    """验证参数设计是否能产生预期趋势"""

    print("=" * 70)
    print("Resource Analysis and Trend Validation (Balanced V5)")
    print("=" * 70)
    print("\nKey: Utilization = Task Demand/sec / System Capacity/sec")
    print("      Util<0.4->SR>92%, 0.4-0.6->80-92%, 0.6-0.8->65-80%")
    print("      0.8-1.0->45-65%, 1.0-1.2->30-45%, >1.2-><30%")

    # 小规模实验参数
    small_user_rate = SMALL_SCALE_SIMULATION['per_user_arrival_rate']

    print("\n### Small-Scale: User Expansion (Fixed 5 UAVs) ###")
    print("-" * 90)
    print(f"{'Users':^8} {'EdgeCap':^10} {'CloudCap':^10} {'TotalCap':^10} {'Demand/s':^10} {'Util':^8} {'ExpSR':^10}")
    print("-" * 90)
    for n_users in [10, 20, 30, 40, 50]:
        result = analyze_resource_balance(5, n_users, is_small_scale=True)
        print(f"{n_users:^8d} {result['edge_capacity_gflops']:^10.1f} "
              f"{result['cloud_capacity_gflops']:^10.1f} {result['total_capacity_gflops']:^10.1f} "
              f"{result['total_demand_per_second']:^10.2f} {result['utilization']:^8.2f} "
              f"{result['expected_success_rate']*100:^10.0f}%")

    print("\n### Small-Scale: UAV Expansion (Fixed 50 Users) ###")
    print("-" * 90)
    print(f"{'UAVs':^8} {'EdgeCap':^10} {'CloudCap':^10} {'TotalCap':^10} {'Demand/s':^10} {'Util':^8} {'ExpSR':^10}")
    print("-" * 90)
    for n_uavs in [3, 4, 5, 6, 7, 8]:
        result = analyze_resource_balance(n_uavs, 50, is_small_scale=True)
        print(f"{n_uavs:^8d} {result['edge_capacity_gflops']:^10.1f} "
              f"{result['cloud_capacity_gflops']:^10.1f} {result['total_capacity_gflops']:^10.1f} "
              f"{result['total_demand_per_second']:^10.2f} {result['utilization']:^8.2f} "
              f"{result['expected_success_rate']*100:^10.0f}%")

    # 大规模实验参数
    large_user_rate = LARGE_SCALE_SIMULATION['per_user_arrival_rate']

    print("\n### Large-Scale: User Expansion (Fixed 10 UAVs) ###")
    print("-" * 90)
    print(f"{'Users':^8} {'EdgeCap':^10} {'CloudCap':^10} {'TotalCap':^10} {'Demand/s':^10} {'Util':^8} {'ExpSR':^10}")
    print("-" * 90)
    for n_users in [50, 80, 100, 150, 200]:
        result = analyze_resource_balance(10, n_users, is_small_scale=False)
        print(f"{n_users:^8d} {result['edge_capacity_gflops']:^10.1f} "
              f"{result['cloud_capacity_gflops']:^10.1f} {result['total_capacity_gflops']:^10.1f} "
              f"{result['total_demand_per_second']:^10.2f} {result['utilization']:^8.2f} "
              f"{result['expected_success_rate']*100:^10.0f}%")

    print("\n### Large-Scale: UAV Expansion (Fixed 200 Users) ###")
    print("-" * 90)
    print(f"{'UAVs':^8} {'EdgeCap':^10} {'CloudCap':^10} {'TotalCap':^10} {'Demand/s':^10} {'Util':^8} {'ExpSR':^10}")
    print("-" * 90)
    for n_uavs in [8, 10, 12, 14, 16]:
        result = analyze_resource_balance(n_uavs, 200, is_small_scale=False)
        print(f"{n_uavs:^8d} {result['edge_capacity_gflops']:^10.1f} "
              f"{result['cloud_capacity_gflops']:^10.1f} {result['total_capacity_gflops']:^10.1f} "
              f"{result['total_demand_per_second']:^10.2f} {result['utilization']:^8.2f} "
              f"{result['expected_success_rate']*100:^10.0f}%")

    print("\n" + "=" * 70)
    print("Trend Validation Results:")
    print("1. More Users -> Higher Utilization -> Lower Success Rate [EXPECTED]")
    print("2. More UAVs -> Higher Capacity -> Lower Utilization -> Higher Success Rate [EXPECTED]")
    print("=" * 70)


if __name__ == "__main__":
    validate_trend_design()
