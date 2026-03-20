"""
快速验证 deadline 配置修改效果
"""
import sys
sys.path.insert(0, ".")

from experiments.task_types import (
    LATENCY_SENSITIVE_CONFIG_SMALL,
    COMPUTE_INTENSIVE_CONFIG_SMALL,
    LATENCY_SENSITIVE_CONFIG_LARGE,
    COMPUTE_INTENSIVE_CONFIG_LARGE,
    MNISTTaskGenerator,
    get_task_configs_for_scale
)

print("=" * 60)
print("Deadline 配置验证")
print("=" * 60)

print("\n【小规模实验配置】")
print(f"  高时延敏感任务: {LATENCY_SENSITIVE_CONFIG_SMALL.min_deadline}s - {LATENCY_SENSITIVE_CONFIG_SMALL.max_deadline}s")
print(f"  计算密集型任务: {COMPUTE_INTENSIVE_CONFIG_SMALL.min_deadline}s - {COMPUTE_INTENSIVE_CONFIG_SMALL.max_deadline}s")

print("\n【大规模实验配置】")
print(f"  时延敏感任务: {LATENCY_SENSITIVE_CONFIG_LARGE.min_deadline}s - {LATENCY_SENSITIVE_CONFIG_LARGE.max_deadline}s")
print(f"  计算密集型任务: {COMPUTE_INTENSIVE_CONFIG_LARGE.min_deadline}s - {COMPUTE_INTENSIVE_CONFIG_LARGE.max_deadline}s")

print("\n【任务生成测试】")
generator = MNISTTaskGenerator(area_size=200.0, latency_ratio=0.5, seed=42)
tasks = generator.generate_tasks(n_users=10)

latency_tasks = [t for t in tasks if t.task_type.value == "latency_sensitive"]
compute_tasks = [t for t in tasks if t.task_type.value == "compute_intensive"]

print(f"\n  延迟敏感任务 ({len(latency_tasks)}个):")
for t in latency_tasks[:3]:
    print(f"    Task {t.task_id}: {t.n_images} images, deadline={t.deadline:.2f}s, priority={t.priority:.2f}")

print(f"\n  计算密集型任务 ({len(compute_tasks)}个):")
for t in compute_tasks[:3]:
    print(f"    Task {t.task_id}: {t.n_images} images, deadline={t.deadline:.2f}s, priority={t.priority:.2f}")

print("\n" + "=" * 60)
print("验证完成 ✓")
print("=" * 60)
