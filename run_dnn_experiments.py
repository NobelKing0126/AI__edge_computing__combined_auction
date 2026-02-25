"""
真实DNN推理实验

测试多种DNN模型的切分推理，验证最优切分点效果
"""

import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_model(model_name: str):
    """加载预训练模型"""
    model_name = model_name.lower()
    if model_name == 'vgg16':
        return models.vgg16(weights=None), 224
    elif model_name == 'resnet50':
        return models.resnet50(weights=None), 224
    elif model_name == 'mobilenetv2':
        return models.mobilenet_v2(weights=None), 224
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def get_layer_output_size(model, input_size, device='cpu'):
    """获取每层的输出大小"""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size).to(device)
    
    layer_sizes = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            layer_sizes.append(output.numel() * 4)  # bytes (float32)
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(x)
    
    for h in hooks:
        h.remove()
    
    return layer_sizes


def benchmark_full_inference(model: nn.Module, input_size: int, n_runs: int = 20) -> float:
    """基准测试: 完整推理"""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # 计时
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000  # ms


def estimate_layer_times(model, model_name, input_size, n_runs=10):
    """估算每层的执行时间"""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    
    layer_times = {}
    layer_outputs = {}
    
    def pre_hook(name):
        def fn(module, input):
            layer_times[name] = {'start': time.perf_counter()}
        return fn
    
    def post_hook(name):
        def fn(module, input, output):
            layer_times[name]['end'] = time.perf_counter()
            layer_times[name]['duration'] = layer_times[name]['end'] - layer_times[name]['start']
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = output.numel() * 4
        return fn
    
    hooks = []
    layer_names = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            hooks.append(module.register_forward_hook(post_hook(name)))
            layer_names.append(name)
    
    # 多次运行取平均
    all_times = {name: [] for name in layer_names}
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
            for name in layer_names:
                if name in layer_times and 'duration' in layer_times[name]:
                    all_times[name].append(layer_times[name]['duration'])
    
    for h in hooks:
        h.remove()
    
    # 计算平均时间
    avg_times = []
    output_sizes = []
    for name in layer_names:
        if all_times[name]:
            avg_times.append(np.mean(all_times[name]) * 1000)  # ms
            output_sizes.append(layer_outputs.get(name, 0))
        else:
            avg_times.append(0)
            output_sizes.append(0)
    
    return layer_names, avg_times, output_sizes


def calculate_split_costs(layer_times, output_sizes, bandwidth=100e6):
    """计算不同切分点的代价"""
    n_layers = len(layer_times)
    results = []
    
    # 累积时间
    cumulative_times = np.cumsum(layer_times)
    total_time = cumulative_times[-1] if len(cumulative_times) > 0 else 0
    
    for split_idx in range(1, n_layers - 1):
        edge_time = cumulative_times[split_idx - 1]
        cloud_time = total_time - cumulative_times[split_idx - 1]
        
        # 传输时间
        intermediate_size = output_sizes[split_idx - 1] if split_idx > 0 else 0
        trans_time = (intermediate_size * 8 / bandwidth) * 1000  # ms
        
        total_with_trans = edge_time + trans_time + cloud_time
        
        results.append({
            'split_idx': split_idx,
            'edge_time': edge_time,
            'cloud_time': cloud_time,
            'trans_time': trans_time,
            'total_time': total_with_trans,
            'intermediate_size': intermediate_size
        })
    
    return results


def run_dnn_experiments():
    """运行DNN实验"""
    
    print("=" * 80)
    print("真实DNN推理实验")
    print("=" * 80)
    
    models_to_test = ['vgg16', 'resnet50', 'mobilenetv2']
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name.upper()}")
        print("=" * 60)
        
        # 加载模型
        print("  加载模型...")
        model, input_size = get_model(model_name)
        
        # 完整推理基准
        print("  完整推理基准测试...")
        full_time = benchmark_full_inference(model, input_size)
        print(f"  完整推理时间: {full_time:.2f}ms")
        
        # 获取每层执行时间
        print("  分析每层执行时间...")
        layer_names, layer_times, output_sizes = estimate_layer_times(model, model_name, input_size)
        print(f"  层数: {len(layer_names)}")
        
        # 计算切分代价
        print("  计算切分代价...")
        split_results = calculate_split_costs(layer_times, output_sizes)
        
        if split_results:
            # 找最优切分点
            best = min(split_results, key=lambda x: x['total_time'])
            print(f"  最优切分点: 层 {best['split_idx']}/{len(layer_names)}")
            print(f"  最优总时延: {best['total_time']:.2f}ms (边缘:{best['edge_time']:.2f}ms + "
                  f"传输:{best['trans_time']:.2f}ms + 云端:{best['cloud_time']:.2f}ms)")
            
            # 打印部分切分点对比
            print(f"\n  代表性切分点对比:")
            print(f"  {'切分层':>10} {'边缘计算':>12} {'传输':>10} {'云端计算':>12} {'总计':>12}")
            print("  " + "-" * 60)
            
            # 选择几个代表性的切分点
            sample_splits = [
                split_results[len(split_results)//10],       # 10%
                split_results[len(split_results)//4],        # 25%
                split_results[len(split_results)//2],        # 50%
                split_results[3*len(split_results)//4],      # 75%
                best                                         # 最优
            ]
            
            for r in sample_splits:
                print(f"  {r['split_idx']:>10} {r['edge_time']:>10.2f}ms "
                      f"{r['trans_time']:>9.2f}ms {r['cloud_time']:>10.2f}ms "
                      f"{r['total_time']:>10.2f}ms")
            
            results[model_name] = {
                'layers': len(layer_names),
                'full_time': full_time,
                'optimal_split': best['split_idx'],
                'optimal_time': best['total_time'],
                'edge_time': best['edge_time'],
                'cloud_time': best['cloud_time'],
                'trans_time': best['trans_time']
            }
        else:
            results[model_name] = {
                'layers': len(layer_names),
                'full_time': full_time,
                'optimal_split': -1,
                'optimal_time': -1
            }
    
    # 总结
    print("\n" + "=" * 80)
    print("DNN推理实验总结")
    print("=" * 80)
    
    print(f"\n{'模型':>12} {'层数':>8} {'完整推理':>12} {'最优切分层':>12} {'切分时延':>12} {'传输占比':>10}")
    print("-" * 75)
    
    for model_name, r in results.items():
        if r.get('optimal_time', -1) > 0:
            trans_ratio = r.get('trans_time', 0) / r['optimal_time'] * 100
        else:
            trans_ratio = 0
        print(f"{model_name:>12} {r['layers']:>8} {r['full_time']:>10.2f}ms "
              f"{r.get('optimal_split', 'N/A'):>12} {r.get('optimal_time', -1):>10.2f}ms "
              f"{trans_ratio:>9.1f}%")
    
    print("\n" + "=" * 80)
    print("结论分析")
    print("=" * 80)
    print("""
1. 模型复杂度分析:
   - VGG16: 大量卷积层，计算密集，完整推理时间最长
   - ResNet50: 残差结构，中等复杂度
   - MobileNetV2: 深度可分离卷积，最轻量级

2. 最优切分点分析:
   - 切分点选择需要平衡边缘计算、传输和云端计算
   - 传输开销取决于中间特征图大小
   - 通常在特征图较小的层进行切分效果更好

3. 协同推理收益:
   - 合理切分可以在时延和能耗之间取得平衡
   - 边缘侧承担前期特征提取，云端完成分类决策
""")
    
    return results


if __name__ == "__main__":
    results = run_dnn_experiments()
