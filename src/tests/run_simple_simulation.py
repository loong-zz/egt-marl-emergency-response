from environments.disaster_sim import DisasterSim
from algorithms.egt_marl import EGTMARL

# 创建环境（可选场景：earthquake_standard, flood_standard, hurricane_standard）
env = DisasterSim(scenario="earthquake_standard")

# 创建算法实例
algorithm = EGTMARL(env)

# 运行单个模拟 episode
print("开始模拟...")
results = algorithm.run_episode()

# 查看结果
# 从 info 中获取统计数据
info = results.get('info', {})
statistics = info.get('statistics', {})

total_survivors = statistics.get('total_survivors', 0)
gini_coefficient = statistics.get('fairness_metrics', {}).get('gini', [0])[-1] if statistics.get('fairness_metrics', {}).get('gini', []) else 0
mean_response_time = statistics.get('response_times', [0]) and sum(statistics.get('response_times', [])) / len(statistics.get('response_times', [])) or 0
resource_utilization = sum(statistics.get('resource_utilization', {}).values()) / len(statistics.get('resource_utilization', {})) if statistics.get('resource_utilization', {}) else 0

output = f"总幸存者数: {total_survivors}"
output += f"\n基尼系数（公平性）: {gini_coefficient:.4f}"
output += f"\n平均响应时间: {mean_response_time:.2f}"
output += f"\n资源利用率: {resource_utilization:.4f}"

# 打印更多详细信息
print("\n详细统计信息:")
print(f"总受害者数: {info.get('num_casualties', 0)}")
print(f"救援 agent 数量: {info.get('num_rescue_agents', 0)}")
print(f"受灾区域数量: {info.get('num_affected_areas', 0)}")
print(f"资源库数量: {info.get('num_resource_depots', 0)}")
print(f"当前时间: {info.get('current_time', 0):.2f} 秒")
print(f"步数: {info.get('step_count', 0)}")

print(output)

# 写入文件以便查看
with open('simulation_results.txt', 'w') as f:
    f.write(output)

print("\nResults written to simulation_results.txt")
