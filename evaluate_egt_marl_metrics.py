import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== EGT-MARL 完整评估 ===")

try:
    import torch
    import numpy as np
    from environments.disaster_sim import DisasterSim
    from environments.config.constants import SimulationConfig, NUM_REGIONS
    from algorithms.egt_marl import EGTMARL
    
    # 配置
    env_config = {
        'scenario': 'earthquake_standard',
        'map_size': (1000, 1000),
        'num_agents': 20,
        'num_victims': 200,
        'num_resources': 10,
        'num_hospitals': 3,
        'num_regions': NUM_REGIONS,
        'disaster_type': 'earthquake',
        'severity': 'medium',
    }
    
    # 创建环境
    sim_config = SimulationConfig()
    sim_config.num_agents = env_config['num_agents']
    sim_config.num_victims = env_config['num_victims']
    sim_config.map_size = env_config['map_size']
    sim_config.num_regions = env_config['num_regions']
    sim_config.num_resources = env_config['num_resources']
    
    env = DisasterSim(
        scenario=env_config['scenario'],
        map_size=env_config['map_size'],
        num_agents=env_config['num_agents'],
        num_victims=env_config['num_victims'],
        num_resources=env_config['num_resources'],
        num_hospitals=env_config['num_hospitals'],
        disaster_type=env_config['disaster_type'],
        severity=env_config['severity'],
        config=sim_config
    )
    
    state_dim = env.get_state_dimension()
    num_agents = len(env.rescue_agents)
    
    print(f"状态维度: {state_dim}, 智能体数量: {num_agents}")
    
    # 创建算法
    algo_config_dict = {
        'marl': {
            'state_dim': state_dim,
            'action_dim': 32,
            'num_agents': num_agents,
            'hidden_dim': 128,
            'mixing_hidden_dim': 64,
            'attention_heads': 4,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'buffer_size': 10000
        },
        'egt': {'num_strategies': 3, 'learning_rate': 0.01},
        'anti_spoofing': {'observation_dim': state_dim, 'action_dim': 32},
        'dynamic_frontier': {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3}
    }
    
    algorithm = EGTMARL(env=env, config=algo_config_dict, hidden_dim=128)
    
    # 加载模型
    checkpoint_path = os.path.join('src', 'experiment_results', 'egt_marl_20260617_215838', 'models', 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'marl_layer_state' in checkpoint:
        algorithm.marl_layer.load_state_dict(checkpoint['marl_layer_state'], strict=False)
    if 'egt_layer_state' in checkpoint:
        algorithm.egt_layer.load_state_dict(checkpoint['egt_layer_state'])
    if 'anti_spoofing_state' in checkpoint:
        if hasattr(algorithm.anti_spoofing, 'load_state_dict'):
            algorithm.anti_spoofing.load_state_dict(checkpoint['anti_spoofing_state'])
    
    print("模型加载成功")
    
    # 评估
    print("\n=== 开始评估 (20 episodes) ===")
    num_episodes = 20
    results = {
        'rescue_rates': [],
        'response_times': [],
        'gini_coefficients': [],
        'resource_utilizations': []
    }
    
    for ep in range(num_episodes):
        sim_config_eval = SimulationConfig()
        sim_config_eval.num_agents = env_config['num_agents']
        sim_config_eval.num_victims = env_config['num_victims']
        sim_config_eval.map_size = env_config['map_size']
        sim_config_eval.num_regions = env_config['num_regions']
        sim_config_eval.num_resources = env_config['num_resources']
        
        eval_env = DisasterSim(
            scenario=env_config['scenario'],
            map_size=env_config['map_size'],
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            num_resources=env_config['num_resources'],
            num_hospitals=env_config['num_hospitals'],
            disaster_type=env_config['disaster_type'],
            severity=env_config['severity'],
            config=sim_config_eval
        )
        
        state, info = eval_env.reset()
        done = False
        step = 0
        max_steps = 500
        
        while not done and step < max_steps:
            actions = algorithm.select_action(state, training=False, epsilon=0.0)
            next_state, rewards, terminated, truncated, info = eval_env.step(actions)
            done = terminated or truncated
            state = next_state
            step += 1
        
        statistics = info.get('statistics', {})
        episode_rescued = statistics.get('total_rescued', 0)
        total_victims = len(eval_env.casualties)
        rescue_rate = (episode_rescued / total_victims * 100)
        
        response_times = statistics.get('response_times', [])
        avg_response_time = np.mean(response_times) if response_times else 0.0
        gini_coefficient = statistics.get('gini_coefficient', 0.0)
        resource_utilization = statistics.get('resource_utilization', 0.0)
        
        results['rescue_rates'].append(rescue_rate)
        results['response_times'].append(avg_response_time)
        results['gini_coefficients'].append(gini_coefficient)
        results['resource_utilizations'].append(resource_utilization)
        
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{num_episodes} - 救援率: {rescue_rate:.1f}%, 基尼: {gini_coefficient:.3f}")
    
    # 输出结果
    print("\n=== EGT-MARL 评估结果 ===")
    print(f"救援率: {np.mean(results['rescue_rates']):.1f}% ± {np.std(results['rescue_rates']):.1f}%")
    print(f"响应时间: {np.mean(results['response_times']):.1f}s ± {np.std(results['response_times']):.1f}s")
    print(f"基尼系数: {np.mean(results['gini_coefficients']):.4f} ± {np.std(results['gini_coefficients']):.4f}")
    print(f"资源利用率: {np.mean(results['resource_utilizations']):.1f}%")
    
    # 保存结果
    with open('egt_marl_evaluation_results.txt', 'w') as f:
        f.write(f"EGT-MARL Evaluation Results (20 episodes)\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Rescue Rate: {np.mean(results['rescue_rates']):.1f}% ± {np.std(results['rescue_rates']):.1f}%\n")
        f.write(f"Response Time: {np.mean(results['response_times']):.1f}s ± {np.std(results['response_times']):.1f}s\n")
        f.write(f"Gini Coefficient: {np.mean(results['gini_coefficients']):.4f} ± {np.std(results['gini_coefficients']):.4f}\n")
        f.write(f"Resource Utilization: {np.mean(results['resource_utilizations']):.1f}%\n")
    
    print("\n结果已保存到 egt_marl_evaluation_results.txt")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
