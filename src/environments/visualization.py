"""
DisasterSim-2026 环境可视化工具

提供灾害场景、智能体行为和实时性能监控的可视化功能。
支持多种可视化模式：2D地图、3D场景、实时监控仪表盘等。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 抑制警告
warnings.filterwarnings('ignore')


class DisasterVisualizer:
    """灾害环境可视化器"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            env_config: 环境配置字典
        """
        self.env_config = env_config
        self.fig = None
        self.ax = None
        self.animation = None
        
        # 颜色映射
        self.colors = {
            'earthquake': '#FF6B6B',  # 地震 - 红色
            'flood': '#4ECDC4',       # 洪水 - 青色
            'hurricane': '#45B7D1',    # 飓风 - 蓝色
            'industrial': '#96CEB4',   # 工业事故 - 绿色
            'pandemic': '#FFEAA7',     # 疫情 - 黄色
            'compound': '#DDA0DD',     # 复合灾害 - 紫色
            
            'drone': '#1F77B4',        # 无人机 - 蓝色
            'ambulance': '#FF7F0E',    # 救护车 - 橙色
            'mobile_hospital': '#2CA02C',  # 移动医院 - 绿色
            
            'low': '#98DF8A',          # 低风险 - 浅绿
            'medium': '#FFBB78',       # 中风险 - 橙色
            'high': '#FF9896',         # 高风险 - 红色
            'critical': '#C5B0D5',     # 危急 - 紫色
            
            'resource': '#17BECF',     # 资源点 - 青色
            'victim': '#E377C2',       # 受害者 - 粉色
            'hospital': '#7F7F7F',     # 医院 - 灰色
        }
        
        # 样式配置
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_scenario_map(self, scenario_data: Dict[str, Any], 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制灾害场景地图
        
        Args:
            scenario_data: 场景数据
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取场景信息
        disaster_type = scenario_data.get('disaster_type', 'earthquake')
        severity = scenario_data.get('severity', 'medium')
        map_size = scenario_data.get('map_size', (100, 100))
        
        # 绘制地图背景
        ax.set_xlim(0, map_size[0])
        ax.set_ylim(0, map_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制灾害中心
        center = scenario_data.get('epicenter', (map_size[0]/2, map_size[1]/2))
        radius = scenario_data.get('radius', min(map_size)/4)
        
        # 根据灾害类型选择颜色
        disaster_color = self.colors.get(disaster_type, '#FF6B6B')
        
        # 绘制灾害影响区域（渐变）
        for i in range(5, 0, -1):
            alpha = 0.1 + (i * 0.05)
            r = radius * (i / 5)
            circle = patches.Circle(center, r, 
                                   color=disaster_color, 
                                   alpha=alpha,
                                   linewidth=0)
            ax.add_patch(circle)
        
        # 绘制灾害中心点
        ax.scatter(center[0], center[1], 
                  color=disaster_color, 
                  s=200, marker='*', 
                  edgecolors='black', linewidth=2,
                  label=f'{disaster_type.capitalize()} Epicenter')
        
        # 绘制资源点
        resources = scenario_data.get('resources', [])
        for i, resource in enumerate(resources):
            pos = resource.get('position', (np.random.rand()*map_size[0], 
                                           np.random.rand()*map_size[1]))
            capacity = resource.get('capacity', 100)
            
            ax.scatter(pos[0], pos[1], 
                      color=self.colors['resource'],
                      s=capacity,  # 大小表示容量
                      alpha=0.7,
                      marker='s',
                      edgecolors='black', linewidth=1,
                      label='Resource Point' if i == 0 else None)
            
            # 添加容量标签
            ax.annotate(f'{capacity}', 
                       xy=pos, 
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8)
        
        # 绘制受害者分布
        victims = scenario_data.get('victims', [])
        for i, victim in enumerate(victims):
            pos = victim.get('position', (np.random.rand()*map_size[0], 
                                         np.random.rand()*map_size[1]))
            severity_level = victim.get('severity', 'medium')
            
            ax.scatter(pos[0], pos[1], 
                      color=self.colors[severity_level],
                      s=50,
                      marker='o',
                      edgecolors='black', linewidth=0.5,
                      label=f'{severity_level.capitalize()} Victim' 
                      if i == 0 and severity_level == 'medium' else None)
        
        # 绘制医院位置
        hospitals = scenario_data.get('hospitals', [])
        for i, hospital in enumerate(hospitals):
            pos = hospital.get('position', (np.random.rand()*map_size[0], 
                                           np.random.rand()*map_size[1]))
            capacity = hospital.get('capacity', 50)
            
            ax.scatter(pos[0], pos[1], 
                      color=self.colors['hospital'],
                      s=capacity * 2,  # 大小表示容量
                      marker='H',
                      edgecolors='black', linewidth=2,
                      label='Hospital' if i == 0 else None)
            
            # 添加医院标签
            ax.annotate(f'H{i+1}', 
                       xy=pos, 
                       xytext=(0, 10),
                       textcoords='offset points',
                       fontsize=10, weight='bold')
        
        # 设置标题和标签
        ax.set_title(f'{disaster_type.capitalize()} Disaster Scenario - {severity.capitalize()} Severity', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10)
        
        # 添加信息框
        info_text = f"""
        Disaster Type: {disaster_type}
        Severity: {severity}
        Map Size: {map_size[0]}x{map_size[1]} m
        Resources: {len(resources)}
        Victims: {len(victims)}
        Hospitals: {len(hospitals)}
        """
        
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scenario map saved to {save_path}")
        
        return fig
    
    def plot_agent_trajectories(self, agent_data: Dict[str, List[Tuple[float, float]]],
                               scenario_data: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制智能体轨迹
        
        Args:
            agent_data: 智能体轨迹数据 {agent_id: [(x1,y1), (x2,y2), ...]}
            scenario_data: 场景数据
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制基础场景
        self._plot_base_scenario(ax, scenario_data)
        
        # 绘制智能体轨迹
        agent_types = ['drone', 'ambulance', 'mobile_hospital']
        
        for agent_id, trajectory in agent_data.items():
            if not trajectory:
                continue
                
            # 解析智能体类型
            agent_type = 'drone'  # 默认
            for atype in agent_types:
                if atype in agent_id.lower():
                    agent_type = atype
                    break
            
            # 转换为numpy数组
            traj_array = np.array(trajectory)
            
            # 绘制轨迹线
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                   color=self.colors[agent_type],
                   alpha=0.6,
                   linewidth=2,
                   label=f'{agent_type.capitalize()} Trajectory' 
                   if agent_id == list(agent_data.keys())[0] else None)
            
            # 绘制起点和终点
            ax.scatter(traj_array[0, 0], traj_array[0, 1],
                      color=self.colors[agent_type],
                      s=100, marker='o',
                      edgecolors='black', linewidth=2,
                      label=f'{agent_type.capitalize()} Start' 
                      if agent_id == list(agent_data.keys())[0] else None)
            
            ax.scatter(traj_array[-1, 0], traj_array[-1, 1],
                      color=self.colors[agent_type],
                      s=100, marker='s',
                      edgecolors='black', linewidth=2,
                      label=f'{agent_type.capitalize()} End' 
                      if agent_id == list(agent_data.keys())[0] else None)
            
            # 添加智能体ID标签
            mid_idx = len(traj_array) // 2
            if mid_idx < len(traj_array):
                ax.annotate(agent_id,
                           xy=(traj_array[mid_idx, 0], traj_array[mid_idx, 1]),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           color=self.colors[agent_type])
        
        # 设置标题和标签
        ax.set_title('Agent Trajectories in Disaster Scenario', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Agent trajectories saved to {save_path}")
        
        return fig
    
    def animate_rescue_operation(self, 
                                episode_data: List[Dict[str, Any]],
                                scenario_data: Dict[str, Any],
                                save_path: Optional[str] = None,
                                fps: int = 10) -> FuncAnimation:
        """
        创建救援操作动画
        
        Args:
            episode_data: 每步的数据列表
            scenario_data: 场景数据
            save_path: 保存路径（可选）
            fps: 帧率
            
        Returns:
            matplotlib动画对象
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 绘制基础场景
        self._plot_base_scenario(ax, scenario_data)
        
        # 初始化绘图元素
        agent_scatters = {}
        victim_scatters = {}
        resource_scatters = {}
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        metrics_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                              fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def init():
            """初始化动画"""
            time_text.set_text('')
            metrics_text.set_text('')
            return [time_text, metrics_text]
        
        def update(frame):
            """更新动画帧"""
            step_data = episode_data[frame]
            step = step_data.get('step', frame)
            
            # 清除之前的智能体、受害者和资源点
            for scatter in list(agent_scatters.values()) + \
                          list(victim_scatters.values()) + \
                          list(resource_scatters.values()):
                scatter.remove()
            
            agent_scatters.clear()
            victim_scatters.clear()
            resource_scatters.clear()
            
            # 绘制智能体
            agents = step_data.get('agents', {})
            for agent_id, agent_info in agents.items():
                pos = agent_info.get('position', (0, 0))
                agent_type = agent_info.get('type', 'drone')
                status = agent_info.get('status', 'idle')
                
                # 根据状态选择标记
                marker = 'o'
                if status == 'rescuing':
                    marker = '^'
                elif status == 'transporting':
                    marker = 's'
                elif status == 'treating':
                    marker = 'D'
                
                scatter = ax.scatter(pos[0], pos[1],
                                    color=self.colors[agent_type],
                                    s=150, marker=marker,
                                    edgecolors='black', linewidth=2,
                                    alpha=0.8)
                agent_scatters[agent_id] = scatter
                
                # 添加智能体标签
                ax.annotate(agent_id,
                           xy=pos,
                           xytext=(0, 15),
                           textcoords='offset points',
                           fontsize=8,
                           color=self.colors[agent_type])
            
            # 绘制受害者
            victims = step_data.get('victims', {})
            for victim_id, victim_info in victims.items():
                pos = victim_info.get('position', (0, 0))
                severity = victim_info.get('severity', 'medium')
                rescued = victim_info.get('rescued', False)
                
                if not rescued:
                    color = self.colors[severity]
                    alpha = 1.0
                else:
                    color = 'lightgray'
                    alpha = 0.3
                
                scatter = ax.scatter(pos[0], pos[1],
                                    color=color,
                                    s=100, marker='o',
                                    edgecolors='black', linewidth=1,
                                    alpha=alpha)
                victim_scatters[victim_id] = scatter
            
            # 绘制资源点
            resources = step_data.get('resources', {})
            for resource_id, resource_info in resources.items():
                pos = resource_info.get('position', (0, 0))
                remaining = resource_info.get('remaining', 100)
                
                scatter = ax.scatter(pos[0], pos[1],
                                    color=self.colors['resource'],
                                    s=remaining,  # 大小表示剩余资源
                                    marker='s',
                                    edgecolors='black', linewidth=1,
                                    alpha=0.6)
                resource_scatters[resource_id] = scatter
                
                # 添加资源标签
                ax.annotate(f'{remaining}',
                           xy=pos,
                           xytext=(0, -15),
                           textcoords='offset points',
                           fontsize=8)
            
            # 更新时间文本
            time_text.set_text(f'Step: {step}/{len(episode_data)-1}\n'
                              f'Time: {step_data.get("timestamp", "N/A")}')
            
            # 更新指标文本
            metrics = step_data.get('metrics', {})
            metrics_str = f"Rescued: {metrics.get('rescued', 0)}\n"
            metrics_str += f"Deaths: {metrics.get('deaths', 0)}\n"
            metrics_str += f"Resources Used: {metrics.get('resources_used', 0)}\n"
            metrics_str += f"Avg Response Time: {metrics.get('avg_response_time', 0):.1f}s"
            
            metrics_text.set_text(metrics_str)
            
            return [time_text, metrics_text] + \
                   list(agent_scatters.values()) + \
                   list(victim_scatters.values()) + \
                   list(resource_scatters.values())
        
        # 创建动画
        animation = FuncAnimation(fig, update, frames=len(episode_data),
                                 init_func=init, blit=True,
                                 interval=1000/fps, repeat=False)
        
        # 设置标题
        ax.set_title('Real-time Rescue Operation Animation', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            animation.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Animation saved to {save_path}")
        
        return animation
    
    def plot_performance_dashboard(self, 
                                  metrics_history: Dict[str, List[float]],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制性能监控仪表盘
        
        Args:
            metrics_history: 指标历史数据 {metric_name: [value1, value2, ...]}
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        # 创建子图
        fig = plt.figure(figsize=(16, 12))
        
        # 定义子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 救援成功率曲线
        ax1 = fig.add_subplot(gs[0, :2])
        if 'rescue_rate' in metrics_history:
            ax1.plot(metrics_history['rescue_rate'], 
                    color='#2E86AB', linewidth=2.5)
            ax1.fill_between(range(len(metrics_history['rescue_rate'])),
                            metrics_history['rescue_rate'],
                            alpha=0.3, color='#2E86AB')
        ax1.set_title('Rescue Success Rate Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. 响应时间曲线
        ax2 = fig.add_subplot(gs[0, 2])
        if 'avg_response_time' in metrics_history:
            ax2.plot(metrics_history['avg_response_time'],
                    color='#A23B72', linewidth=2.5)
        ax2.set_title('Average Response Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 资源利用率
        ax3 = fig.add_subplot(gs[1, 0])
        if 'resource_utilization' in metrics_history:
            ax3.plot(metrics_history['resource_utilization'],
                    color='#F18F01', linewidth=2.5)
        ax3.set_title('Resource Utilization', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Utilization (%)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. 公平性指标
        ax4 = fig.add_subplot(gs[1, 1])
        fairness_metrics = ['gini_index', 'max_min_fairness', 'theil_index']
        colors = ['#C73E1D', '#2E86AB', '#A23B72']
        
        for i, metric in enumerate(fairness_metrics):
            if metric in metrics_history:
                ax4.plot(metrics_history[metric],
                        color=colors[i], linewidth=2,
                        label=metric.replace('_', ' ').title())
        
        ax4.set_title('Fairness Metrics', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Fairness Score')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # 5. 鲁棒性指标
        ax5 = fig.add_subplot(gs[1, 2])
        robustness_metrics = ['performance_under_attack', 'recovery_time']
        colors = ['#3B1F2B', '#DB5461']
        
        for i, metric in enumerate(robustness_metrics):
            if metric in metrics_history:
                ax5.plot(metrics_history[metric],
                        color=colors[i], linewidth=2,
                        label=metric.replace('_', ' ').title())
        
        ax5.set_title('Robustness Metrics', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Robustness Score')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        # 6. 智能体效率热力图
        ax6 = fig.add_subplot(gs[2, :])
        if 'agent_efficiency' in metrics_history:
            # 假设agent_efficiency是二维数组 [episodes x agents]
            efficiency_data = np.array(metrics_history['agent_efficiency'])
            
            if efficiency_data.ndim == 2 and efficiency_data.shape[0] > 0:
                im = ax6.imshow(efficiency_data.T, aspect='auto',
                               cmap='YlOrRd', interpolation='nearest')
                ax6.set_title('Agent Efficiency Heatmap', 
                             fontsize=14, fontweight='bold')
                ax6.set_xlabel('Episode')
                ax6.set_ylabel('Agent ID')
                
                # 添加颜色条
                plt.colorbar(im, ax=ax6, label='Efficiency Score')
        
        # 添加整体标题
        fig.suptitle('EGT-MARL Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    def create_interactive_3d_visualization(self, 
                                           scenario_data: Dict[str, Any],
                                           agent_data: Dict[str, Any]) -> go.Figure:
        """
        创建交互式3D可视化
        
        Args:
            scenario_data: 场景数据
            agent_data: 智能体数据
            
        Returns:
            plotly图形对象
        """
        # 创建3D图形
        fig = go.Figure()
        
        # 添加灾害中心
        center = scenario_data.get('epicenter', (50, 50))
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[0],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond'
            ),
            name='Disaster Epicenter'
        ))
        
        # 添加受害者
        victims = scenario_data.get('victims', [])
        victim_x, victim_y, victim_z, victim_colors = [], [], [], []
        
        for victim in victims:
            pos = victim.get('position', (0, 0))
            severity = victim.get('severity', 'medium')
            
            victim_x.append(pos[0])
            victim_y.append(pos[1])
            victim_z.append(0)
            victim_colors.append(self.colors.get(severity, 'orange'))
        
        if victim_x:
            fig.add_trace(go.Scatter3d(
                x=victim_x, y=victim_y, z=victim_z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=victim_colors,
                    symbol='circle'
                ),
                name='Victims'
            ))
        
        # 添加智能体
        for agent_id, agent_info in agent_data.items():
            pos = agent_info.get('position', (0, 0))
            agent_type = agent_info.get('type', 'drone')
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[10],  # 智能体在空中
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self.colors.get(agent_type, 'blue'),
                    symbol='triangle-up'
                ),
                text=[agent_id],
                textposition="top center",
                name=f'{agent_type.capitalize()}'
            ))
        
        # 更新布局
        fig.update_layout(
            title='3D Disaster Scenario Visualization',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Altitude',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_comparison_chart(self, 
                             algorithms_data: Dict[str, Dict[str, List[float]]],
                             metric: str = 'rescue_rate',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制算法对比图表
        
        Args:
            algorithms_data: 算法数据 {algorithm_name: {metric: values}}
            metric: 要比较的指标
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 颜色调色板
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms_data)))
        
        # 绘制每个算法的指标曲线
        for i, (algo_name, algo_data) in enumerate(algorithms_data.items()):
            if metric in algo_data:
                values = algo_data[metric]
                episodes = range(len(values))
                
                # 绘制曲线
                ax.plot(episodes, values,
                       color=colors[i],
                       linewidth=2.5,
                       label=algo_name)
                
                # 添加置信区间（如果可用）
                if f'{metric}_std' in algo_data:
                    std = algo_data[f'{metric}_std']
                    ax.fill_between(episodes,
                                   np.array(values) - np.array(std),
                                   np.array(values) + np.array(std),
                                   color=colors[i], alpha=0.2)
        
        # 设置图表属性
        ax.set_title(f'Algorithm Comparison: {metric.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # 添加统计信息
        stats_text = "Statistics (Final Episode):\n"
        for algo_name, algo_data in algorithms_data.items():
            if metric in algo_data:
                values = algo_data[metric]
                if values:
                    final_value = values[-1]
                    stats_text += f"{algo_name}: {final_value:.2f}\n"
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to {save_path}")
        
        return fig
    
    def _plot_base_scenario(self, ax, scenario_data: Dict[str, Any]):
        """
        绘制基础场景（内部辅助函数）
        
        Args:
            ax: matplotlib坐标轴
            scenario_data: 场景数据
        """
        map_size = scenario_data.get('map_size', (100, 100))
        
        # 设置坐标轴
        ax.set_xlim(0, map_size[0])
        ax.set_ylim(0, map_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制灾害中心
        center = scenario_data.get('epicenter', (map_size[0]/2, map_size[1]/2))
        radius = scenario_data.get('radius', min(map_size)/4)
        disaster_type = scenario_data.get('disaster_type', 'earthquake')
        
        disaster_color = self.colors.get(disaster_type, '#FF6B6B')
        circle = patches.Circle(center, radius, 
                               color=disaster_color, 
                               alpha=0.1,
                               linewidth=0)
        ax.add_patch(circle)
        
        # 绘制资源点
        resources = scenario_data.get('resources', [])
        for resource in resources:
            pos = resource.get('position', (0, 0))
            ax.scatter(pos[0], pos[1],
                      color=self.colors['resource'],
                      s=50, marker='s',
                      alpha=0.6,
                      edgecolors='black', linewidth=1)
        
        # 绘制医院
        hospitals = scenario_data.get('hospitals', [])
        for hospital in hospitals:
            pos = hospital.get('position', (0, 0))
            ax.scatter(pos[0], pos[1],
                      color=self.colors['hospital'],
                      s=100, marker='H',
                      edgecolors='black', linewidth=2)
    
    def save_visualization_report(self, 
                                 visualizations: Dict[str, plt.Figure],
                                 report_path: str):
        """
        保存可视化报告
        
        Args:
            visualizations: 可视化图形字典 {name: figure}
            report_path: 报告保存路径
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(report_path) as pdf:
            for name, fig in visualizations.items():
                pdf.savefig(fig, bbox_inches='tight')
                print(f"Added {name} to report")
        
        print(f"Visualization report saved to {report_path}")


class RealTimeMonitor:
    """实时性能监控器"""
    
    def __init__(self, update_interval: float = 1.0):
        """
        初始化实时监控器
        
        Args:
            update_interval: 更新间隔（秒）
        """
        self.update_interval = update_interval
        self.metrics_history = {}
        self.start_time = time.time()
        
        # 创建实时图表
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.axes = self.axes.flatten()
        
    def update(self, metrics: Dict[str, float]):
        """
        更新监控数据
        
        Args:
            metrics: 当前指标数据
        """
        current_time = time.time() - self.start_time
        
        # 更新历史数据
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = {'time': [], 'value': []}
            
            self.metrics_history[metric_name]['time'].append(current_time)
            self.metrics_history[metric_name]['value'].append(value)
        
        # 更新图表
        self._update_plots()
        
    def _update_plots(self):
        """更新实时图表"""
        # 清除所有坐标轴
        for ax in self.axes:
            ax.clear()
        
        # 绘制救援成功率
        if 'rescue_rate' in self.metrics_history:
            ax = self.axes[0]
            data = self.metrics_history['rescue_rate']
            ax.plot(data['time'], data['value'], 'b-', linewidth=2)
            ax.set_title('Rescue Success Rate', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rate (%)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        # 绘制响应时间
        if 'avg_response_time' in self.metrics_history:
            ax = self.axes[1]
            data = self.metrics_history['avg_response_time']
            ax.plot(data['time'], data['value'], 'r-', linewidth=2)
            ax.set_title('Average Response Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.grid(True, alpha=0.3)
        
        # 绘制资源利用率
        if 'resource_utilization' in self.metrics_history:
            ax = self.axes[2]
            data = self.metrics_history['resource_utilization']
            ax.plot(data['time'], data['value'], 'g-', linewidth=2)
            ax.set_title('Resource Utilization', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Utilization (%)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        # 绘制公平性指标
        if 'gini_index' in self.metrics_history:
            ax = self.axes[3]
            data = self.metrics_history['gini_index']
            ax.plot(data['time'], data['value'], 'm-', linewidth=2, label='Gini')
            
            if 'max_min_fairness' in self.metrics_history:
                data2 = self.metrics_history['max_min_fairness']
                ax.plot(data2['time'], data2['value'], 'c-', linewidth=2, label='Max-Min')
            
            ax.set_title('Fairness Metrics', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Fairness Score')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        # 更新图表
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        """关闭监控器"""
        plt.ioff()
        plt.close(self.fig)


# 使用示例
if __name__ == "__main__":
    # 示例配置
    sample_config = {
        'map_size': (100, 100),
        'disaster_type': 'earthquake',
        'severity': 'high',
        'epicenter': (50, 50),
        'radius': 30
    }
    
    # 示例场景数据
    sample_scenario = {
        'disaster_type': 'earthquake',
        'severity': 'high',
        'map_size': (100, 100),
        'epicenter': (50, 50),
        'radius': 30,
        'resources': [
            {'position': (20, 20), 'capacity': 100},
            {'position': (80, 80), 'capacity': 150},
            {'position': (30, 70), 'capacity': 80}
        ],
        'victims': [
            {'position': (40, 40), 'severity': 'high'},
            {'position': (60, 60), 'severity': 'medium'},
            {'position': (70, 30), 'severity': 'low'}
        ],
        'hospitals': [
            {'position': (10, 10), 'capacity': 50},
            {'position': (90, 90), 'capacity': 100}
        ]
    }
    
    # 示例智能体数据
    sample_agents = {
        'drone_1': [(10, 10), (20, 20), (30, 30), (40, 40)],
        'ambulance_1': [(90, 90), (80, 80), (70, 70), (60, 60)],
        'mobile_hospital_1': [(50, 50), (55, 55), (60, 60)]
    }
    
    # 创建可视化器
    visualizer = DisasterVisualizer(sample_config)
    
    print("Creating visualizations...")
    
    # 1. 绘制场景地图
    fig1 = visualizer.plot_scenario_map(sample_scenario, 
                                       save_path='scenario_map.png')
    print("✓ Scenario map created")
    
    # 2. 绘制智能体轨迹
    fig2 = visualizer.plot_agent_trajectories(sample_agents, sample_scenario,
                                             save_path='agent_trajectories.png')
    print("✓ Agent trajectories created")
    
    # 3. 示例性能数据
    sample_metrics = {
        'rescue_rate': [10, 20, 30, 40, 50, 60, 70, 80, 85, 90],
        'avg_response_time': [120, 110, 100, 95, 90, 85, 80, 75, 70, 65],
        'resource_utilization': [20, 30, 40, 50, 60, 65, 70, 75, 80, 85],
        'gini_index': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
        'max_min_fairness': [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        'theil_index': [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
        'performance_under_attack': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'recovery_time': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        'agent_efficiency': np.random.rand(10, 5)  # 10个episode，5个智能体
    }
    
    # 4. 绘制性能仪表盘
    fig3 = visualizer.plot_performance_dashboard(sample_metrics,
                                                save_path='performance_dashboard.png')
    print("✓ Performance dashboard created")
    
    # 5. 算法对比示例
    algorithms_data = {
        'EGT-MARL': {
            'rescue_rate': [10, 25, 45, 60, 75, 85, 90, 92, 94, 95],
            'rescue_rate_std': [2, 3, 4, 5, 4, 3, 2, 2, 1, 1]
        },
        'QMIX': {
            'rescue_rate': [5, 15, 30, 45, 60, 70, 78, 83, 87, 90],
            'rescue_rate_std': [3, 4, 5, 6, 5, 4, 3, 3, 2, 2]
        },
        'MADDPG': {
            'rescue_rate': [8, 20, 35, 50, 65, 75, 82, 86, 89, 91],
            'rescue_rate_std': [2, 3, 4, 5, 4, 3, 3, 2, 2, 1]
        }
    }
    
    fig4 = visualizer.plot_comparison_chart(algorithms_data, 'rescue_rate',
                                           save_path='algorithm_comparison.png')
    print("✓ Algorithm comparison chart created")
    
    # 6. 创建可视化报告
    report_visualizations = {
        'Scenario Map': fig1,
        'Agent Trajectories': fig2,
        'Performance Dashboard': fig3,
        'Algorithm Comparison': fig4
    }
    
    visualizer.save_visualization_report(report_visualizations,
                                        'visualization_report.pdf')
    print("✓ Visualization report created")
    
    print("\nAll visualizations created successfully!")
    print("Check the generated files:")
    print("  - scenario_map.png")
    print("  - agent_trajectories.png")
    print("  - performance_dashboard.png")
    print("  - algorithm_comparison.png")
    print("  - visualization_report.pdf")
    
    # 显示一个图表
    plt.show()