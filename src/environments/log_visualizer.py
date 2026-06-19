"""
基于日志的训练过程可视化工具

从training.log文件中解析数据并生成动图动画
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings('ignore')

class LogVisualizer:
    """日志可视化器"""
    
    def __init__(self):
        # 颜色配置
        self.colors = {
            'agent_personnel': '#1F77B4',      # 人员 - 蓝色
            'agent_vehicle': '#FF7F0E',         # 车辆 - 橙色
            'agent_drone': '#2CA02C',           # 无人机 - 绿色
            'casualty_critical': '#9467BD',     # 危急 - 深紫色
            'casualty_severe': '#E377C2',       # 严重 - 粉色
            'casualty_moderate': '#FF7F0E',     # 中等 - 橙色
            'casualty_mild': '#2CA02C',         # 轻微 - 绿色
            'depot': '#17BECF',                 # 资源点 - 青色
            'hospital': '#7F7F7F',              # 医院 - 灰色
            'rescued': '#98D8C8',               # 已救援 - 浅绿
            'dead': '#E4E4E4',                  # 死亡 - 灰色
        }
        
        # 状态标记 - 用户要求：圆形=agent，三角形=伤员，矩形=depot
        self.status_markers = {
            'exploring': 'o',      # 圆形
            'go_to_casualty': 'o', # 圆形
            'treat_casualty': 'o', # 圆形
            'go_to_depot': 'o',    # 圆形
            'idle': 'o',           # 圆形
        }
        
        # 大小配置
        self.sizes = {
            'agent_personnel': 120,
            'agent_vehicle': 180,
            'agent_drone': 100,
            'casualty': 100,       # 伤员变大一点
            'depot': 250,          # depot更大
        }
    
    def parse_log_file(self, log_path: str) -> dict:
        """
        解析日志文件
        
        Args:
            log_path: 日志文件路径
            
        Returns:
            解析后的数据字典
        """
        episode_data = []
        current_step = None
        current_agents = {}
        current_casualties = {}
        current_rescued = 0
        current_deaths = 0
        current_time = 0
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 解析Step信息
                step_match = re.search(r'Step (\d+): Reward=([\d.]+), Rescued=(\d+), Deaths=(\d+)', line)
                if step_match:
                    if current_step is not None and current_agents:
                        episode_data.append({
                            'step': current_step,
                            'time': current_time,
                            'agents': current_agents.copy(),
                            'casualties': current_casualties.copy(),
                            'rescued': current_rescued,
                            'deaths': current_deaths,
                        })
                    
                    current_step = int(step_match.group(1))
                    current_rescued = int(step_match.group(3))
                    current_deaths = int(step_match.group(4))
                    current_agents = {}
                    current_casualties = {}
                    continue
                
                # 解析时间戳
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                if time_match:
                    # 提取秒数
                    time_str = time_match.group(1)
                    parts = time_str.split(':')
                    current_time = int(parts[0].split(' ')[1]) * 3600 + int(parts[1]) * 60 + float(parts[2].replace(',', '.'))
                
                # 解析Agent信息
                agent_match = re.search(
                    r'AGENT (\d+)/(\w+) \| Status=(\w+) \| Pos=\[([\d.]+),([\d.]+)\] \| Rescued=(\d+) \| Known=\[(.+?)\] \| Resources=(.+)',
                    line
                )
                if agent_match:
                    agent_id = int(agent_match.group(1))
                    agent_type = agent_match.group(2).lower()
                    status = agent_match.group(3)
                    pos_x = float(agent_match.group(4))
                    pos_y = float(agent_match.group(5))
                    rescued = int(agent_match.group(6))
                    known_str = agent_match.group(7)
                    
                    # 解析已知伤员
                    known_casualties = []
                    if known_str and known_str != '[]':
                        try:
                            known_casualties = [int(x.strip()) for x in known_str.split(',') if x.strip()]
                        except:
                            pass
                    
                    current_agents[agent_id] = {
                        'type': agent_type,
                        'status': status,
                        'position': (pos_x, pos_y),
                        'rescued': rescued,
                        'known_casualties': known_casualties,
                    }
                    continue
                
                # 解析Casualty信息
                casualty_match = re.search(
                    r'CASUALTY (\d+) \| Pos=\[([\d.]+),([\d.]+)\] \| Sev=(\w+) \| Status=(\w+) \| Survival=([\d.]+) \| DiscoveredBy=(\w+) \| Treating=(\w+)',
                    line
                )
                if casualty_match:
                    casualty_id = int(casualty_match.group(1))
                    pos_x = float(casualty_match.group(2))
                    pos_y = float(casualty_match.group(3))
                    severity = casualty_match.group(4)
                    status = casualty_match.group(5)
                    survival = float(casualty_match.group(6))
                    discovered_by = casualty_match.group(7)
                    treating = casualty_match.group(8)
                    
                    current_casualties[casualty_id] = {
                        'position': (pos_x, pos_y),
                        'severity': severity.lower(),
                        'status': status,
                        'survival': survival,
                        'discovered_by': None if discovered_by == 'None' else int(discovered_by),
                        'treating': None if treating == 'None' else int(treating),
                    }
                    continue
        
        # 添加最后一步
        if current_step is not None and current_agents:
            episode_data.append({
                'step': current_step,
                'time': current_time,
                'agents': current_agents,
                'casualties': current_casualties,
                'rescued': current_rescued,
                'deaths': current_deaths,
            })
        
        # 反转列表，确保step从小到大排列
        episode_data.reverse()
        
        return episode_data
    
    def create_animation(self, episode_data: list, save_path: str = None, fps: int = 5):
        """
        创建训练过程动画
        
        Args:
            episode_data: 解析后的每步数据
            save_path: 保存路径（可选）
            fps: 帧率
        """
        if not episode_data:
            print("No data to visualize")
            return None
        
        # 获取地图边界
        all_x = []
        all_y = []
        for step_data in episode_data:
            for agent in step_data['agents'].values():
                all_x.append(agent['position'][0])
                all_y.append(agent['position'][1])
            for casualty in step_data['casualties'].values():
                all_x.append(casualty['position'][0])
                all_y.append(casualty['position'][1])
        
        x_min, x_max = min(all_x) - 50, max(all_x) + 50
        y_min, y_max = min(all_y) - 50, max(all_y) + 50
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title('Disaster Rescue Simulation', fontsize=14, fontweight='bold')
        
        # 初始化绘图元素
        agent_scatters = {}
        casualty_scatters = {}
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        metrics_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                              fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 存储标注文本对象
        annotations = []
        
        def update(frame):
            """更新动画帧"""
            step_data = episode_data[frame]
            
            # 清除之前的绘图
            for scatter in list(agent_scatters.values()) + list(casualty_scatters.values()):
                scatter.remove()
            agent_scatters.clear()
            casualty_scatters.clear()
            
            # 清除之前的标注
            for ann in annotations:
                ann.remove()
            annotations.clear()
            
            # 绘制智能体
            for agent_id, agent_info in step_data['agents'].items():
                pos = agent_info['position']
                agent_type = agent_info['type']
                status = agent_info['status']
                
                # 选择颜色
                if agent_type == 'drone':
                    color = self.colors['agent_drone']
                    size = self.sizes['agent_drone']
                    label = f'D{agent_id}'
                elif agent_type == 'vehicle':
                    color = self.colors['agent_vehicle']
                    size = self.sizes['agent_vehicle']
                    label = f'V{agent_id}'
                else:
                    color = self.colors['agent_personnel']
                    size = self.sizes['agent_personnel']
                    label = f'P{agent_id}'
                
                # 选择标记
                marker = self.status_markers.get(status, 'o')
                
                scatter = ax.scatter(pos[0], pos[1],
                                    color=color,
                                    s=size, marker=marker,
                                    edgecolors='black', linewidth=2,
                                    alpha=0.9)
                agent_scatters[agent_id] = scatter
                
                # 添加标签（仅显示ID）
                ann = ax.annotate(label, xy=pos,
                                 xytext=(3, 3), textcoords='offset points',
                                 fontsize=9, fontweight='bold', color=color)
                annotations.append(ann)
            
            # 绘制伤员（使用三角形）
            for casualty_id, casualty_info in step_data['casualties'].items():
                pos = casualty_info['position']
                severity = casualty_info['severity']
                status = casualty_info['status']
                survival = casualty_info['survival']
                
                # 选择颜色
                if status == 'RESCUED':
                    color = self.colors['rescued']
                    marker = 's'  # 已救援用方形
                    alpha = 0.5
                elif status == 'DEAD':
                    color = self.colors['dead']
                    marker = 'x'  # 死亡用叉
                    alpha = 0.4
                else:
                    color_key = f'casualty_{severity}'
                    color = self.colors.get(color_key, self.colors['casualty_moderate'])
                    marker = '^'  # 存活伤员用三角形
                    alpha = 0.4 + (survival * 0.6)
                
                scatter = ax.scatter(pos[0], pos[1],
                                    color=color,
                                    s=self.sizes['casualty'], marker=marker,
                                    edgecolors='black', linewidth=1,
                                    alpha=alpha)
                casualty_scatters[casualty_id] = scatter
                
                # 仅对存活概率低的伤员显示概率（< 0.7）
                if status not in ['RESCUED', 'DEAD'] and survival < 0.7:
                    ann = ax.annotate(f'{survival:.1f}', xy=pos,
                                      xytext=(3, -12), textcoords='offset points',
                                      fontsize=7, color='darkred', fontweight='bold')
                    annotations.append(ann)
            
            # 更新时间文本
            time_text.set_text(f'Step: {step_data["step"]}\nTime: {step_data["time"]:.1f}s')
            
            # 更新指标文本
            metrics_text.set_text(
                f"Rescued: {step_data['rescued']}\n"
                f"Deaths: {step_data['deaths']}\n"
                f"Agents: {len(step_data['agents'])}\n"
                f"Casualties: {len(step_data['casualties'])}"
            )
            
            return [time_text, metrics_text] + list(agent_scatters.values()) + list(casualty_scatters.values()) + annotations
        
        # 创建动画
        animation = FuncAnimation(fig, update, frames=len(episode_data),
                                 blit=True, interval=1000/fps, repeat=False)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['agent_personnel'], label='Personnel ● (P#)'),
            Patch(facecolor=self.colors['agent_vehicle'], label='Vehicle ● (V#)'),
            Patch(facecolor=self.colors['agent_drone'], label='Drone ● (D#)'),
            Patch(facecolor=self.colors['casualty_critical'], label='Critical ▲'),
            Patch(facecolor=self.colors['casualty_severe'], label='Severe ▲'),
            Patch(facecolor=self.colors['casualty_moderate'], label='Moderate ▲'),
            Patch(facecolor=self.colors['casualty_mild'], label='Mild ▲'),
            Patch(facecolor=self.colors['rescued'], label='Rescued □'),
            Patch(facecolor=self.colors['dead'], label='Dead ✕'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, title='Legend')
        
        # 添加说明文本
        help_text = ax.text(0.98, 0.55, 
                           '图例说明:\n'
                           '● Agent (圆形)\n'
                           '▲ Casualty (三角)\n'
                           '■ Depot (矩形)\n'
                           '数字<0.7=存活概率',
                           transform=ax.transAxes,
                           fontsize=7,
                           ha='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        annotations.append(help_text)
        
        plt.tight_layout()
        
        if save_path:
            # 尝试使用ffmpeg，如果不可用则使用pillow
            try:
                animation.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
            except:
                print("FFmpeg not available, using Pillow")
                animation.save(save_path, writer='pillow', fps=fps, dpi=100)
            print(f"Animation saved to {save_path}")
        
        return animation

    def visualize_from_log(self, log_path: str, save_path: str = None, fps: int = 5, max_frames: int = 100):
        """
        从日志文件生成可视化动画
        
        Args:
            log_path: 日志文件路径
            save_path: 保存路径（可选）
            fps: 帧率
            max_frames: 最大帧数（用于采样，避免处理过多帧）
        """
        print(f"Parsing log file: {log_path}")
        episode_data = self.parse_log_file(log_path)
        total_steps = len(episode_data)
        print(f"Parsed {total_steps} steps")
        
        if not episode_data:
            print("No data found in log file")
            return None
        
        # 如果帧数太多，进行采样
        if total_steps > max_frames:
            step = total_steps // max_frames
            episode_data = episode_data[::step][:max_frames]
            print(f"Sampled {len(episode_data)} frames from {total_steps} steps")
        
        return self.create_animation(episode_data, save_path, fps)


if __name__ == '__main__':
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python log_visualizer.py <log_file> [output.gif] [fps] [max_frames]")
        print("Example: python log_visualizer.py training.log animation.gif 5 50")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    # 默认输出路径
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    else:
        # 自动生成输出路径
        log_dir = os.path.dirname(log_path)
        base_name = os.path.basename(log_path).replace('.log', '_animation.gif')
        save_path = os.path.join(log_dir, base_name)
    
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    print(f"Log file: {log_path}")
    print(f"Output file: {save_path}")
    print(f"FPS: {fps}")
    print(f"Max frames: {max_frames}")
    
    visualizer = LogVisualizer()
    animation = visualizer.visualize_from_log(log_path, save_path, fps, max_frames)
    
    if animation:
        print("Animation created successfully!")
        # 非交互式环境不调用plt.show()
        try:
            plt.show(block=False)
            plt.close('all')
        except:
            pass