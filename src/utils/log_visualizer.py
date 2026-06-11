import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

class LogVisualizer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.steps = []
        self.agent_positions = {}
        self.casualty_positions = {}
        self.rescued_casualties = set()
        self.dead_casualties = set()
        self.treated_casualties = {}
        self.affected_areas = []
        self.map_size = 800
        self._parse_log()

    def _parse_log(self):
        """Parse the training log file to extract agent and casualty positions over time."""
        step_data = {}
        current_step = None

        with open(self.log_file_path, 'r') as f:
            for line in f:
                line = line.strip()

                step_match = re.search(r'\[STEP (\d+)\]', line)
                if step_match:
                    if current_step is not None and step_data:
                        self.steps.append(current_step)
                        for aid, pos in step_data.get('agents', {}).items():
                            if aid not in self.agent_positions:
                                self.agent_positions[aid] = {}
                            self.agent_positions[aid][current_step] = pos

                    current_step = int(step_match.group(1))
                    step_data = {'agents': {}, 'casualties': {}}
                    continue

                agent_match = re.search(r'\[AGENT (\d+)\].*Position=\[([\d.]+), ([\d.]+)\]', line)
                if agent_match and current_step is not None:
                    aid = int(agent_match.group(1))
                    x = float(agent_match.group(2))
                    y = float(agent_match.group(3))
                    step_data['agents'][aid] = (x, y)
                    continue

                casualty_found_match = re.search(r'\[CASUALTY FOUND\].*CasualtyID=(\d+).*Position=\[([\d.]+), ([\d.]+)\]', line)
                if casualty_found_match:
                    cid = int(casualty_found_match.group(1))
                    x = float(casualty_found_match.group(2))
                    y = float(casualty_found_match.group(3))
                    if cid not in self.casualty_positions:
                        self.casualty_positions[cid] = (x, y)
                    continue

                rescued_match = re.search(r'\[CASUALTY RESCUED\] ID=(\d+)', line)
                if rescued_match:
                    cid = int(rescued_match.group(1))
                    self.rescued_casualties.add(cid)
                    continue

                death_match = re.search(r'\[CASUALTY DEATH\] ID=(\d+)', line)
                if death_match:
                    cid = int(death_match.group(1))
                    self.dead_casualties.add(cid)
                    continue

        if current_step is not None and step_data:
            self.steps.append(current_step)
            for aid, pos in step_data.get('agents', {}).items():
                if aid not in self.agent_positions:
                    self.agent_positions[aid] = {}
                self.agent_positions[aid][current_step] = pos

    def visualize(self, output_file='simulation.gif', fps=1, max_frames=None):
        """Create an animation showing agent and casualty movements.

        Args:
            output_file: Output file path
            fps: Frames per second (lower = slower)
            max_frames: Maximum number of frames (None = no limit)
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_aspect('equal')
        ax.set_title('Emergency Response Simulation')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')

        center_x, center_y = self.map_size / 2, self.map_size / 2
        affected_radius = 140
        ax.add_patch(Circle((center_x, center_y), affected_radius, color='red', alpha=0.2, label='Affected Area'))

        agent_types = {
            0: 'drone', 1: 'vehicle', 2: 'personnel',
            3: 'drone', 4: 'vehicle', 5: 'personnel',
            6: 'drone', 7: 'vehicle', 8: 'personnel',
            9: 'drone', 10: 'vehicle', 11: 'personnel',
            12: 'drone', 13: 'vehicle', 14: 'personnel',
            15: 'drone', 16: 'vehicle', 17: 'personnel',
            18: 'drone', 19: 'vehicle'
        }

        agent_colors = {
            'drone': 'blue',
            'vehicle': 'green',
            'personnel': 'orange'
        }

        legend_elements = []
        type_counts = {'drone': 0, 'vehicle': 0, 'personnel': 0}

        for aid in self.agent_positions:
            a_type = agent_types.get(aid, 'personnel')
            type_counts[a_type] += 1
            marker, = ax.plot([], [], 'o', color=agent_colors[a_type], markersize=8)

        for a_type, count in type_counts.items():
            if count > 0:
                marker, = ax.plot([], [], 'o', color=agent_colors[a_type], markersize=8,
                                label=f'{a_type.capitalize()} (n={count})')
                legend_elements.append(marker)

        rescued_count = len(self.rescued_casualties)
        dead_count = len(self.dead_casualties)
        total_casualties = len(self.casualty_positions)
        remaining_count = total_casualties - rescued_count - dead_count

        if remaining_count > 0:
            marker, = ax.plot([], [], 'x', color='red', markersize=6, label=f'Casualty - Remaining (n={remaining_count})')
            legend_elements.append(marker)
        if rescued_count > 0:
            marker, = ax.plot([], [], 'x', color='cyan', markersize=6, label=f'Casualty - Rescued (n={rescued_count})')
            legend_elements.append(marker)
        if dead_count > 0:
            marker, = ax.plot([], [], 'x', color='black', markersize=6, label=f'Casualty - Dead (n={dead_count})')
            legend_elements.append(marker)

        ax.legend(handles=legend_elements, loc='upper right')

        agent_markers = {}
        for aid in self.agent_positions:
            a_type = agent_types.get(aid, 'personnel')
            marker, = ax.plot([], [], 'o', color=agent_colors[a_type], markersize=8)
            agent_markers[aid] = marker

        casualty_markers = {}
        for cid, pos in self.casualty_positions.items():
            if cid in self.rescued_casualties:
                color = 'cyan'
            elif cid in self.dead_casualties:
                color = 'black'
            else:
                color = 'red'
            marker, = ax.plot([], [], 'x', color=color, markersize=6)
            casualty_markers[cid] = marker

        ax.grid(True, alpha=0.3)

        total_frames = len(self.steps)
        if max_frames is not None and total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))

        def init():
            for marker in agent_markers.values():
                marker.set_data([], [])
            for marker in casualty_markers.values():
                marker.set_data([], [])
            return list(agent_markers.values()) + list(casualty_markers.values())

        def update(frame_idx):
            real_frame = frame_indices[frame_idx]
            step = self.steps[real_frame]

            for aid, marker in agent_markers.items():
                if step in self.agent_positions[aid]:
                    x, y = self.agent_positions[aid][step]
                    marker.set_data([x], [y])
                else:
                    marker.set_data([], [])

            for cid, marker in casualty_markers.items():
                if cid not in self.rescued_casualties and cid not in self.dead_casualties:
                    x, y = self.casualty_positions[cid]
                    marker.set_data([x], [y])
                elif cid in self.rescued_casualties and frame_idx < len(frame_indices) - 1:
                    marker.set_data([], [])
                elif cid in self.dead_casualties and frame_idx < len(frame_indices) - 1:
                    marker.set_data([], [])

            ax.set_title(f'Emergency Response Simulation - Step {step}')
            return list(agent_markers.values()) + list(casualty_markers.values())

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), init_func=init,
                                        interval=1000/fps, blit=True)

        writer = animation.PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        print(f"Total frames: {len(frame_indices)} out of {total_frames} steps, fps={fps}")

        plt.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python log_visualizer.py <log_file_path> [output_file]")
        sys.exit(1)

    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'simulation.gif'

    visualizer = LogVisualizer(log_file)
    visualizer.visualize(output_file)