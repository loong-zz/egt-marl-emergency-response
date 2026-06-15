import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class StrategyDetectionManager:
    """
    策略性行为检测管理器
    检测agent的虚报需求和资源囤积行为
    """
    
    def __init__(self):
        # 存储每个agent报告的伤员信息
        self.reported_casualties = defaultdict(list)  # {agent_id: [(position, timestamp, severity), ...]}
        
        # 存储每个agent的实际救援记录
        self.rescue_records = defaultdict(list)  # {agent_id: [(casualty_id, timestamp, success), ...]}
        
        # 存储每个agent的资源使用记录
        self.resource_usage = defaultdict(list)  # {agent_id: [(timestamp, resources, capacity), ...]}
        
        # 存储检测到的策略性行为
        self.detected_strategies = defaultdict(list)  # {agent_id: [(type, timestamp, reason), ...]}
        
        # 检测阈值配置
        self.false_report_threshold = 0.3  # 虚假报告比例阈值
        self.resource_hoarding_threshold = 0.8  # 资源囤积阈值（资源/容量）
        self.min_reports_for_detection = 5  # 最小报告数量
        
        # 历史数据窗口
        self.history_window = 100  # 保留最近100步的数据
    
    def report_casualty(self, agent_id: int, position: np.ndarray, timestamp: float, severity: str):
        """
        记录agent报告的伤员信息
        """
        self.reported_casualties[agent_id].append({
            'position': position.copy(),
            'timestamp': timestamp,
            'severity': severity,
            'verified': False
        })
        
        # 保持历史窗口大小
        if len(self.reported_casualties[agent_id]) > self.history_window:
            self.reported_casualties[agent_id].pop(0)
    
    def record_rescue(self, agent_id: int, casualty_id: int, timestamp: float, success: bool):
        """
        记录agent的实际救援记录
        """
        self.rescue_records[agent_id].append({
            'casualty_id': casualty_id,
            'timestamp': timestamp,
            'success': success
        })
        
        if len(self.rescue_records[agent_id]) > self.history_window:
            self.rescue_records[agent_id].pop(0)
    
    def record_resource_state(self, agent_id: int, timestamp: float, resources: float, capacity: float):
        """
        记录agent的资源状态
        """
        self.resource_usage[agent_id].append({
            'timestamp': timestamp,
            'resources': resources,
            'capacity': capacity
        })
        
        if len(self.resource_usage[agent_id]) > self.history_window:
            self.resource_usage[agent_id].pop(0)
    
    def verify_reports(self, verified_casualties: dict):
        """
        验证agent报告的伤员真实性
        verified_casualties: {casualty_id: {'position': np.ndarray, 'severity': str}}
        """
        for agent_id, reports in self.reported_casualties.items():
            unverified_count = 0
            
            for report in reports:
                if report['verified']:
                    continue
                
                # 检查报告的伤员是否存在
                found = False
                for casualty in verified_casualties.values():
                    pos = casualty.get('position')
                    if pos is not None:
                        distance = np.linalg.norm(report['position'] - pos)
                        if distance < 10.0:  # 10单位范围内认为是同一个伤员
                            report['verified'] = True
                            found = True
                            break
                
                if not found:
                    unverified_count += 1
            
            # 检测虚报行为
            total_reports = len(reports)
            if total_reports >= self.min_reports_for_detection:
                false_report_ratio = unverified_count / total_reports
                if false_report_ratio > self.false_report_threshold:
                    self._detect_strategy(agent_id, 'false_reporting', 
                                         f"虚假报告比例 {false_report_ratio:.2f} > 阈值 {self.false_report_threshold}")
    
    def detect_resource_hoarding(self, agent_id: int):
        """
        检测资源囤积行为
        """
        records = self.resource_usage.get(agent_id, [])
        if len(records) < self.min_reports_for_detection:
            return False
        
        # 计算平均资源使用率
        avg_utilization = sum(r['resources'] / r['capacity'] for r in records) / len(records)
        
        # 检查是否长时间保持高资源状态
        high_resource_count = sum(1 for r in records if r['resources'] / r['capacity'] > self.resource_hoarding_threshold)
        high_resource_ratio = high_resource_count / len(records)
        
        # 同时检查救援活跃度
        rescue_activity = len(self.rescue_records.get(agent_id, []))
        
        # 如果资源使用率高但救援活动少，可能存在囤积行为
        if avg_utilization > self.resource_hoarding_threshold and high_resource_ratio > 0.7 and rescue_activity < 5:
            self._detect_strategy(agent_id, 'resource_hoarding',
                                 f"资源使用率 {avg_utilization:.2f}, 高资源占比 {high_resource_ratio:.2f}, 救援次数 {rescue_activity}")
            return True
        
        return False
    
    def detect_unfair_claiming(self, agent_id: int, global_stats: dict):
        """
        检测不公平索取行为
        检查agent是否索取了超过其公平份额的资源
        """
        if agent_id not in self.rescue_records:
            return False
        
        agent_rescues = len(self.rescue_records[agent_id])
        total_rescues = global_stats.get('total_rescues', 1)
        agent_fair_share = 1.0 / global_stats.get('num_agents', 1)
        
        # 计算实际救援比例与公平份额的偏差
        actual_share = agent_rescues / total_rescues
        share_ratio = actual_share / agent_fair_share if agent_fair_share > 0 else float('inf')
        
        # 如果救援比例超过公平份额的2倍，可能存在不公平索取
        if share_ratio > 2.0:
            self._detect_strategy(agent_id, 'unfair_claiming',
                                 f"救援份额 {actual_share:.2f} 是公平份额 {agent_fair_share:.2f} 的 {share_ratio:.1f} 倍")
            return True
        
        return False
    
    def _detect_strategy(self, agent_id: int, strategy_type: str, reason: str):
        """
        记录检测到的策略性行为
        """
        timestamp = self._get_current_timestamp()
        self.detected_strategies[agent_id].append({
            'type': strategy_type,
            'timestamp': timestamp,
            'reason': reason
        })
        
        # 保持历史记录
        if len(self.detected_strategies[agent_id]) > 20:
            self.detected_strategies[agent_id].pop(0)
        
        logger.warning(f"[STRATEGY DETECTION] Agent {agent_id} detected {strategy_type}: {reason}")
    
    def _get_current_timestamp(self) -> float:
        """获取当前时间戳（简化实现）"""
        import time
        return time.time()
    
    def get_detection_summary(self) -> dict:
        """
        获取策略性行为检测摘要
        """
        summary = {
            'total_agents': len(self.reported_casualties),
            'agents_with_detections': len(self.detected_strategies),
            'detections_by_type': defaultdict(int),
            'agent_detections': {}
        }
        
        for agent_id, detections in self.detected_strategies.items():
            summary['agent_detections'][agent_id] = len(detections)
            for det in detections:
                summary['detections_by_type'][det['type']] += 1
        
        return dict(summary)
    
    def reset(self):
        """重置检测状态"""
        self.reported_casualties.clear()
        self.rescue_records.clear()
        self.resource_usage.clear()
        self.detected_strategies.clear()