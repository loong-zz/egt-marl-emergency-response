corrected_allocations[resource_type] = max(0.0, corrected)
                else:
                    corrected_allocations[resource_type] = original_allocations.get(resource_type, 0.0)
            
            corrected_action['resource_allocation'] = corrected_allocations
        
        # Add correction metadata
        corrected_action['_corrected'] = True
        corrected_action['_original_reputation'] = reputation
        corrected_action['_correction_strength'] = self.correction_strength * (1 - reputation)
        
        return corrected_action
    
    def update(self, batch: Dict[str, Any]) -> float:
        """
        Update anti-spoofing mechanism.
        
        Args:
            batch: Experience batch
            
        Returns:
            Loss value
        """
        # In practice, would update based on verification outcomes
        # For now, return placeholder loss
        return 0.0
    
    def get_detection_rate(self) -> float:
        """Get spoofing detection rate."""
        if len(self.detection_history) == 0:
            return 0.0
        
        spoofing_count = sum(1 for _, _, is_spoofing in self.detection_history if is_spoofing)
        return spoofing_count / len(self.detection_history)
    
    def get_correction_rate(self) -> float:
        """Get action correction rate."""
        if len(self.correction_history) == 0:
            return 0.0
        
        correction_count = sum(1 for _, status in self.correction_history if status == 'corrected')
        return correction_count / len(self.correction_history)
    
    def get_reputation_report(self) -> Dict[str, Any]:
        """Get reputation system report."""
        if self.reputation_system is None:
            return {'error': 'Reputation system not initialized'}
        
        return self.reputation_system.get_reputation_report()
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get spoofing detection statistics."""
        if len(self.detection_history) == 0:
            return {
                'total_checks': 0,
                'spoofing_detected': 0,
                'detection_rate': 0.0,
                'avg_spoofing_score': 0.0
            }
        
        total_checks = len(self.detection_history)
        spoofing_detected = sum(1 for _, _, is_spoofing in self.detection_history if is_spoofing)
        avg_spoofing_score = np.mean([score for _, score, _ in self.detection_history])
        
        return {
            'total_checks': total_checks,
            'spoofing_detected': spoofing_detected,
            'detection_rate': spoofing_detected / total_checks,
            'avg_spoofing_score': avg_spoofing_score,
            'recent_detection_rate': self._get_recent_detection_rate()
        }
    
    def _get_recent_detection_rate(self, window: int = 100) -> float:
        """Get detection rate in recent history."""
        if len(self.detection_history) == 0:
            return 0.0
        
        recent = self.detection_history[-window:]
        if not recent:
            return 0.0
        
        spoofing_count = sum(1 for _, _, is_spoofing in recent if is_spoofing)
        return spoofing_count / len(recent)
    
    def save(self, path: str) -> None:
        """Save anti-spoofing mechanism state."""
        state = {
            'verifier_state': self.verifier.state_dict(),
            'spoofing_detector_state': self.spoofing_detector.state_dict(),
            'correction_network_state': self.correction_network.state_dict(),
            'detection_history': self.detection_history,
            'correction_history': self.correction_history,
            'reputation_system': self.reputation_system.reputations.tolist() if self.reputation_system else None,
            'config': {
                'observation_dim': self.observation_dim,
                'detection_threshold': self.detection_threshold,
                'correction_strength': self.correction_strength
            }
        }
        
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """Load anti-spoofing mechanism state."""
        state = torch.load(path, map_location=self.device)
        
        self.verifier.load_state_dict(state['verifier_state'])
        self.spoofing_detector.load_state_dict(state['spoofing_detector_state'])
        self.correction_network.load_state_dict(state['correction_network_state'])
        
        self.detection_history = state['detection_history']
        self.correction_history = state['correction_history']
        
        if state['reputation_system'] is not None and self.reputation_system is not None:
            self.reputation_system.reputations = np.array(state['reputation_system'])
        
        self.detection_threshold = state['config']['detection_threshold']
        self.correction_strength = state['config']['correction_strength']