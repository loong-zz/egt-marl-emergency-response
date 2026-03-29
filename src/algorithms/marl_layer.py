# Get target joint Q-value
            target_joint_q = self.target_mixing_network(next_qs, next_states)
            
            # Compute target
            target = rewards + self.gamma * target_joint_q.squeeze() * (1 - dones.float())
        
        # Compute loss
        loss = loss_fn(joint_q.squeeze(), target)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Update target networks
        self._update_target_networks(self.tau)
        
        return loss.item()
    
    def _get_batch_agent_observations(self, batch_states: torch.Tensor, 
                                     agent_id: int) -> torch.Tensor:
        """Get batch of observations for a specific agent."""
        # This is a simplified version - in practice would need proper state processing
        batch_size = batch_states.shape[0]
        
        # For simplicity, assume states are already agent observations
        # In real implementation, would extract agent-specific features
        return batch_states
    
    def _actions_to_indices(self, batch_actions: torch.Tensor, 
                           agent_id: int) -> torch.Tensor:
        """Convert batch of actions to action indices for a specific agent."""
        batch_size = batch_actions.shape[0]
        
        # Simplified: assume actions are already indices
        # In real implementation, would need to convert from action dictionaries
        return batch_actions[:, agent_id].long()
    
    def get_communication_matrix(self) -> torch.Tensor:
        """Get communication matrix showing agent interactions."""
        if not self.communication_enabled or self.communication_network is None:
            return torch.zeros((self.num_agents, self.num_agents), device=self.device)
        
        # Create dummy input to extract communication patterns
        dummy_state = torch.randn(1, self.state_dim, device=self.device)
        
        with torch.no_grad():
            # Pass through communication network
            output = self.communication_network(dummy_state)
            
            # Reshape to get agent-wise communication
            # This is simplified - actual implementation would depend on network architecture
            comm_matrix = output.view(self.num_agents, -1)
        
        return comm_matrix
    
    def save(self, path: str) -> None:
        """Save MARL layer state."""
        torch.save({
            'agent_networks_state': [net.state_dict() for net in self.agent_networks],
            'target_agent_networks_state': [net.state_dict() for net in self.target_agent_networks],
            'mixing_network_state': self.mixing_network.state_dict(),
            'target_mixing_network_state': self.target_mixing_network.state_dict(),
            'communication_network_state': self.communication_network.state_dict() if self.communication_enabled else None,
            'epsilon': self.epsilon,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_agents': self.num_agents,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load MARL layer state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load agent networks
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(checkpoint['agent_networks_state'][i])
        
        # Load target networks
        for i, net in enumerate(self.target_agent_networks):
            net.load_state_dict(checkpoint['target_agent_networks_state'][i])
        
        # Load mixing networks
        self.mixing_network.load_state_dict(checkpoint['mixing_network_state'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network_state'])
        
        # Load communication network
        if self.communication_enabled and checkpoint['communication_network_state'] is not None:
            self.communication_network.load_state_dict(checkpoint['communication_network_state'])
        
        # Load epsilon
        self.epsilon = checkpoint['epsilon']