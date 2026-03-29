elif selected_severity == CasualtySeverity.MODERATE:
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 1.0,
                    }
                else:  # MILD
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 0.5,
                    }
                
                casualty = Casualty(
                    id=casualty_id,
                    position=position,
                    severity=selected_severity,
                    injury_time=0.0,
                    resources_needed=resources,
                )
                
                area.casualties.append(casualty)
                self.casualties[casualty_id] = casualty
                casualty_id += 1
    
    def _initialize_road_network(self) -> None:
        """Initialize road network connecting affected areas and depots."""
        # Add nodes for all important locations
        all_positions = []
        node_ids = []
        
        # Add affected areas
        for area_id, area in self.affected_areas.items():
            self.road_network.add_node(f"area_{area_id}", pos=tuple(area.position))
            all_positions.append(area.position)
            node_ids.append(f"area_{area_id}")
        
        # Add resource depots
        for depot_id, depot in self.resource_depots.items():
            self.road_network.add_node(f"depot_{depot_id}", pos=tuple(depot.position))
            all_positions.append(depot.position)
            node_ids.append(f"depot_{depot_id}")
        
        # Connect nodes based on proximity (Delaunay triangulation would be better)
        positions_array = np.array(all_positions)
        distances = cdist(positions_array, positions_array)
        
        # Connect each node to its 3 nearest neighbors
        for i, node_i in enumerate(node_ids):
            # Get indices of nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:4]  # 3 nearest
            
            for j in neighbor_indices:
                node_j = node_ids[j]
                distance = distances[i, j]
                
                # Add edge with weight = distance
                self.road_network.add_edge(node_i, node_j, weight=distance)
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space per agent
        # Features: position(2), velocity(2), capacity(4), endurance(1), 
        # mission_status(1), nearest_area_info(8), global_resource_levels(4)
        obs_dim = 2 + 2 + 4 + 1 + 1 + 8 + 4  # Total: 22
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: hierarchical
        # Strategic: resource allocation [0,1]^4 (continuous)
        # Tactical: movement direction (8 discrete)
        # Communication: information sharing (4 discrete)
        self.action_space = spaces.Dict({
            "strategic": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "tactical": spaces.Discrete(8),
            "communication": spaces.Discrete(4),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation state
        self.current_time = 0.0
        self.step_count = 0
        
        # Reinitialize components
        self._initialize_affected_areas()
        self._initialize_resource_depots()
        self._initialize_rescue_agents()
        self._initialize_casualties()
        self._initialize_road_network()
        
        # Reset statistics
        self.statistics = {
            "total_survivors": 0,
            "total_casualties": len(self.casualties),
            "resource_utilization": {rt: 0.0 for rt in ResourceType},
            "response_times": [],
            "fairness_metrics": {"gini": [], "theil": [], "max_min": []},
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, actions: Dict[int, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            observation: New observation
            reward: Total reward
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Apply actions to agents
        for agent_id, action in actions.items():
            if agent_id in self.rescue_agents:
                self._apply_action(agent_id, action)
        
        # Update simulation state
        self._update_dynamics()
        
        # Update casualties
        self._update_casualties()
        
        # Update communication network
        self._update_communication()
        
        # Apply secondary disasters
        if np.random.rand() < self._get_secondary_disaster_probability():
            self._apply_secondary_disaster()
        
        # Update weather
        self._update_weather()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update statistics
        self._update_statistics()
        
        # Increment time and step count
        self.current_time += self.time_step
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, agent_id: int, action: Dict[str, Any]) -> None:
        """Apply action to a specific agent."""
        agent = self.rescue_agents[agent_id]
        
        # Strategic action: resource allocation
        if "strategic" in action:
            allocation = action["strategic"]
            # Normalize allocation to sum to 1
            allocation = allocation / (np.sum(allocation) + 1e-8)
            
            # Update agent's resource allocation strategy
            # This affects which resources the agent prioritizes
            pass  # Implementation depends on specific strategy
        
        # Tactical action: movement
        if "tactical" in action:
            direction_idx = action["tactical"]
            # Convert direction index to vector (8 directions)
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])
            
            # Set target position
            max_speed = agent.get_max_speed()
            target_distance = max_speed * self.time_step
            target_position = agent.position + direction * target_distance
            
            # Clip to map boundaries
            target_position = np.clip(target_position, 0, self.map_size)
            
            # Plan route to target (simple straight line for now)
            agent.route = [target_position]
        
        # Communication action: information sharing
        if "communication" in action:
            comm_action = action["communication"]
            # Implement communication strategy
            pass  # Implementation depends on communication protocol
        
        # Move agent
        agent.move(self.time_step)
    
    def _update_dynamics(self) -> None:
        """Update dynamic factors in the environment."""
        # Update agent endurance and refuel/resupply if at depot
        for agent in self.rescue_agents.values():
            # Check if agent is at a depot
            at_depot = False
            for depot in self.resource_depots.values():
                distance = np.linalg.norm(agent.position - depot.position)
                if distance < 100.0:  # Within 100m of depot
                    at_depot = True
                    
                    # Refuel (restore endurance)
                    agent.endurance = min(
                        agent.endurance + self.time_step * 2,  # Faster recovery at depot
                        agent.max_endurance
                    )
                    
                    # Resupply resources
                    for resource_type in ResourceType:
                        if agent.capacity[resource_type] < agent.max_capacity[resource_type]:
                            # Try to get resources from depot
                            needed = agent.max_capacity[resource_type] - agent.capacity[resource_type]
                            available = depot.resources.get(resource_type, 0.0)
                            transfer = min(needed, available)
                            
                            if transfer > 0:
                                agent.capacity[resource_type] += transfer
                                depot.resources[resource_type] -= transfer
                    break
            
            # If not at depot, endurance decreases normally
            if not at_depot:
                agent.endurance = max(agent.endurance - self.time_step, 0.0)
    
    def _update_casualties(self) -> None:
        """Update casualty states and check for deaths."""
        casualties_to_remove = []
        
        for casualty_id, casualty in self.casualties.items():
            # Update survival probability
            casualty.update_survival_probability(self.current_time)
            
            # Check if casualty dies
            if not casualty.is_alive(self.current_time):
                casualties_to_remove.append(casualty_id)
                continue
            
            # Check if casualty is being treated
            if casualty.treated and casualty.treatment_start is not None:
                treatment_duration = self.current_time - casualty.treatment_start
                
                # Check if treatment is complete
                required_time = {
                    CasualtySeverity.CRITICAL: 3600,  # 1 hour
                    CasualtySeverity.SEVERE: 1800,    # 30 minutes
                    CasualtySeverity.MODERATE: 900,   # 15 minutes
                    CasualtySeverity.MILD: 300,       # 5 minutes
                }[casualty.severity]
                
                if treatment_duration >= required_time:
                    # Treatment complete, casualty survives
                    self.statistics["total_survivors"] += 1
        
        # Remove dead casualties
        for casualty_id in casualties_to_remove:
            casualty = self.casualties.pop(casualty_id)
            # Remove from affected area
            for area in self.affected_areas.values():
                if casualty in area.casualties:
                    area.casualties.remove(casualty)
                    break
    
    def _update_communication(self) -> None:
        """Update communication network between agents."""
        # Reset connections
        for agent in self.rescue_agents.values():
            agent.connected_agents = []
        
        # Establish new connections based on distance and communication status
        agent_ids = list(self.rescue_agents.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i = self.rescue_agents[agent_ids[i]]
                agent_j = self.rescue_agents[agent_ids[j]]
                
                # Check if agents can communicate
                if agent_i.can_communicate(agent_j.position):
                    # Apply communication status (weather, interference)
                    if np.random.rand() < self.communication_status:
                        agent_i.connected_agents.append(agent_j.id)
                        agent_j.connected_agents.append(agent_i.id)
    
    def _get_secondary_disaster_probability(self) -> float:
        """Calculate probability of secondary disaster."""
        # K/(t+c) model
        K = 10.0
        c = 1.0
        probability = K / (self.current_time / 3600.0 + c)  # time in hours
        return min(probability / 100.0, 0.1)  # Normalize to reasonable probability
    
    def _apply_secondary_disaster(self) -> None:
        """Apply a secondary disaster event."""
        self.secondary_disaster_counter += 1
        
        # Randomly select affected area
        area_id = np.random.choice(list(self.affected_areas.keys()))
        area = self.affected_areas[area_id]
        
        # Apply effects
        effect_type = np.random.choice(["additional_casualties", "road_damage", "building_collapse"])
        
        if effect_type == "additional_casualties":
            # Add new casualties
            additional_rate = np.random.uniform(0.05, 0.10)
            num_additional = int(area.population * additional_rate)
            
            # Similar to initial casualty creation
            for _ in range(num_additional):
                # Create new casualty
                pass  # Implementation similar to _initialize_casualties
        
        elif effect_type == "road_damage":
            # Damage roads in the area
            area.road_accessibility *= np.random.uniform(0.5, 0.8)
            
            # Update road network weights
            for u, v, data in self.road_network.edges(data=True):
                if f"area_{area_id}" in (u, v):
                    # Increase travel time on damaged roads
                    data["weight"] *= 1.5
        
        else:  # building_collapse
            area.building_damage = min(area.building_damage + 0.2, 1.0)
    
    def _update_weather(self) -> None:
        """Update weather conditions based on time."""
        hours = self.current_time / 3600.0
        
        if hours < 12:
            self.weather_conditions = "clear"
            self.communication_status = 1.0
        elif hours < 36:
            self.weather_conditions = "light_rain"
            self.communication_status = 0.8
        elif hours < 60:
            self.weather_conditions = "heavy_rain"
            self.communication_status = 0.5
        else:
            self.weather_conditions = "clearing"
            self.communication_status = 0.7
    
    def _calculate_reward(self) -> float:
        """Calculate total reward for the current step."""
        reward = 0.0
        
        # Individual efficiency: casualties treated
        for agent in self.rescue_agents.values():
            if agent.current_mission is not None:
                area = self.affected_areas.get(agent.current_mission)
                if area:
                    # Reward for treating casualties
                    treated_count = sum(1 for c in area.casualties if c.treated)
                    reward += 0.1 * treated_count
        
        # Global efficiency: total survivors
        reward += 0.05 * self.statistics["total_survivors"]
        
        # Cooperation bonus: connected agents
        total_connections = sum(len(agent.connected_agents) for agent in self.rescue_agents.values())
        reward += 0.01 * total_connections
        
        # Fairness penalty: based on Gini coefficient
        survival_rates = [area.survival_rate for area in self.affected_areas.values()]
        if survival_rates:
            gini = self._calculate_gini(survival_rates)
            reward -= 0.2 * gini  # Penalize inequality
        
        return reward
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values."""
        if not values:
            return 0.0
        
        values = np.array(values)
        n = len(values)
        abs_diffs = np.abs(values[:, None] - values[None, :])
        gini = np.sum(abs_diffs) / (2 * n * np.sum(values))
        return float(gini)
    
    def _update_statistics(self) -> None:
        """Update simulation statistics."""
        # Calculate fairness metrics
        survival_rates = [area.survival_rate for area in self.affected_areas.values()]
        
        if survival_rates:
            # Gini coefficient
            gini = self._calculate_gini(survival_rates)
            self.statistics["fairness_metrics"]["gini"].append(gini)
            
            # Theil index
            mean_rate = np.mean(survival_rates)
            if mean_rate > 0:
                theil = np.mean([(r/mean_rate) * np.log(r/mean_rate) for r in survival_rates if r > 0])
                self.statistics["fairness_metrics"]["theil"].append(theil)
            
            # Max-min fairness
            max_min = np.min(survival_rates) if survival_rates else 0.0
            self.statistics["fairness_metrics"]["max_min"].append(max_min)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if all casualties are either treated or dead
        active_casualties = sum(1 for c in self.casualties.values() if not c.treated)
        return active_casualties == 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for all agents."""
        observations = []
        
        for agent in self.rescue_agents.values():
            obs = self._get_agent_observation(agent)
            observations.append(obs)
        
        return np.array(observations)
    
    def _get_agent_observation(self, agent: RescueAgent) -> np.ndarray:
        """Get observation for a specific agent."""
        obs = []
        
        # Agent state (9 features)
        obs.extend(agent.position / self.map_size)  # Normalized position
        obs.extend(agent.velocity / agent.get_max_speed())  # Normalized velocity
        obs.extend([agent.capacity[rt] / agent.max_capacity[rt] for rt in ResourceType])  # Resource levels
        obs.append(agent.endurance / agent.max_endurance)  # Normalized endurance
        obs.append(1.0 if agent.current_mission is not None else 0.0)  # Mission status
        
        # Nearest affected area info (8 features)
        nearest_area = self._get_nearest_affected_area(agent.position)
        if nearest