"""
TreatmentManager class for disaster simulation.

Handles all casualty treatment logic including priority calculation,
resource checking, and treatment execution.
"""

import logging
from typing import Dict, Optional, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from ..config.constants import (
    CasualtySeverity, ResourceType, SimulationConfig,
    TREATMENT_DURATION, CONSUMPTION_RATE, CONSUMPTION_FACTOR, RESOURCE_ABBR,
    TREATMENT_RANGE
)
from ..entities.agent import RescueAgent
from ..entities.casualty import Casualty

logger = logging.getLogger(__name__)


class TreatmentStrategy(ABC):
    """Abstract base class for treatment priority strategies."""

    @abstractmethod
    def calculate_priority(self, casualty: Casualty, agent: RescueAgent) -> float:
        """Calculate treatment priority for a casualty."""
        pass

    @abstractmethod
    def check_resources(self, agent: RescueAgent, casualty: Casualty) -> bool:
        """Check if agent has sufficient resources to treat casualty."""
        pass


class SeverityBasedStrategy(TreatmentStrategy):
    """Treatment strategy based on casualty severity."""

    def calculate_priority(self, casualty: Casualty, agent: RescueAgent) -> float:
        """Calculate priority based on severity and distance."""
        severity_weights = {
            CasualtySeverity.CRITICAL: 10.0,
            CasualtySeverity.SEVERE: 7.0,
            CasualtySeverity.MODERATE: 4.0,
            CasualtySeverity.MILD: 1.0
        }

        distance = np.linalg.norm(agent.position - casualty.position)
        distance_factor = max(0.1, 1.0 - distance / 100.0)

        base_priority = severity_weights[casualty.severity]
        survival_bonus = casualty.survival_probability

        return base_priority * distance_factor * survival_bonus

    def check_resources(self, agent: RescueAgent, casualty: Casualty) -> bool:
        """Check if agent has enough resources for full treatment."""
        for resource_type, needed in casualty.resources_needed.items():
            consumption_factor = CONSUMPTION_FACTOR[casualty.severity]
            total_needed = needed * (1 + consumption_factor)

            if agent.capacity.get(resource_type, 0.0) < total_needed:
                return False

        return True


class SurvivalProbabilityStrategy(TreatmentStrategy):
    """Treatment strategy based on survival probability."""

    def calculate_priority(self, casualty: Casualty, agent: RescueAgent) -> float:
        """Calculate priority based on survival probability."""
        distance = np.linalg.norm(agent.position - casualty.position)
        distance_factor = max(0.1, 1.0 - distance / 100.0)

        survival_priority = (1.0 - casualty.survival_probability) * 10.0

        return survival_priority * distance_factor

    def check_resources(self, agent: RescueAgent, casualty: Casualty) -> bool:
        """Check if agent has enough resources for full treatment."""
        strategy = SeverityBasedStrategy()
        return strategy.check_resources(agent, casualty)


class TreatmentManager:
    """Manages casualty treatment operations."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.strategy = SeverityBasedStrategy()
        self.total_resources_used = 0.0

    def set_strategy(self, strategy: TreatmentStrategy) -> None:
        """Set the treatment strategy."""
        self.strategy = strategy

    def calculate_treatment_priority(
        self,
        agent: RescueAgent,
        casualties: Dict[int, Casualty]
    ) -> List[Tuple[float, int]]:
        """
        Calculate treatment priorities for all casualties.

        Args:
            agent: Agent considering treatment
            casualties: All casualties in the simulation

        Returns:
            List of (priority, casualty_id) tuples sorted by priority
        """
        priorities = []

        for casualty_id, casualty in casualties.items():
            if not casualty.is_alive(self.config.time_step * 0):
                continue

            if casualty.treated and casualty.treating_agent_id is not None:
                continue

            priority = self.strategy.calculate_priority(casualty, agent)
            priorities.append((priority, casualty_id))

        priorities.sort(key=lambda x: -x[0])

        return priorities

    def can_treat_casualty(self, agent: RescueAgent, casualty: Casualty) -> bool:
        """
        Check if agent can treat a casualty.

        Considers both resource availability and proximity.
        """
        if casualty.treated:
            return False

        if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
            return False

        distance = np.linalg.norm(agent.position - casualty.position)
        if distance > TREATMENT_RANGE:
            return False

        has_resources = self.strategy.check_resources(agent, casualty)
        if not has_resources:
            self._log_resource_shortage(agent, casualty)
        return has_resources

    def _log_resource_shortage(self, agent: RescueAgent, casualty: Casualty) -> None:
        """Log which resources are insufficient for treatment."""
        shortage = []
        for resource_type, needed in casualty.resources_needed.items():
            consumption_factor = CONSUMPTION_FACTOR[casualty.severity]
            total_needed = needed * (1 + consumption_factor)
            current = agent.capacity.get(resource_type, 0.0)
            if current < total_needed:
                abbr = RESOURCE_ABBR.get(resource_type.name, resource_type.name[:4])
                shortage.append(f"{abbr}:{current:.1f}/{total_needed:.1f}")
        logger.debug(
            f"[RESOURCE SHORTAGE] Agent{agent.id} cannot treat Casualty{casualty.id} "
            f"(Severity={casualty.severity.name}) - {', '.join(shortage)}"
        )

    def process_treatment_step(
        self,
        agent: RescueAgent,
        casualty: Casualty,
        current_time: float
    ) -> bool:
        """
        Process one time step of treatment.

        Args:
            agent: Agent performing treatment
            casualty: Casualty being treated
            current_time: Current simulation time

        Returns:
            True if treatment completed this step, False otherwise
        """
        if casualty.treating_agent_id is None:
            casualty.start_treatment(agent.id, current_time)
            logger.info(
                f"[TREATMENT START] Agent{agent.id} treating Casualty{casualty.id} "
                f"(Severity={casualty.severity.name}) at time={current_time:.1f}s"
            )

        duration = TREATMENT_DURATION[casualty.severity]
        elapsed = current_time - casualty.treatment_start

        step_consumption = 0.0
        for resource_type, needed in casualty.resources_needed.items():
            consumption_rate = CONSUMPTION_RATE[casualty.severity]
            consumption = needed * consumption_rate * self.config.time_step

            if agent.capacity.get(resource_type, 0.0) >= consumption:
                agent.capacity[resource_type] -= consumption
                step_consumption += consumption
            else:
                logger.warning(
                    f"[TREATMENT RESOURCE LOW] Agent{agent.id} treating Casualty{casualty.id} "
                    f"Resource={RESOURCE_ABBR.get(resource_type.name, resource_type.name[:4])}"
                )
                casualty.stop_treatment()
                return False

        self.total_resources_used += step_consumption

        if elapsed >= duration:
            self.complete_treatment(agent, casualty, current_time)
            return True

        return False

    def complete_treatment(
        self,
        agent: RescueAgent,
        casualty: Casualty,
        current_time: float
    ) -> None:
        """
        Complete treatment for a casualty.

        Args:
            agent: Agent that completed treatment
            casualty: Casualty that was treated
            current_time: Current simulation time
        """
        casualty.treated = True
        casualty.stop_treatment()
        agent.rescued_count += 1

        # Remove from agent's known casualties list
        agent.remove_known_casualty(casualty.id)

        response_time = 0.0
        if casualty.treatment_start is not None and casualty.injury_time is not None:
            response_time = casualty.treatment_start - casualty.injury_time

        resources_used = {}
        for rt in ResourceType:
            resources_used[rt] = agent.max_capacity[rt] - agent.capacity.get(rt, 0.0)

        logger.info(
            f"[TREATMENT COMPLETE] Agent{agent.id} rescued Casualty{casualty.id} "
            f"(Severity={casualty.severity.name}) | "
            f"ResponseTime={response_time:.1f}s | "
            f"Used={self._format_resources(resources_used)} | "
            f"Remaining={self._format_resources(agent.capacity)}"
        )

    def _format_resources(self, capacity: Dict[ResourceType, float]) -> str:
        """Format resource capacity for logging."""
        return ", ".join(
            f"{RESOURCE_ABBR.get(rt.name, rt.name[:4])}:{v:.2f}"
            for rt, v in capacity.items()
        )

    def find_best_treatment_target(
        self,
        agent: RescueAgent,
        casualties: Dict[int, Casualty]
    ) -> Optional[int]:
        """
        Find the best casualty for the agent to treat.

        Args:
            agent: Agent looking for treatment target
            casualties: All casualties

        Returns:
            ID of best casualty or None
        """
        priorities = self.calculate_treatment_priority(agent, casualties)

        for _, casualty_id in priorities:
            casualty = casualties[casualty_id]
            if self.can_treat_casualty(agent, casualty):
                return casualty_id

        return None
