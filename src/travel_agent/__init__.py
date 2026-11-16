"""TripMate - the travel booking agent under test.

This package is the *system under test*. It knows nothing about the QA framework in
`agent_qa`; the only contract between them is `TravelAgent.respond()`.
"""

from travel_agent.agent import TravelAgent, build_agent

__all__ = ["TravelAgent", "build_agent"]
