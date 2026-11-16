"""agent_qa - an agent that tests another agent.

Pipeline per scenario:  Tester agent  <->  Agent under test  ->  Deterministic checks
                                                             ->  LLM judge
                                                             ->  Langfuse trace + scores
                                                             ->  Risk gate (CI exit code)
"""

__version__ = "0.1.0"
