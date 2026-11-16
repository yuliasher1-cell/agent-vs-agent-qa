import pytest

from agent_qa.runner import RunConfig, run
from agent_qa.scenarios import load_scenarios
from agent_qa.tracing import NoopTracer


@pytest.fixture(scope="session")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="session")
def clean_run(scenarios):
    return run(scenarios, RunConfig(agent="clean", min_pass_rate=1.0), NoopTracer())


@pytest.fixture(scope="session")
def buggy_run(scenarios):
    return run(scenarios, RunConfig(agent="buggy", min_pass_rate=1.0), NoopTracer())
