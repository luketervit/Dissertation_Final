from pathlib import Path

from openrouter_runtime.simulation import (
    OpenRouterThreadSimulation,
    load_simulation_config,
)


class DummyClient:
    model = "dummy/model"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **_: object):
        self.calls += 1

        class Result:
            text = "That argument ignores what voters are actually paying every week."
            provider = "dummy"
            model = "dummy/model"
            latency_seconds = 0.01
            usage = {"prompt_tokens": 1, "completion_tokens": 1}
            raw_id = "dummy-id"

        return Result()


def test_smoke_config_runs_one_round() -> None:
    config = load_simulation_config(
        Path("config/openrouter_smoke_config.yaml")
    )
    client = DummyClient()
    sim = OpenRouterThreadSimulation(config=config, client=client)
    posts = sim.run(rounds=1)
    assert len(posts) >= 2
    assert client.calls >= 1
