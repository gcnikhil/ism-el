"""Phase-13 temperature workload and information-barrier tests."""

from __future__ import annotations

from dataclasses import fields

from config import load_config
from policies import AccessEvent, PolicyView, TemperaturePolicy
from simulator import run_simulation
from workload import generate_workload


def test_true_temp_is_not_policy_observable() -> None:
    assert "true_temp" not in {field.name for field in fields(AccessEvent)}
    assert "true_temp" not in PolicyView.__dict__
    assert "true_temp" not in TemperaturePolicy.__init__.__annotations__


def test_temperature_workloads_are_seeded_and_rotating() -> None:
    config = load_config()
    e = generate_workload(config, "E", 0, epochs=3)
    f = generate_workload(config, "F", 0, epochs=3)
    assert {value for value in e.temperature_by_epoch.values()} == {"hot", "warm", "cold"}
    epoch_zero = {key[1]: value for key, value in f.temperature_by_epoch.items() if key[0] == 0}
    epoch_two = {key[1]: value for key, value in f.temperature_by_epoch.items() if key[0] == 2}
    assert epoch_zero != epoch_two


def test_temperature_run_logs_eval_only_temperatures() -> None:
    result = run_simulation(load_config(), "temperature", "E", 0, epochs=1)
    trace = result.logger.trace_frame()
    assert "true_temp" in trace.columns
    assert {"hot", "warm", "cold"}.issubset(set(trace["true_temp"]))
