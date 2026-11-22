"""Information-barrier tests: policies cannot see truth or the oracle."""

from __future__ import annotations

from dataclasses import fields

import pytest

from policies import AccessEvent, Oracle, Policy, PolicyView


def test_access_event_has_no_true_type_or_name() -> None:
    names = {field.name for field in fields(AccessEvent)}
    assert names == {
        "file_id",
        "size_bytes",
        "op",
        "epoch",
        "step",
        "phase",
    }
    assert "true_type" not in names
    assert "name" not in names
    assert "path" not in names


def test_access_event_rejects_true_type_keyword() -> None:
    with pytest.raises(TypeError):
        AccessEvent(  # type: ignore[call-arg]
            file_id=1,
            size_bytes=2,
            op="read",
            epoch=0,
            step=0,
            phase="train",
            true_type="checkpoint",
        )


def test_policy_protocols_expose_no_truth_channel() -> None:
    for protocol in (PolicyView, Policy, Oracle):
        annotations = getattr(protocol, "__annotations__", {})
        assert "true_type" not in annotations
        assert "display_name" not in annotations
        assert not hasattr(protocol, "true_type")
