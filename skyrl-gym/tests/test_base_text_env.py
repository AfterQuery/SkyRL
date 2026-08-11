"""Tests for skyrl_gym.envs.base_text_env."""

from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


def test_step_output_postprocessed_action_is_optional():
    # postprocessed_action is documented as optional and step() implementations
    # omit it. A "= None" default on a TypedDict field is a no-op that left the
    # key required, so constructing the output without it violated the type.
    assert "postprocessed_action" in BaseTextEnvStepOutput.__optional_keys__
    assert "postprocessed_action" not in BaseTextEnvStepOutput.__required_keys__
    assert BaseTextEnvStepOutput.__required_keys__ == frozenset(
        {"observations", "reward", "done", "metadata"}
    )

    # Constructing without postprocessed_action is valid.
    out = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    assert "postprocessed_action" not in out
