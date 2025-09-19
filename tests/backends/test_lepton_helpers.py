from types import SimpleNamespace

from leptonai.api.v1.types.deployment import EnvVar

from domyn_swarm.helpers.lepton import (
    get_env_var_by_name,
    sanitize_tokens_in_deployment,
)

# ------------------------------
# get_env_var_by_name
# ------------------------------


def test_get_env_var_by_name_returns_value_when_present():
    envs = [
        EnvVar(name="FOO", value="bar"),
        EnvVar(name="API_TOKEN_SECRET_NAME", value="my-secret"),
    ]
    assert get_env_var_by_name(envs, "API_TOKEN_SECRET_NAME") == "my-secret"


def test_get_env_var_by_name_returns_none_when_absent():
    envs = [EnvVar(name="FOO", value="bar")]
    assert get_env_var_by_name(envs, "MISSING") is None


def test_get_env_var_by_name_returns_none_when_value_is_none():
    envs = [EnvVar(name="API_TOKEN_SECRET_NAME", value=None)]
    assert get_env_var_by_name(envs, "API_TOKEN_SECRET_NAME") is None


def test_get_env_var_by_name_picks_first_match_if_duplicates():
    envs = [
        EnvVar(name="X", value="first"),
        EnvVar(name="X", value="second"),
    ]
    assert get_env_var_by_name(envs, "X") == "first"


def test_get_env_var_by_name_empty_list():
    assert get_env_var_by_name([], "ANY") is None


# ------------------------------
# sanitize_tokens_in_deployment
# ------------------------------


def test_sanitize_tokens_in_deployment_on_dict_input_redacts_all_values():
    dep_dict = {
        "metadata": {"name": "ep1"},
        "spec": {
            "envs": [],
            "api_tokens": [
                {"value": "s1"},
                {"value": "s2"},
            ],
        },
        "status": None,
    }

    out = sanitize_tokens_in_deployment(dep_dict)
    # `out` is a LeptonDeployment model (via model_validate), but we only need to
    # verify the token values are redacted.
    tokens = out.spec.api_tokens if (out.spec and out.spec.api_tokens) else []
    assert len(tokens) == 2
    assert all(getattr(tok, "value", None) == "REDACTED" for tok in tokens)

    # idempotency: running again keeps them redacted
    out2 = sanitize_tokens_in_deployment(out)
    tokens2 = out2.spec.api_tokens if (out2.spec and out2.spec.api_tokens) else []
    assert all(getattr(tok, "value", None) == "REDACTED" for tok in tokens2)


def test_sanitize_tokens_in_deployment_when_no_tokens_present_is_noop():
    dep_dict = {
        "metadata": {"name": "ep2"},
        "spec": {
            # no api_tokens field
            "envs": [{"name": "FOO", "value": "BAR"}],
        },
    }
    out = sanitize_tokens_in_deployment(dep_dict)
    # Should not raise; and api_tokens remains absent/None
    assert out.spec is not None
    assert getattr(out.spec, "api_tokens", None) in (None, [])


def test_sanitize_tokens_in_deployment_on_object_like_input():
    # Build a minimal object with the same attribute shape (.spec.api_tokens[*].value)
    class _Tok:
        def __init__(self, v):
            self.value = v

    class _Spec:
        def __init__(self):
            self.api_tokens = [_Tok("secret")]

    obj = SimpleNamespace(spec=_Spec())

    out = sanitize_tokens_in_deployment(obj)
    assert out.spec.api_tokens[0].value == "REDACTED"
