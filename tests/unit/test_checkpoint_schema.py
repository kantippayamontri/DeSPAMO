import pytest
import torch

from despamo.checkpoints import schema_from_model, validate_converted_state_dict


def test_schema_derives_keys_shapes_and_dtypes_from_model_state_dict() -> None:
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16)
    model.register_buffer("steps", torch.tensor(0, dtype=torch.int64))

    assert schema_from_model(model) == {
        "proj.weight": {"shape": [2, 3], "dtype": "torch.float16"},
        "steps": {"shape": [], "dtype": "torch.int64"},
    }


def test_schema_validation_accepts_exact_model_state() -> None:
    model = torch.nn.Linear(2, 3)
    validate_converted_state_dict(model.state_dict(), schema_from_model(model))


def test_schema_validation_reports_missing_extra_shape_and_dtype_with_examples() -> None:
    schema = {
        "missing.weight": {"shape": [1], "dtype": "torch.float32"},
        "wrong_shape": {"shape": [2, 3], "dtype": "torch.float32"},
        "wrong_dtype": {"shape": [2], "dtype": "torch.float16"},
    }
    converted = {
        "extra.bias": torch.ones(1),
        "wrong_shape": torch.ones(3, 2),
        "wrong_dtype": torch.ones(2, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="checkpoint schema mismatch") as error:
        validate_converted_state_dict(converted, schema)

    message = str(error.value)
    for detail in (
        "missing.weight",
        "extra.bias",
        "wrong_shape",
        "[2, 3]",
        "[3, 2]",
        "wrong_dtype",
        "torch.float16",
        "torch.float32",
    ):
        assert detail in message


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"weight": {"shape": [2]}},
        {"weight": {"shape": [-1], "dtype": "torch.float32"}},
        {"weight": {"shape": [True], "dtype": "torch.float32"}},
    ],
)
def test_schema_validation_rejects_malformed_or_empty_schema(schema: dict) -> None:
    with pytest.raises(ValueError, match="invalid target schema"):
        validate_converted_state_dict({"weight": torch.ones(2)}, schema)
