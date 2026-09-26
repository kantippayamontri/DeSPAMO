import json
import traceback

import pytest
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError
from requests import Response
from requests.exceptions import Timeout

from tools.dinov3.identity import digest, identity, read_token, resolve_sha


def test_token_file_is_data_and_never_executed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=ignored\nHF_TOKEN='hf_test; touch /tmp/not-executed'\n")
    assert read_token(env) == "hf_test; touch /tmp/not-executed"
    env.write_text("HF_TOKEN=one\nHF_TOKEN=two\n")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        read_token(env)


def test_revision_pin_and_canonical_version(tmp_path, monkeypatch):
    class Info:
        sha = "a" * 40

    class Api:
        def model_info(self, repo_id, revision, token):
            assert revision == "main" and token == "secret"
            return Info()

    monkeypatch.setattr("tools.dinov3.identity.HfApi", lambda: Api())
    assert resolve_sha("secret", "main") == "a" * 40
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"fixed-lock")
    key, meta = identity("a" * 40, lock)
    assert key == digest(meta) and meta["model_sha"] == "a" * 40
    assert meta["scales"] == [224, 448]
    assert meta["inference_batch_size"] == 8
    assert meta["cudnn_tf32"] is False
    assert meta["source_resolution_policy"] == "record-per-frame"
    assert "secret" not in json.dumps(meta)
    with pytest.raises(ValueError, match="SHA"):
        identity("main", lock)


def test_identity_has_no_fixed_source_size_and_stable_digest(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"fixed-lock")
    key, meta = identity("a" * 40, lock)
    repeated_key, repeated_meta = identity("a" * 40, lock)
    assert "source_resolution" not in meta
    assert [210, 260] not in meta.values()
    assert meta["source_resolution_policy"] == "record-per-frame"
    assert key == repeated_key == digest(meta)
    assert meta == repeated_meta


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (401, HfHubHTTPError, "access denied"),
        (403, HfHubHTTPError, "access denied"),
        (400, HfHubHTTPError, "Hub request rejected"),
        (404, RevisionNotFoundError, "Hub request rejected"),
        (500, HfHubHTTPError, "Hub unavailable; retry"),
        (503, HfHubHTTPError, "Hub unavailable; retry"),
    ],
)
def test_resolve_sha_distinguishes_authorization_from_other_http_errors(
    monkeypatch, status, error_type, message
):
    token = "hf_private_test_value"
    url = "https://invalid.example/models?token=hf_private_test_value"
    response = Response()
    response.status_code = status
    response.url = url

    class Api:
        def model_info(self, repo_id, revision, token):
            raise error_type(f"server returned {url}", response=response)

    monkeypatch.setattr("tools.dinov3.identity.HfApi", lambda: Api())
    with pytest.raises(RuntimeError, match=message) as caught:
        resolve_sha(token, "main")
    rendered = "".join(traceback.format_exception(caught.value))
    assert (status == 404) == ("revision" in str(caught.value).lower())
    assert token not in str(caught.value) and token not in rendered
    assert url not in str(caught.value) and url not in rendered
    assert "HfHubHTTPError" not in rendered and "RevisionNotFoundError" not in rendered


def test_resolve_sha_timeout_does_not_claim_license_gate_or_leak_details(monkeypatch):
    token = "hf_private_test_value"
    url = "https://invalid.example/models?token=hf_private_test_value"

    class Api:
        def model_info(self, repo_id, revision, token):
            raise Timeout(f"timed out at {url}")

    monkeypatch.setattr("tools.dinov3.identity.HfApi", lambda: Api())
    with pytest.raises(RuntimeError, match="Hub request failed") as caught:
        resolve_sha(token, "main")
    rendered = "".join(traceback.format_exception(caught.value))
    assert "license" not in str(caught.value)
    assert token not in rendered and url not in rendered
    assert "Timeout" not in rendered
