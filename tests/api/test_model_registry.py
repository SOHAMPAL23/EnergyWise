"""
tests/api/test_model_registry.py
────────────────────────────────
Unit tests for the Model Registry and artifact verification.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from api.core.exceptions import ArtifactCorruptionError, ModelNotAvailableError
from api.services.model_registry import (
    ModelRegistry,
    _load_manifest,
    _sha256,
    _verify_checksum,
)


class TestModelRegistryHelpers:
    def test_sha256_computation(self, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world", encoding="utf-8")
        digest = _sha256(f)
        import hashlib
        assert digest == hashlib.sha256(b"hello world").hexdigest()

    def test_load_manifest_success(self, tmp_path: Path):
        mf = tmp_path / "manifest.json"
        mf.write_text('{"summary": {"test": 1}}', encoding="utf-8")
        data = _load_manifest(mf)
        assert data["summary"]["test"] == 1

    def test_load_manifest_nonexistent(self, tmp_path: Path):
        mf = tmp_path / "nonexistent.json"
        with pytest.raises(ModelNotAvailableError, match="Artifact manifest not found"):
            _load_manifest(mf)

    def test_load_manifest_corrupted_json(self, tmp_path: Path):
        mf = tmp_path / "bad.json"
        mf.write_text("{this is not json", encoding="utf-8")
        with pytest.raises(ModelNotAvailableError, match="Manifest is not valid JSON"):
            _load_manifest(mf)

    def test_verify_checksum_success(self, tmp_path: Path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        model_file = art_dir / "model.txt"
        model_file.write_text("model weights", encoding="utf-8")

        import hashlib
        sha = hashlib.sha256(b"model weights").hexdigest()
        entry = {"path": "model.txt", "sha256": sha}

        verified = _verify_checksum(art_dir, entry)
        assert verified == model_file

    def test_verify_checksum_missing_file(self, tmp_path: Path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        entry = {"path": "missing.txt", "sha256": "abc123"}

        with pytest.raises(ModelNotAvailableError, match="Model artifact missing on disk"):
            _verify_checksum(art_dir, entry)

    def test_verify_checksum_mismatch(self, tmp_path: Path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        model_file = art_dir / "model.txt"
        model_file.write_text("model weights", encoding="utf-8")

        entry = {"path": "model.txt", "sha256": "wrong_sha"}
        with pytest.raises(ArtifactCorruptionError, match="SHA-256 mismatch"):
            _verify_checksum(art_dir, entry)


class TestModelRegistryClass:
    def test_empty_registry_not_available(self):
        reg = ModelRegistry()
        assert reg.is_available("daily") is False
        assert reg.is_available("weekly") is False
        assert reg.manifest_summary() is None
        assert reg.manifest_generated_at() is None

    def test_get_unloaded_model_raises_503(self):
        reg = ModelRegistry()
        with pytest.raises(ModelNotAvailableError, match="Model for granularity 'daily' is not available"):
            reg.get_model("daily")

    def test_get_unloaded_model_info_raises_503(self):
        reg = ModelRegistry()
        with pytest.raises(ModelNotAvailableError, match="Metadata for granularity 'daily' is not available"):
            reg.get_model_info("daily")

