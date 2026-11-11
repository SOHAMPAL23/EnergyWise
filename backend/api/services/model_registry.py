"""
api/services/model_registry.py
────────────────────────────────
Model Registry — the single source of truth for loaded Prophet models.

Responsibilities
────────────────
* Parse `manifest.json` produced by the ML pipeline.
* Verify SHA-256 checksums for every model artifact before first use.
* Lazily load model files from disk (Prophet JSON serialisation).
* Expose a thread-safe registry so multiple workers each maintain their own
  in-process copy (no shared state across processes — intentional).

Design constraints
──────────────────
* NO training logic lives here.
* The registry is populated at application startup (lifespan event) and then
  read-only for the duration of the process.
* Failures during loading raise `ModelNotAvailableError`, which surfaces as
  HTTP 503 to callers.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from api.core.config import get_settings
from api.core.exceptions import ArtifactCorruptionError, ModelNotAvailableError

logger = logging.getLogger(__name__)

# Supported granularities
GRANULARITIES = ("daily", "weekly")


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Read and parse manifest.json."""
    if not manifest_path.exists():
        raise ModelNotAvailableError(
            f"Artifact manifest not found: {manifest_path}"
        )
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelNotAvailableError(
            f"Manifest is not valid JSON: {exc}"
        ) from exc


def _verify_checksum(artifacts_dir: Path, entry: Dict[str, Any]) -> Path:
    """
    Resolve and verify a single artifact entry from the manifest.

    Returns the resolved Path on success; raises ArtifactCorruptionError
    or ModelNotAvailableError on failure.
    """
    rel_path: str = entry["path"]
    expected_sha: str = entry["sha256"]

    full_path = artifacts_dir / rel_path
    if not full_path.exists():
        raise ModelNotAvailableError(
            f"Model artifact missing on disk: {full_path}"
        )

    actual_sha = _sha256(full_path)
    if actual_sha != expected_sha:
        raise ArtifactCorruptionError(
            f"SHA-256 mismatch for {rel_path}: "
            f"expected={expected_sha[:12]}…  actual={actual_sha[:12]}…"
        )

    return full_path


class ModelRegistry:
    """
    In-process registry for Prophet model artifacts.

    Lifecycle
    ---------
    1. Instantiate once at startup (see `build_registry()`).
    2. Call `is_available(granularity)` to check availability.
    3. Call `get_model(granularity)` to retrieve a loaded Prophet model.
    4. Call `get_model_info(granularity)` to retrieve metadata dict.
    """

    def __init__(self) -> None:
        self._manifest: Optional[Dict[str, Any]] = None
        self._models: Dict[str, Any] = {}          # granularity → Prophet model
        self._metadata: Dict[str, Any] = {}        # granularity → meta dict
        self._available: Dict[str, bool] = {g: False for g in GRANULARITIES}
        self._load_errors: Dict[str, str] = {}

    # ─── Public API ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load all model artifacts from disk.

        Intended to be called exactly once, inside the FastAPI lifespan
        startup handler. Partial failures are recorded in `_load_errors` so
        the health endpoint can surface them without crashing the service.
        """
        settings = get_settings()
        artifacts_dir: Path = settings.ARTIFACTS_DIR
        manifest_path: Path = settings.MANIFEST_PATH

        logger.info("Loading artifact manifest from %s", manifest_path)
        try:
            self._manifest = _load_manifest(manifest_path)
        except ModelNotAvailableError as exc:
            logger.error("Failed to load manifest: %s", exc)
            for g in GRANULARITIES:
                self._load_errors[g] = str(exc)
            return

        for granularity in GRANULARITIES:
            self._load_one(granularity, artifacts_dir)

    def is_available(self, granularity: str) -> bool:
        """Return True if the model for `granularity` loaded successfully."""
        return self._available.get(granularity, False)

    def get_model(self, granularity: str):
        """
        Return the loaded Prophet model for the given granularity.

        Raises ModelNotAvailableError if the model is not loaded.
        """
        if not self.is_available(granularity):
            err = self._load_errors.get(granularity, "Unknown load error.")
            raise ModelNotAvailableError(
                f"Model for granularity '{granularity}' is not available: {err}"
            )
        return self._models[granularity]

    def get_model_info(self, granularity: str) -> Dict[str, Any]:
        """Return the metadata dict for a granularity (raises if not loaded)."""
        if granularity not in self._metadata:
            raise ModelNotAvailableError(
                f"Metadata for granularity '{granularity}' is not available."
            )
        return self._metadata[granularity]

    def manifest_summary(self) -> Optional[Dict[str, Any]]:
        """Return the top-level `summary` block from the manifest, or None."""
        if self._manifest is None:
            return None
        return self._manifest.get("summary")

    def manifest_generated_at(self) -> Optional[str]:
        """ISO timestamp when the manifest was generated, or None."""
        if self._manifest is None:
            return None
        return self._manifest.get("generated_at")

    def load_error(self, granularity: str) -> Optional[str]:
        """Return the load error message for a granularity, or None."""
        return self._load_errors.get(granularity)

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _load_one(self, granularity: str, artifacts_dir: Path) -> None:
        """Load and verify a single model. Stores result or records error."""
        settings = get_settings()
        version = (
            settings.MODEL_VERSION_DAILY
            if granularity == "daily"
            else settings.MODEL_VERSION_WEEKLY
        )
        model_id = f"prophet-{granularity}-{version}"
        model_rel = f"models/{model_id}.json"
        meta_rel = f"models/{model_id}.meta.json"

        logger.info("Loading model %s …", model_id)
        try:
            # Locate the artifact entry in the manifest to get its checksum
            model_entry = self._find_artifact_entry(model_rel)
            meta_entry = self._find_artifact_entry(meta_rel)

            # Verify checksums
            model_path = _verify_checksum(artifacts_dir, model_entry)
            meta_path = _verify_checksum(artifacts_dir, meta_entry)

            # Deserialise the Prophet model (prophet must be installed)
            try:
                from prophet.serialize import model_from_json  # type: ignore[import]
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "Prophet is not installed. "
                    "Run `pip install prophet` to enable forecasting."
                ) from exc

            model = model_from_json(model_path.read_text(encoding="utf-8"))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

            self._models[granularity] = model
            self._metadata[granularity] = metadata
            self._available[granularity] = True

            logger.info(
                "Loaded model %s | granularity=%s | version=%s",
                model_id,
                granularity,
                version,
            )

        except (ModelNotAvailableError, ArtifactCorruptionError) as exc:
            self._load_errors[granularity] = str(exc)
            self._available[granularity] = False
            logger.error(
                "Failed to load model for granularity '%s': %s",
                granularity,
                exc,
            )

    def _find_artifact_entry(self, rel_path: str) -> Dict[str, Any]:
        """
        Find an artifact entry in the manifest by its relative path.

        Raises ModelNotAvailableError if not found.
        """
        norm_rel = rel_path.replace("\\", "/")
        artifacts = self._manifest.get("artifacts", {})
        for entries in artifacts.values():
            for entry in entries:
                entry_path = entry.get("path", "").replace("\\", "/")
                if entry_path == norm_rel:
                    return entry
        raise ModelNotAvailableError(
            f"Artifact '{rel_path}' not listed in manifest."
        )


# ── Module-level singleton ────────────────────────────────────────────────────
# Populated during the FastAPI lifespan startup event (see main.py).
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Return the global ModelRegistry instance.

    Raises RuntimeError if called before `initialise_registry()`.
    """
    if _registry is None:
        raise RuntimeError(
            "ModelRegistry has not been initialised. "
            "Ensure `initialise_registry()` is called during application startup."
        )
    return _registry


def initialise_registry() -> ModelRegistry:
    """
    Build, load, and store the global ModelRegistry.

    Called once from the FastAPI lifespan startup handler.
    """
    global _registry
    _registry = ModelRegistry()
    _registry.load()
    return _registry
