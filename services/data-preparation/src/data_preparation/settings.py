"""Environment-only service configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from croniter import croniter


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Required environment variable {name} is not set")
    return value.strip()


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _floating(name: str, default: float, minimum: float = 0) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    sources: str
    keycloak_realm_url: str
    dataset_id: str
    data_dir: Path
    download_chunk_size: int
    write_batch_size: int
    max_concurrent_participants: int
    max_retries: int
    retry_backoff_base: float
    request_timeout_seconds: int
    download_timeout_seconds: int
    auth_client_id_template: str
    auth_client_secret_template: str
    result_archive_name: str
    poll_enabled: bool
    poll_interval: str

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("DATA_DIR", "/app/data")).expanduser()
        if not data_dir.is_absolute():
            raise ValueError("DATA_DIR must be an absolute path")
        sources = _required("SOURCES").rstrip("/")
        parsed_sources = urlparse(sources)
        if not parsed_sources.scheme or not parsed_sources.netloc:
            raise ValueError("SOURCES must be an absolute URL")
        path_parts = [unquote(part) for part in parsed_sources.path.split("/") if part]
        if len(path_parts) < 3 or path_parts[-2] != "slices" or not path_parts[-1].startswith("case-"):
            raise ValueError(
                "SOURCES must end in /<hospital-pod>/slices/case-<case-id> and must not include /query"
            )
        archive_name = os.getenv("RESULT_ARCHIVE_NAME", "prepared-data.zip").strip()
        if not archive_name or Path(archive_name).name != archive_name or not archive_name.endswith(".zip"):
            raise ValueError("RESULT_ARCHIVE_NAME must be a .zip filename without a directory")
        poll_interval = os.getenv("POLL_INTERVAL", "@hourly").strip()
        if not croniter.is_valid(poll_interval):
            raise ValueError(
                "POLL_INTERVAL must be a valid five-field cron schedule or alias such as @hourly"
            )
        return cls(
            sources=sources,
            keycloak_realm_url=_required("AUTHN").rstrip("/"),
            dataset_id=_required("DATASET"),
            data_dir=data_dir,
            download_chunk_size=_integer("DOWNLOAD_CHUNK_SIZE", 100 * 1024 * 1024),
            write_batch_size=_integer("WRITE_BATCH_SIZE", 500_000),
            max_concurrent_participants=_integer("MAX_CONCURRENT_PARTICIPANTS", 1),
            max_retries=_integer("MAX_RETRIES", 5),
            retry_backoff_base=_floating("RETRY_BACKOFF_BASE", 2.0),
            request_timeout_seconds=_integer("REQUEST_TIMEOUT_SECONDS", 60),
            download_timeout_seconds=_integer("DOWNLOAD_TIMEOUT_SECONDS", 3600),
            auth_client_id_template=os.getenv("AUTH_CLIENT_ID_TEMPLATE", "{participant_id}_client"),
            auth_client_secret_template=os.getenv("AUTH_CLIENT_SECRET_TEMPLATE", "{participant_id}"),
            result_archive_name=archive_name,
            poll_enabled=_boolean("POL_ENABLED", True),
            poll_interval=poll_interval,
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw_long"

    @property
    def final_dir(self) -> Path:
        return self.data_dir / "final"

    @property
    def timing_report_path(self) -> Path:
        return self.data_dir / "timing_report.json"

    @property
    def result_archive_path(self) -> Path:
        return self.data_dir / self.result_archive_name

    @property
    def database_path(self) -> Path:
        return self.data_dir / "aggregate.duckdb"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def source_pod_id(self) -> str:
        return unquote(urlparse(self.sources).path.rstrip("/").rsplit("/slices/", 1)[0].rsplit("/", 1)[-1])

    @property
    def kvasir_server(self) -> str:
        parsed = urlparse(self.sources)
        pod_path = parsed.path.rstrip("/").rsplit("/slices/", 1)[0]
        base_path = pod_path.rsplit("/", 1)[0]
        return f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")

    def auth_credentials(self, participant_id: str) -> tuple[str, str]:
        values = {"participant_id": participant_id}
        try:
            return (
                self.auth_client_id_template.format(**values),
                self.auth_client_secret_template.format(**values),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid AUTH_CLIENT_ID_TEMPLATE or AUTH_CLIENT_SECRET_TEMPLATE") from exc
