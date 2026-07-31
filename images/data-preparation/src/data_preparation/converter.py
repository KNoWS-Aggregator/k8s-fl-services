"""Disk-backed Kvasir RDF-to-Parquet conversion pipeline."""
from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
import zipfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, unquote, urlparse

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .settings import Settings

METRICS = {
    "smartphone.acceleration.x": "ACC_x",
    "smartphone.acceleration.y": "ACC_y",
    "smartphone.acceleration.z": "ACC_z",
    "wearable.activity": "GT",
}
OBSERVES = "https://saref.etsi.org/core/observes"
HAS_TIMESTAMP = "https://saref.etsi.org/core/hasTimestamp"
HAS_VALUE = "https://saref.etsi.org/core/hasValue"
PREDICATES = {OBSERVES, HAS_TIMESTAMP, HAS_VALUE}


class Converter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokens: dict[str, tuple[str, float]] = {}

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = requests.request(method, url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"{response.status_code} {response.reason} from {url}", response=response
                    )
                response.raise_for_status()
                return response
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt + 1 == self.settings.max_retries:
                    raise
                time.sleep(self.settings.retry_backoff_base * (2**attempt) + random.random())
        raise RuntimeError("Request failed") from last_error

    def _token(self, participant_id: str) -> str:
        cached = self._tokens.get(participant_id)
        if cached and cached[1] > time.time() + 5:
            return cached[0]
        client_id, client_secret = self.settings.auth_credentials(participant_id)
        response = self._request(
            "POST",
            f"{self.settings.keycloak_realm_url}/protocol/openid-connect/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        payload = response.json()
        token = payload["access_token"]
        self._tokens[participant_id] = (token, time.time() + payload.get("expires_in", 60))
        return token

    def _auth(self, participant_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token(participant_id)}"}

    def _pod_url(self, pod_id: str) -> str:
        return f"{self.settings.kvasir_server}/{quote(pod_id, safe='')}"

    def _query(self, pod_id: str, slice_id: str, query: str) -> dict:
        response = self._request(
            "POST",
            f"{self._pod_url(pod_id)}/slices/{quote(slice_id, safe='')}/query",
            headers={**self._auth(pod_id), "Content-Type": "application/json"},
            json={"query": query},
            timeout=self.settings.request_timeout_seconds,
        )
        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(error.get("message", str(error)) for error in payload["errors"])
            raise RuntimeError(f"Slice query failed for {pod_id}/{slice_id}: {messages}")
        return payload.get("data") or {}

    @staticmethod
    def _pod_id(url: str) -> str:
        parsed = urlparse(url)
        pod_id = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
        if not parsed.scheme or not parsed.netloc or not pod_id:
            raise ValueError(f"Invalid enrolled pod URL: {url}")
        if pod_id in {".", ".."} or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in pod_id
        ):
            raise ValueError(f"Unsafe participant pod ID: {pod_id!r}")
        return pod_id

    def discover(self) -> dict[str, list[str]]:
        response = self._request(
            "POST",
            f"{self.settings.sources}/query",
            headers={
                **self._auth(self.settings.source_pod_id),
                "Content-Type": "application/json",
            },
            json={"query": "{ case { id enrolled } }"},
            timeout=self.settings.request_timeout_seconds,
        )
        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(error.get("message", str(error)) for error in payload["errors"])
            raise RuntimeError(f"Source slice query failed: {messages}")
        case_data = payload.get("data") or {}
        enrolled = (case_data.get("case") or {}).get("enrolled") or []
        participants: dict[str, list[str]] = {}
        for enrolled_url in dict.fromkeys(enrolled):
            participant_id = self._pod_id(enrolled_url)
            data = self._query(
                participant_id,
                self.settings.dataset_id,
                "{ dataset { id distributions { id downloadURL } } }",
            )
            distributions = (data.get("dataset") or {}).get("distributions") or []
            urls = list(dict.fromkeys(item.get("downloadURL") for item in distributions))
            participants[participant_id] = [url for url in urls if url]
        return {participant: urls for participant, urls in participants.items() if urls}

    def _file_size(self, participant_id: str, url: str) -> int:
        response = self._request(
            "GET",
            url,
            headers={**self._auth(participant_id), "Range": "bytes=0-0"},
            timeout=self.settings.request_timeout_seconds,
        )
        return int(response.headers["Content-Range"].split("/")[-1])

    def _range(self, participant_id: str, url: str, start: int, end: int) -> bytes:
        expected = end - start + 1
        for attempt in range(self.settings.max_retries):
            try:
                response = self._request(
                    "GET",
                    url,
                    headers={**self._auth(participant_id), "Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=self.settings.download_timeout_seconds,
                )
                body = b"".join(response.iter_content(chunk_size=1024 * 1024))
                if len(body) != expected:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"Expected {expected} bytes for range {start}-{end}, got {len(body)}"
                    )
                return body
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                if attempt + 1 == self.settings.max_retries:
                    raise
                time.sleep(self.settings.retry_backoff_base * (2**attempt) + random.random())
        raise RuntimeError("Range download failed")

    def _lines(self, participant_id: str, url: str) -> Iterator[str]:
        size, start, remainder = self._file_size(participant_id, url), 0, b""
        while start < size:
            end = min(start + self.settings.download_chunk_size - 1, size - 1)
            pieces = (remainder + self._range(participant_id, url, start, end)).split(b"\n")
            remainder = pieces.pop()
            yield from (line.decode("utf-8", errors="ignore") for line in pieces)
            start = end + 1
        if remainder:
            yield remainder.decode("utf-8", errors="ignore")

    @staticmethod
    def _spo(line: str) -> tuple[str, str, str]:
        line = line.rstrip()
        if not line.endswith(" ."):
            raise ValueError
        line = line[:-2]
        first, second = line.find(" "), line.find(" ", line.find(" ") + 1)
        return line[1:first - 1], line[first + 2:second - 1], line[second + 1:-1].lstrip("<")

    @staticmethod
    def _value(predicate: str, obj: str) -> str | None:
        if predicate == OBSERVES:
            metric = obj.rsplit("/", 1)[-1]
            return metric if metric in METRICS else None
        return obj[1:obj.find('"', 1)]

    def _stream_file(self, participant_id: str, url: str, output: Path) -> int:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        temporary_output.unlink(missing_ok=True)
        pending: dict[str, dict[str, str]] = {}
        discarded: set[str] = set()
        timestamps: list[str] = []
        metrics: list[str] = []
        values: list[str] = []
        writer: pq.ParquetWriter | None = None

        def flush() -> None:
            nonlocal writer, timestamps, metrics, values
            if not timestamps:
                return
            table = pa.table({"timestamp": timestamps, "metric": metrics, "value": values})
            writer = writer or pq.ParquetWriter(str(temporary_output), table.schema)
            writer.write_table(table)
            timestamps, metrics, values = [], [], []

        try:
            for line in self._lines(participant_id, url):
                try:
                    subject, predicate, obj = self._spo(line)
                except ValueError:
                    continue
                if predicate not in PREDICATES or subject in discarded:
                    continue
                value = self._value(predicate, obj)
                if value is None:
                    discarded.add(subject)
                    pending.pop(subject, None)
                    continue
                row = pending.setdefault(subject, {})
                row[{OBSERVES: "metric", HAS_TIMESTAMP: "timestamp", HAS_VALUE: "value"}[predicate]] = value
                if len(row) == 3:
                    pending.pop(subject)
                    timestamps.append(row["timestamp"])
                    metrics.append(METRICS[row["metric"]])
                    values.append(row["value"])
                    if len(timestamps) >= self.settings.write_batch_size:
                        flush()
            flush()
        finally:
            if writer:
                writer.close()
        if writer:
            temporary_output.replace(output)
        else:
            output.unlink(missing_ok=True)
        return len(pending)

    @staticmethod
    def _safe_name(url: str, index: int) -> str:
        name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]) or str(index)
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in name)

    def _ingest_participant(self, participant_id: str, urls: list[str]) -> dict:
        started = time.perf_counter()
        incomplete = {}
        for index, url in enumerate(urls, start=1):
            name = self._safe_name(url, index)
            incomplete[name] = self._stream_file(
                participant_id, url, self.settings.raw_dir / participant_id / f"{name}.parquet"
            )
        return {"seconds": time.perf_counter() - started, "incomplete": incomplete}

    def _pivot(self, participant_ids: list[str]) -> dict:
        started = time.perf_counter()
        completed, skipped = [], []
        connection = duckdb.connect()
        try:
            for participant_id in participant_ids:
                files = list((self.settings.raw_dir / participant_id).glob("*.parquet"))
                if not files:
                    skipped.append(participant_id)
                    continue
                accel_out = self.settings.staging_dir / "accel" / f"{participant_id}.parquet"
                gt_out = self.settings.staging_dir / "gt" / f"{participant_id}.parquet"
                accel_out.parent.mkdir(parents=True, exist_ok=True)
                gt_out.parent.mkdir(parents=True, exist_ok=True)
                pattern = str(self.settings.raw_dir / participant_id / "*.parquet")
                for metric_filter, output, numeric in (
                    ("'ACC_x','ACC_y','ACC_z'", accel_out, "TRY_CAST(value AS DOUBLE)"),
                    ("'GT'", gt_out, "value"),
                ):
                    temporary_output = output.with_suffix(output.suffix + ".tmp")
                    temporary_output.unlink(missing_ok=True)
                    connection.sql(f"""
                        COPY (PIVOT (
                            SELECT '{participant_id}' participant_id,
                                   CAST(timestamp AS TIMESTAMP) AS event_time,
                                   metric, {numeric} AS metric_value
                            FROM read_parquet('{pattern}')
                            WHERE metric IN ({metric_filter})
                        ) ON metric IN ({metric_filter}) USING first(metric_value)
                          GROUP BY participant_id, event_time ORDER BY event_time)
                        TO '{temporary_output}' (FORMAT PARQUET)
                    """)
                    temporary_output.replace(output)
                completed.append(participant_id)
        finally:
            connection.close()
        return {"seconds": time.perf_counter() - started, "completed": completed, "skipped": skipped}

    def _load_manifest(self) -> dict[str, dict]:
        if not self.settings.manifest_path.is_file():
            return {}
        try:
            payload = json.loads(self.settings.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read persistent manifest: {exc}") from exc
        participants = payload.get("participants", {})
        if not isinstance(participants, dict):
            raise RuntimeError("Persistent manifest has an invalid participants field")
        return participants

    @staticmethod
    def _fingerprint(urls: list[str]) -> str:
        # Ignore transient signed-query parameters so refreshed download
        # credentials do not make an unchanged distribution look new.
        stable_urls = []
        for url in urls:
            parsed = urlparse(url)
            stable_urls.append(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
        encoded = json.dumps(sorted(stable_urls), separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _update_database(
        self,
        changed_participants: list[str],
        removed_participants: list[str],
    ) -> None:
        connection = duckdb.connect(str(self.settings.database_path))
        connection.sql("""
            CREATE TABLE IF NOT EXISTS accel (
                participant_id VARCHAR,
                event_time TIMESTAMP,
                ACC_x DOUBLE,
                ACC_y DOUBLE,
                ACC_z DOUBLE
            );
            CREATE TABLE IF NOT EXISTS gt (
                participant_id VARCHAR,
                event_time TIMESTAMP,
                GT VARCHAR
            );
        """)
        try:
            connection.sql("BEGIN")
            for participant_id in sorted(set(changed_participants + removed_participants)):
                connection.execute("DELETE FROM accel WHERE participant_id = ?", [participant_id])
                connection.execute("DELETE FROM gt WHERE participant_id = ?", [participant_id])
            for participant_id in changed_participants:
                accel = self.settings.staging_dir / "accel" / f"{participant_id}.parquet"
                gt = self.settings.staging_dir / "gt" / f"{participant_id}.parquet"
                connection.execute(
                    """
                    INSERT INTO accel
                    SELECT participant_id, event_time, ACC_x, ACC_y, ACC_z
                    FROM read_parquet(?)
                    """,
                    [str(accel)],
                )
                connection.execute(
                    """
                    INSERT INTO gt
                    SELECT participant_id, event_time, GT
                    FROM read_parquet(?)
                    """,
                    [str(gt)],
                )
            connection.sql("COMMIT")
        except Exception:
            connection.sql("ROLLBACK")
            raise
        finally:
            connection.close()

    def _export_results(self) -> None:
        """Atomically export the canonical tables and publish the download ZIP."""
        self.settings.downloads_dir.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.settings.database_path), read_only=True)
        try:
            for table in ("accel", "gt"):
                output = self.settings.downloads_dir / f"{table}.parquet"
                temporary = output.with_suffix(".parquet.tmp")
                temporary.unlink(missing_ok=True)
                connection.execute(
                    f"COPY (SELECT * FROM {table} ORDER BY participant_id, event_time) "
                    f"TO '{temporary}' (FORMAT PARQUET)"
                )
                temporary.replace(output)
        finally:
            connection.close()

        temporary = self.settings.result_archive_path.with_suffix(".zip.tmp")
        temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in ("accel.parquet", "gt.parquet"):
                archive.write(self.settings.downloads_dir / name, name)
        temporary.replace(self.settings.result_archive_path)

    def _save_manifest(self, participants: dict[str, list[str]], generation: str) -> None:
        payload = {
            "generation": generation,
            "updated_at_epoch": time.time(),
            "participants": {
                participant: {
                    "fingerprint": self._fingerprint(urls),
                }
                for participant, urls in sorted(participants.items())
            },
        }
        temporary = self.settings.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.settings.manifest_path)

    def _publish_dataset_state(
        self,
        status: str,
        generation: str,
        error: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "generation": generation,
            "updated_at_epoch": time.time(),
        }
        if error:
            payload["error"] = error
        path = self.settings.data_dir / "dataset-state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _cleanup_removed(self, participant_ids: list[str]) -> None:
        for participant_id in participant_ids:
            shutil.rmtree(self.settings.raw_dir / participant_id, ignore_errors=True)
            for category in ("accel", "gt"):
                (self.settings.staging_dir / category / f"{participant_id}.parquet").unlink(
                    missing_ok=True
                )

    def run(self) -> dict:
        started = time.perf_counter()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        participant_files = self.discover()
        previous = self._load_manifest()
        current_fingerprints = {
            participant: self._fingerprint(urls)
            for participant, urls in participant_files.items()
        }
        added = sorted(set(participant_files) - set(previous))
        removed = sorted(set(previous) - set(participant_files))
        changed = sorted(
            participant
            for participant in set(participant_files) & set(previous)
            if previous[participant].get("fingerprint") != current_fingerprints[participant]
        )
        to_process = added + changed
        self._cleanup_removed(to_process)

        ingest_started = time.perf_counter()
        ingested, failures = {}, {}
        with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_participants) as executor:
            futures = {
                executor.submit(self._ingest_participant, participant, urls): participant
                for participant, urls in participant_files.items()
                if participant in to_process
            }
            for future in as_completed(futures):
                participant = futures[future]
                try:
                    ingested[participant] = future.result()
                except Exception as exc:
                    failures[participant] = str(exc)

        pivot = self._pivot([participant for participant in to_process if participant not in failures])
        if failures:
            failed = ", ".join(sorted(failures))
            raise RuntimeError(f"Conversion failed for participant(s): {failed}")
        needs_publish = bool(to_process or removed) or not self.settings.result_archive_path.is_file()
        if needs_publish:
            generation = str(uuid.uuid4())
            self._publish_dataset_state("updating", generation)
            try:
                self._update_database(to_process, removed)
                self._export_results()
                self._save_manifest(participant_files, generation)
            except Exception as exc:
                self._publish_dataset_state("failed", generation, str(exc))
                raise
            else:
                self._publish_dataset_state("ready", generation)
                self._cleanup_removed(removed)
        elif self.settings.manifest_path.is_file():
            manifest = json.loads(self.settings.manifest_path.read_text(encoding="utf-8"))
            generation = manifest.get("generation")
            state_path = self.settings.data_dir / "dataset-state.json"
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else {}
            )
            if generation and (
                state.get("status") != "ready"
                or state.get("generation") != generation
            ):
                # The manifest is written only after both exports complete, so
                # it can safely recover a crash between manifest and state publication.
                self._publish_dataset_state("ready", generation)
        report = {
            "participants_discovered": len(participant_files),
            "changes": {"added": added, "changed": changed, "removed": removed},
            "ingest": {
                "seconds": time.perf_counter() - ingest_started,
                "participants": ingested,
                "failures": failures,
            },
            "pivot": pivot,
            "published": needs_publish,
            "total_seconds": time.perf_counter() - started,
        }
        temporary = self.settings.timing_report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(self.settings.timing_report_path)
        return report
