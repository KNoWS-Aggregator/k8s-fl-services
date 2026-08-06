import json
import zipfile
from unittest.mock import Mock

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.converter import Converter
from data_preparation.settings import Settings


def _settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCES", "https://kvasir.example/hospital/slices/case-test")
    monkeypatch.setenv("DATASET", "accellero")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return Settings.from_env()


def _fake_ingest(converter, participant_id, _urls):
    output = converter.settings.raw_dir / participant_id / "data.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "timestamp": ["2026-01-01T00:00:00"] * 4,
                "metric": ["ACC_x", "ACC_y", "ACC_z", "GT"],
                "value": ["1.0", "2.0", "3.0", "walking"],
            }
        ),
        output,
    )
    return {"seconds": 0.0, "incomplete": {}}


def test_discovery_queries_dataset_at_enrolled_pod_url(monkeypatch, tmp_path):
    converter = Converter(_settings(monkeypatch, tmp_path))
    responses = [
        Mock(
            json=lambda: {
                "data": {
                    "case": {
                        "id": "case-test",
                        "enrolled": [
                            "https://participant.example/custom/kvasir/participant1/"
                        ],
                    }
                }
            }
        ),
        Mock(
            json=lambda: {
                "data": {
                    "dataset": {
                        "id": "dataset-test",
                        "distributions": [
                            {"id": "part-1", "downloadURL": "https://files.example/1"}
                        ],
                    }
                }
            }
        ),
    ]
    requests = []

    def fake_request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(converter, "_request", fake_request)

    assert converter.discover() == {"participant1": ["https://files.example/1"]}
    assert requests[0][1] == "https://kvasir.example/hospital/slices/case-test/query"
    assert requests[1][1] == (
        "https://participant.example/custom/kvasir/participant1/slices/accellero/query"
    )
    assert requests[1][2]["headers"] == {"Content-Type": "application/json"}


def test_added_changed_and_removed_participants_are_applied_incrementally(
    monkeypatch, tmp_path
):
    converter = Converter(_settings(monkeypatch, tmp_path))
    monkeypatch.setattr(
        converter,
        "_ingest_participant",
        lambda participant, urls: _fake_ingest(converter, participant, urls),
    )

    monkeypatch.setattr(
        converter, "discover", lambda: {"participant1": ["url-1"], "participant2": ["url-2"]}
    )
    first = converter.run()
    assert first["changes"] == {
        "added": ["participant1", "participant2"],
        "changed": [],
        "removed": [],
    }

    monkeypatch.setattr(converter, "discover", lambda: {"participant2": ["url-2-updated"]})
    second = converter.run()
    assert second["changes"] == {
        "added": [],
        "changed": ["participant2"],
        "removed": ["participant1"],
    }

    connection = duckdb.connect(str(converter.settings.database_path), read_only=True)
    try:
        assert connection.sql("SELECT DISTINCT participant_id FROM accel").fetchall() == [
            ("participant2",)
        ]
        assert connection.sql("SELECT COUNT(*) FROM gt").fetchone()[0] == 1
    finally:
        connection.close()

    manifest = json.loads(converter.settings.manifest_path.read_text())
    assert list(manifest["participants"]) == ["participant2"]
    state = json.loads((converter.settings.data_dir / "dataset-state.json").read_text())
    assert state["status"] == "ready"
    assert state["generation"] == manifest["generation"]
    with zipfile.ZipFile(converter.settings.result_archive_path) as archive:
        assert archive.namelist() == ["accel.parquet", "gt.parquet"]
