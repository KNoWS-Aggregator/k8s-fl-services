from unittest.mock import Mock

from weight_aggregation.client_registry import ClientRegistry


def test_refresh_reads_training_services_from_case_slice(monkeypatch):
    request = Mock(
        return_value=Mock(
            json=lambda: {
                "data": {
                    "case": {
                        "id": "research-case",
                        "trainingServices": [
                            "https://hospital-a.example/training/",
                            "https://hospital-b.example/training",
                        ],
                    }
                }
            },
            raise_for_status=lambda: None,
        )
    )
    monkeypatch.setattr("weight_aggregation.client_registry.httpx.post", request)
    registry = ClientRegistry("https://researcher.example/slices/case-study/")

    changes = registry.refresh()

    assert changes == {
        "added": [
            "https://hospital-a.example/training",
            "https://hospital-b.example/training",
        ],
        "removed": [],
    }
    assert registry.snapshot() == {
        "https://hospital-a.example/training",
        "https://hospital-b.example/training",
    }
    assert request.call_args.args == (
        "https://researcher.example/slices/case-study/query",
    )
    assert request.call_args.kwargs["json"] == {
        "query": "{ case { id trainingServices } }"
    }
