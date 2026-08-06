from data_preparation.main import _public_result


def test_public_result_reports_change_counts_without_participant_ids():
    result = {
        "participants_discovered": 3,
        "changes": {
            "added": ["participant-a", "participant-b"],
            "changed": ["participant-c"],
            "removed": [],
        },
        "published": True,
    }

    public = _public_result(result)

    assert public["changes"] == {"added": 2, "changed": 1, "removed": 0}
    assert "participant-a" not in str(public["changes"])
    assert result["changes"]["added"] == ["participant-a", "participant-b"]
