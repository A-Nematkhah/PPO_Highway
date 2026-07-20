"""Unit tests for eureka/csv_export.py."""

import csv

from eureka.csv_export import (
    export_all,
    export_candidate_metrics_csv,
    export_generation_summary_csv,
    export_pareto_archive_csv,
)


def _candidate(candidate_id, module_path, generation, crash, speed, overtakes, rank=0):
    return {
        "candidate_id": candidate_id,
        "module_path": module_path,
        "generation": generation,
        "pareto_rank": rank,
        "metrics": {
            "crash_rate": crash,
            "mean_speed": speed,
            "mean_overtakes": overtakes,
            "mean_raw_return": 10.0,
        },
    }


def _sample_full_log():
    return [
        {
            "generation": 0,
            "selection_mode": "pareto",
            "pareto_front_size": 2,
            "archive_size": 2,
            "legacy_scalar_winner_id": "a",
            "legacy_scalar_winner_on_front": True,
            "representative_id": "a",
            "results": [
                dict(_candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0),
                     candidate_index=0, source="llm", legacy_fitness=1.1, crowding_distance=float("inf"), archive_member=True),
                dict(_candidate("b", "eureka.candidates.gen0_cand1", 0, 0.3, 25.0, 2.0),
                     candidate_index=1, source="llm", legacy_fitness=0.9, crowding_distance=0.4, archive_member=True),
            ],
        },
        {
            "generation": 1,
            "selection_mode": "pareto",
            "pareto_front_size": 1,
            "archive_size": 2,
            "legacy_scalar_winner_id": "c",
            "legacy_scalar_winner_on_front": False,
            "representative_id": "a",
            "results": [
                dict(_candidate("c", "eureka.candidates.gen1_cand0", 1, 0.5, 30.0, 3.0, rank=1),
                     candidate_index=0, source="llm", legacy_fitness=1.5, crowding_distance=float("inf"), archive_member=False),
            ],
        },
    ]


def test_export_pareto_archive_csv_writes_expected_rows(tmp_path):
    archive = [_candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0)]
    path = export_pareto_archive_csv(archive, tmp_path / "pareto_archive.csv", representative_id="a")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "a"
    assert rows[0]["module_path"] == "eureka.candidates.gen0_cand0"
    assert rows[0]["crash_rate"] == "0.1"
    assert rows[0]["archive_member"] == "True"
    assert rows[0]["is_winner"] == "True"
    assert rows[0]["smoothness"] == ""  # not present in metrics -> empty column


def test_export_pareto_archive_csv_empty_archive_still_writes_header(tmp_path):
    path = export_pareto_archive_csv([], tmp_path / "pareto_archive.csv", representative_id=None)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_export_pareto_archive_csv_includes_smoothness_when_present(tmp_path):
    candidate = _candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0)
    candidate["metrics"]["smoothness"] = 0.87
    path = export_pareto_archive_csv([candidate], tmp_path / "out.csv", representative_id=None)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["smoothness"] == "0.87"


def test_export_generation_summary_csv_one_row_per_generation(tmp_path):
    full_log = _sample_full_log()
    path = export_generation_summary_csv(full_log, tmp_path / "generation_summary.csv")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["generation"] == "0"
    assert rows[0]["n_results"] == "2"
    assert rows[0]["legacy_scalar_winner_on_front"] == "True"
    assert rows[1]["legacy_scalar_winner_on_front"] == "False"


def test_export_generation_summary_csv_handles_all_candidates_failed_record(tmp_path):
    full_log = [{
        "generation": 0, "results": [], "pareto_front_size": 0, "archive_size": 0,
        "selection_mode": "pareto", "all_candidates_failed": True,
    }]
    path = export_generation_summary_csv(full_log, tmp_path / "out.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["n_results"] == "0"
    assert rows[0]["all_candidates_failed"] == "True"


def test_export_candidate_metrics_csv_one_row_per_evaluated_candidate(tmp_path):
    full_log = _sample_full_log()
    path = export_candidate_metrics_csv(full_log, tmp_path / "candidate_metrics.csv")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3  # 2 in gen0 + 1 in gen1
    assert {r["candidate_id"] for r in rows} == {"a", "b", "c"}
    row_c = next(r for r in rows if r["candidate_id"] == "c")
    assert row_c["generation"] == "1"
    assert row_c["pareto_rank"] == "1"
    assert row_c["archive_member"] == "False"


def test_export_all_writes_all_three_files(tmp_path):
    full_log = _sample_full_log()
    archive = [_candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0)]

    paths = export_all(full_log, archive, tmp_path, representative_id="a")

    assert set(paths) == {"pareto_archive", "generation_summary", "candidate_metrics"}
    for path in paths.values():
        assert path.is_file()
        assert path.parent == tmp_path
