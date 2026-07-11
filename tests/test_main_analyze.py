"""Smoke tests for the `analyze` CLI subcommand wiring in nbody_pipeline.__main__.

The two-task shared-read session behavior itself is already covered
end-to-end by tests/test_pilot_tasks.py; these tests only exercise the CLI
argument parsing and dispatch, with ConfigManager/HDF5ScanSession/processors
mocked out so no real HDF5 data or config file is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nbody_pipeline import __main__ as main_module


def _mock_config() -> MagicMock:
    config = MagicMock()
    config.pathof = {"sim_a": "/tmp/sim_a", "sim_b": "/tmp/sim_b"}
    return config


def test_build_analyze_parser_defaults() -> None:
    parser = main_module._build_analyze_parser()
    args = parser.parse_args([])
    assert args.simu_name is None
    assert args.force is False
    assert args.debug is False


def test_print_help_topic_analyze(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_module._print_help_topic("analyze") == 0
    assert "analyze" in capsys.readouterr().out


def test_main_dispatches_analyze_subcommand() -> None:
    with patch.object(main_module, "_main_analyze", return_value=0) as mock_main_analyze:
        assert main_module.main(["analyze", "--simu", "sim_a"]) == 0
    mock_main_analyze.assert_called_once_with(["--simu", "sim_a"])


def test_main_analyze_builds_shared_session_for_all_simulations() -> None:
    config = _mock_config()
    with (
        patch.object(main_module, "ConfigManager", return_value=config),
        patch.object(main_module, "CompactObjectHistoryProcessor") as mock_compact_cls,
        patch.object(main_module, "SnapshotSummaryProcessor") as mock_snapshot_cls,
        patch.object(main_module, "HDF5ScanSession") as mock_session_cls,
    ):
        mock_session = mock_session_cls.return_value
        mock_compact = mock_compact_cls.return_value
        mock_snapshot = mock_snapshot_cls.return_value

        result = main_module._main_analyze([])

    assert result == 0
    assert mock_compact.build_scan_job.call_count == 2
    assert mock_snapshot.build_scan_job.call_count == 2
    assert mock_session.add_job.call_count == 4
    mock_session.run.assert_called_once()


def test_main_analyze_limits_to_one_simulation_and_respects_force() -> None:
    config = _mock_config()
    with (
        patch.object(main_module, "ConfigManager", return_value=config),
        patch.object(main_module, "CompactObjectHistoryProcessor") as mock_compact_cls,
        patch.object(main_module, "SnapshotSummaryProcessor") as mock_snapshot_cls,
        patch.object(main_module, "HDF5ScanSession") as mock_session_cls,
    ):
        mock_session = mock_session_cls.return_value
        mock_compact = mock_compact_cls.return_value
        mock_snapshot = mock_snapshot_cls.return_value

        result = main_module._main_analyze(["--simu", "sim_a", "--force"])

    assert result == 0
    mock_compact.build_scan_job.assert_called_once_with("sim_a", force=True)
    mock_snapshot.build_scan_job.assert_called_once_with("sim_a", force=True)
    mock_session.run.assert_called_once()


def test_main_analyze_rejects_unknown_simulation() -> None:
    config = _mock_config()
    with patch.object(main_module, "ConfigManager", return_value=config):
        with pytest.raises(SystemExit):
            main_module._main_analyze(["--simu", "nope"])
