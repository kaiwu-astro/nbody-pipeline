"""Tests for nbody_pipeline.visualization.evolution_path."""

from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from nbody_pipeline.visualization.evolution_path import (
    EvolutionPath,
    EvolutionPathVisualizer,
    PathBinary,
    PathEpoch,
    PathMember,
    assign_member_columns,
    mass_to_display_radius,
)


@pytest.fixture(autouse=True)
def mock_color_converter():
    """Mock BlackbodyColorConverter (same pattern as test_visualization.py)."""
    with patch("nbody_pipeline.visualization.base.BlackbodyColorConverter") as mock:
        mock.return_value.get_rgb = Mock(return_value=np.array([[0.5, 0.5, 0.5]]))
        yield mock


@pytest.fixture
def mock_config():
    config = Mock()
    config.kw_to_stellar_type = {1: "MS", 14: "BH"}
    config.close_figure_in_ipython = False
    return config


class TestMassToDisplayRadius:
    def test_bounded(self):
        radii = [mass_to_display_radius(m) for m in (0.01, 0.1, 1, 10, 100, 1000)]
        assert all(0.12 <= r <= 0.85 for r in radii)

    def test_monotonic_sqrt(self):
        masses = [0.5, 1, 5, 10, 50, 100]
        radii = [mass_to_display_radius(m) for m in masses]
        assert radii == sorted(radii)

    def test_monotonic_log(self):
        masses = [0.5, 1, 5, 10, 50, 100]
        radii = [mass_to_display_radius(m, scale="log") for m in masses]
        assert radii == sorted(radii)

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError):
            mass_to_display_radius(1.0, scale="linear")


class TestAssignMemberColumns:
    def test_binary_members_get_adjacent_columns(self):
        path = EvolutionPath(
            epochs=[
                PathEpoch(
                    time_myr=0.0,
                    members=[
                        PathMember("A", 10.0, 1),
                        PathMember("B", 5.0, 1),
                    ],
                    binaries=[PathBinary(member_ids=("A", "B"))],
                )
            ]
        )
        columns = assign_member_columns(path)
        assert abs(columns["A"] - columns["B"]) == 1

    def test_object_id_column_is_stable_across_epochs(self):
        path = EvolutionPath(
            epochs=[
                PathEpoch(
                    time_myr=0.0,
                    members=[PathMember("A", 10.0, 1), PathMember("B", 5.0, 1)],
                    binaries=[PathBinary(member_ids=("A", "B"))],
                ),
                PathEpoch(
                    time_myr=1.0,
                    members=[PathMember("A", 10.0, 1), PathMember("B", 5.0, 1)],
                    binaries=[PathBinary(member_ids=("A", "B"))],
                ),
                PathEpoch(
                    time_myr=2.0,
                    members=[PathMember("A", 10.0, 1), PathMember("B", 5.0, 1)],
                    binaries=[PathBinary(member_ids=("A", "B"))],
                ),
            ]
        )
        columns = assign_member_columns(path)
        # Re-derive from the prefix only and check the assignment agrees.
        prefix_path = EvolutionPath(epochs=path.epochs[:1])
        prefix_columns = assign_member_columns(prefix_path)
        assert prefix_columns["A"] == columns["A"]
        assert prefix_columns["B"] == columns["B"]

    def test_exchange_reuses_freed_column_to_stay_adjacent(self):
        # Epoch 0: A-B binary. Epoch 1: B is exchanged out, C exchanged in.
        path = EvolutionPath(
            epochs=[
                PathEpoch(
                    time_myr=0.0,
                    members=[PathMember("A", 10.0, 1), PathMember("B", 5.0, 1)],
                    binaries=[PathBinary(member_ids=("A", "B"))],
                ),
                PathEpoch(
                    time_myr=1.0,
                    members=[PathMember("A", 10.0, 1), PathMember("C", 6.0, 1)],
                    binaries=[PathBinary(member_ids=("A", "C"))],
                ),
            ]
        )
        columns = assign_member_columns(path)
        assert columns["A"] == 0
        # B's column (1) should be freed and reused by C to stay adjacent.
        assert columns["C"] == 1

    def test_new_unrelated_member_gets_its_own_column(self):
        path = EvolutionPath(
            epochs=[
                PathEpoch(time_myr=0.0, members=[PathMember("A", 10.0, 1)]),
                PathEpoch(
                    time_myr=1.0,
                    members=[PathMember("A", 10.0, 1), PathMember("D", 3.0, 1)],
                ),
            ]
        )
        columns = assign_member_columns(path)
        assert columns["A"] != columns["D"]


class TestEvolutionPathVisualizerPlot:
    def _simple_path(self) -> EvolutionPath:
        return EvolutionPath(
            title="Test system",
            epochs=[
                PathEpoch(
                    time_myr=0.0,
                    members=[
                        PathMember("A", 20.0, 1, label="A"),
                        PathMember("B", 8.0, 1, label="B"),
                    ],
                    binaries=[PathBinary(member_ids=("A", "B"), a=1.0, e=0.1)],
                    event_label="t0",
                ),
                PathEpoch(
                    time_myr=5.0,
                    members=[
                        PathMember("A", 20.0, 14, label="A"),
                        PathMember("B", 8.0, 2, in_ce=True, label="B"),
                    ],
                    binaries=[PathBinary(member_ids=("A", "B"), is_common_envelope=True)],
                    event_label="CE",
                    annotation="SN forms BH",
                ),
            ],
        )

    def test_plot_creates_output_file(self, mock_config, tmp_path):
        visualizer = EvolutionPathVisualizer(mock_config)
        out = tmp_path / "evo.jpg"

        result = visualizer.plot(self._simple_path(), out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_myr_time_axis(self, mock_config, tmp_path):
        visualizer = EvolutionPathVisualizer(mock_config)
        out = tmp_path / "evo_myr.jpg"

        result = visualizer.plot(self._simple_path(), out, time_axis="myr")

        assert result == out
        assert out.exists()

    def test_unknown_time_axis_raises(self, mock_config, tmp_path):
        visualizer = EvolutionPathVisualizer(mock_config)
        with pytest.raises(ValueError):
            visualizer.plot(self._simple_path(), tmp_path / "bad.jpg", time_axis="linear")

    def test_empty_path_does_not_crash(self, mock_config, tmp_path):
        visualizer = EvolutionPathVisualizer(mock_config)
        out = tmp_path / "empty.jpg"
        result = visualizer.plot(EvolutionPath(epochs=[], title="Empty"), out)
        assert result == out
        assert out.exists()
