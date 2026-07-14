"""Reusable Dragon-2-style "evolution path" cartoon diagrams for a binary's history.

Renders a vertical sequence of epochs (time flowing top-to-bottom) with
member stars drawn as circles (size ~ mass, colour ~ stellar type), binary
pairs linked by a dashed orbit ellipse (or a filled common-envelope blob),
and same-object continuity across epochs drawn as connecting lines. See
Dragon-2 (Rizzuto et al. 2023, doi:10.1093/mnras/stad2292 Fig. 1) and
Rantala et al. 2025 (doi:10.1093/mnrasl/slaf064 Fig. 3) for the reference
style this mirrors.

The data model (:class:`PathMember`, :class:`PathBinary`, :class:`PathEpoch`,
:class:`EvolutionPath`) is intentionally decoupled from any particular study
or KSTAR-decoding convention -- callers populate it from whatever analysis
they are doing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from nbody_pipeline.visualization.base import BaseVisualizer

logger = logging.getLogger(__name__)

__all__ = [
    "PathMember",
    "PathBinary",
    "PathEpoch",
    "EvolutionPath",
    "EvolutionPathVisualizer",
    "DEFAULT_KW_COLORS",
    "mass_to_display_radius",
    "assign_member_columns",
]


@dataclass(frozen=True)
class PathMember:
    """One star present at one epoch."""

    object_id: Hashable
    mass_msun: float
    kw: int
    in_ce: bool = False
    label: Optional[str] = None


@dataclass(frozen=True)
class PathBinary:
    """One bound pair present at one epoch."""

    member_ids: Tuple[Hashable, Hashable]
    a: Optional[float] = None
    e: Optional[float] = None
    is_common_envelope: bool = False


@dataclass(frozen=True)
class PathEpoch:
    """One labelled snapshot in the path (a row in the diagram)."""

    time_myr: float
    members: Sequence[PathMember] = field(default_factory=tuple)
    binaries: Sequence[PathBinary] = field(default_factory=tuple)
    event_label: Optional[str] = None
    annotation: Optional[str] = None


@dataclass(frozen=True)
class EvolutionPath:
    """A full evolution path: an ordered sequence of epochs plus a title."""

    epochs: Sequence[PathEpoch]
    title: str = ""


# Qualitative default colour per KSTAR code (see config.kw_to_stellar_type
# for the code->label mapping); loosely follows the Dragon-2 figure palette.
DEFAULT_KW_COLORS: Dict[int, str] = {
    -1: "#87CEFA",  # PMS
    0: "#6699CC",  # LMS
    1: "#4169E1",  # MS
    2: "#8B4513",  # HG
    3: "#B22222",  # GB
    4: "#DAA520",  # CHeB
    5: "#CD853F",  # EAGB
    6: "#A0522D",  # TPAGB
    7: "#FF69B4",  # HeMS
    8: "#FF1493",  # HeHG
    9: "#C71585",  # HeGB
    10: "#B0C4DE",  # HeWD
    11: "#A9A9A9",  # COWD
    12: "#808080",  # ONeWD
    13: "#4B0082",  # NS
    14: "#000000",  # BH
    15: "#D3D3D3",  # MLR
}
_FALLBACK_COLOR = "#999999"
_COMMON_ENVELOPE_FACECOLOR = "#F4A460"


def _format_myr(time_myr: float) -> str:
    """Human-friendly time label: avoid ``3.3e+03`` for ordinary magnitudes."""
    if abs(time_myr) >= 100:
        return f"{time_myr:,.0f}"
    return f"{time_myr:.3g}"


def mass_to_display_radius(
    mass_msun: Any,
    *,
    mass_min: float = 0.1,
    mass_max: float = 100.0,
    radius_min: float = 0.12,
    radius_max: float = 0.85,
    scale: str = "sqrt",
) -> Any:
    """Map stellar mass to a bounded, monotonic circle radius for display.

    ``scale`` is ``"sqrt"`` (default, circle *area* roughly tracks mass) or
    ``"log"`` (compresses a wide mass range, e.g. sub-solar donors next to a
    massive BH). Output is always clipped to ``[radius_min, radius_max]``.
    """
    if scale not in ("sqrt", "log"):
        raise ValueError(f"Unknown scale {scale!r}; expected 'sqrt' or 'log'")

    mass = np.clip(np.asarray(mass_msun, dtype=float), mass_min, mass_max)
    if scale == "sqrt":
        normalized = np.sqrt((mass - mass_min) / (mass_max - mass_min))
    else:
        log_min, log_max = np.log10(mass_min), np.log10(mass_max)
        normalized = (np.log10(mass) - log_min) / (log_max - log_min)

    radius = radius_min + normalized * (radius_max - radius_min)
    result = np.clip(radius, radius_min, radius_max)
    return result.item() if np.isscalar(mass_msun) or np.ndim(mass_msun) == 0 else result


def assign_member_columns(path: EvolutionPath) -> Dict[Hashable, int]:
    """Assign a stable horizontal column to every ``object_id`` in ``path``.

    Rules (a simple, deterministic swim-lane layout -- not a general graph
    layout solver):

    - An ``object_id``'s column, once assigned, never changes.
    - The two members of a :class:`PathBinary` are assigned adjacent
      columns whenever at least one of them is new.
    - A column frees up (becomes reusable by a later, unrelated member)
      once its current occupant has appeared for the last time in the
      path -- this lets an exchanged-in partner reuse the slot vacated by
      the exchanged-out one, keeping binaries visually adjacent across an
      exchange event.
    """
    last_seen: Dict[Hashable, int] = {}
    for i, epoch in enumerate(path.epochs):
        for member in epoch.members:
            last_seen[member.object_id] = i

    column_of: Dict[Hashable, int] = {}
    used_columns: set[int] = set()

    def acquire(preferred: Optional[int] = None) -> int:
        if preferred is not None and preferred >= 0 and preferred not in used_columns:
            used_columns.add(preferred)
            return preferred
        col = 0
        while col in used_columns:
            col += 1
        used_columns.add(col)
        return col

    for i, epoch in enumerate(path.epochs):
        for binary in epoch.binaries:
            id1, id2 = binary.member_ids
            has1, has2 = id1 in column_of, id2 in column_of
            if has1 and has2:
                continue
            if has1 and not has2:
                column_of[id2] = acquire(column_of[id1] + 1)
            elif has2 and not has1:
                column_of[id1] = acquire(column_of[id2] + 1)
            else:
                c1 = acquire()
                column_of[id1] = c1
                column_of[id2] = acquire(c1 + 1)

        for member in epoch.members:
            if member.object_id not in column_of:
                column_of[member.object_id] = acquire()

        for member in epoch.members:
            if last_seen[member.object_id] == i:
                used_columns.discard(column_of[member.object_id])

    return column_of


class EvolutionPathVisualizer(BaseVisualizer):
    """Render an :class:`EvolutionPath` as a Dragon-2-style cartoon diagram."""

    def _kw_color(self, kw: int) -> str:
        return DEFAULT_KW_COLORS.get(int(kw), _FALLBACK_COLOR)

    def _kw_label(self, kw: int) -> str:
        mapping = getattr(self.config, "kw_to_stellar_type", None) or {}
        return mapping.get(int(kw), str(kw))

    def plot(
        self,
        path: EvolutionPath,
        output_path: str | Path,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        time_axis: str = "ordinal",
    ) -> Path:
        """Draw ``path`` and save it to ``output_path``.

        ``time_axis``:
        - ``"ordinal"`` (default): epochs are equally spaced rows (readable
          even when epoch spacing in real time is wildly uneven); the real
          time in Myr is annotated as text next to each row.
        - ``"myr"``: row position is proportional to ``epoch.time_myr``
          itself, so visual spacing reflects real elapsed time.
        """
        if time_axis not in ("ordinal", "myr"):
            raise ValueError(f"Unknown time_axis {time_axis!r}; expected 'ordinal' or 'myr'")

        output_path = Path(output_path)
        columns = assign_member_columns(path)
        n_cols = max(columns.values(), default=-1) + 1
        n_epochs = len(path.epochs)

        if figsize is None:
            figsize = (max(4.0, 1.8 * max(n_cols, 1) + 2.0), max(3.0, 1.6 * max(n_epochs, 1)))
        fig, ax = plt.subplots(figsize=figsize)

        if time_axis == "ordinal":
            y_positions = list(range(n_epochs))
        else:
            y_positions = [epoch.time_myr for epoch in path.epochs]

        last_epoch_of_column: Dict[int, Tuple[int, float]] = {}

        for epoch, y in zip(path.epochs, y_positions):
            member_by_id = {m.object_id: m for m in epoch.members}

            for binary in epoch.binaries:
                self._draw_binary(ax, binary, member_by_id, columns, y)

            for member in epoch.members:
                col = columns[member.object_id]
                self._draw_member(ax, member, col, y)

                if member.object_id in last_epoch_of_column:
                    prev_col, prev_y = last_epoch_of_column[member.object_id]
                    ax.plot(
                        [prev_col, col],
                        [prev_y, y],
                        color="black",
                        linewidth=0.8,
                        linestyle="-",
                        zorder=1,
                        alpha=0.6,
                    )
                last_epoch_of_column[member.object_id] = (col, y)

            right_x = n_cols - 0.3
            label_bits = [f"{_format_myr(epoch.time_myr)} Myr"]
            if epoch.event_label:
                label_bits.append(epoch.event_label)
            ax.text(
                right_x + 0.15,
                y,
                "\n".join(label_bits),
                va="center",
                ha="left",
                fontsize=8,
            )
            if epoch.annotation:
                ax.text(
                    -0.9,
                    y,
                    epoch.annotation,
                    va="center",
                    ha="right",
                    fontsize=7,
                    style="italic",
                    color="dimgray",
                )

        ax.set_xlim(-1.0, n_cols + 2.0)
        y_min, y_max = min(y_positions, default=0), max(y_positions, default=0)
        pad = 0.8 if time_axis == "ordinal" else max(0.05 * (y_max - y_min), 0.5)
        ax.set_ylim(y_max + pad, y_min - pad)  # time flows top -> bottom
        ax.set_xticks([])
        if time_axis == "ordinal":
            ax.set_yticks([])
        else:
            ax.set_ylabel("Time [Myr]")
        ax.set_title(path.title)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

        self._add_legend(ax, path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        self._close_figure(fig)
        return output_path

    def _draw_member(self, ax: plt.Axes, member: PathMember, col: int, y: float) -> None:
        radius = mass_to_display_radius(member.mass_msun)
        facecolor = self._kw_color(member.kw)
        edgecolor = _COMMON_ENVELOPE_FACECOLOR if member.in_ce else "black"
        circle = mpatches.Circle(
            (col, y),
            radius=radius * 0.35,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.3 if member.in_ce else 0.8,
            zorder=3,
        )
        ax.add_patch(circle)
        label = member.label if member.label is not None else f"{member.mass_msun:.2g}"
        ax.text(
            col,
            y + radius * 0.35 + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            zorder=4,
        )

    def _draw_binary(
        self,
        ax: plt.Axes,
        binary: PathBinary,
        member_by_id: Dict[Hashable, PathMember],
        columns: Dict[Hashable, int],
        y: float,
    ) -> None:
        id1, id2 = binary.member_ids
        if id1 not in member_by_id or id2 not in member_by_id:
            return
        col1, col2 = columns[id1], columns[id2]
        cx = (col1 + col2) / 2.0
        width = abs(col2 - col1) + 0.5

        if binary.is_common_envelope:
            ellipse = mpatches.Ellipse(
                (cx, y),
                width=width,
                height=0.55,
                facecolor=_COMMON_ENVELOPE_FACECOLOR,
                edgecolor="none",
                alpha=0.55,
                zorder=2,
            )
            ax.add_patch(ellipse)
        else:
            ellipse = mpatches.Ellipse(
                (cx, y),
                width=width,
                height=0.4,
                facecolor="none",
                edgecolor="gray",
                linestyle="--",
                linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(ellipse)
            if binary.a is not None or binary.e is not None:
                a_txt = f"a={binary.a:.2g}" if binary.a is not None else ""
                e_txt = f"e={binary.e:.2g}" if binary.e is not None else ""
                ax.text(
                    cx,
                    y - 0.35,
                    " ".join(t for t in (a_txt, e_txt) if t),
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="gray",
                )

    def _add_legend(self, ax: plt.Axes, path: EvolutionPath) -> None:
        seen_kw = sorted(
            {int(m.kw) for epoch in path.epochs for m in epoch.members},
            key=lambda kw: (kw < 0, kw),
        )
        if not seen_kw:
            return
        handles = [
            mpatches.Patch(
                facecolor=self._kw_color(kw), edgecolor="black", label=self._kw_label(kw)
            )
            for kw in seen_kw
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="Stellar types",
            frameon=False,
        )

    def _close_figure(self, fig: plt.Figure) -> None:
        try:
            __IPYTHON__  # type: ignore[name-defined]  # noqa: F821
            if getattr(self.config, "close_figure_in_ipython", True):
                plt.close(fig)
        except NameError:
            plt.close(fig)
