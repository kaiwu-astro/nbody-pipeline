#!/usr/bin/env python3
"""
Main entry point for dragon3_pipelines CLI
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 限制线程数避免forkserver问题

import sys
import gc
import argparse
import getopt
import logging
import functools
import multiprocessing

import matplotlib.pyplot as plt
from rich.progress import Progress
from rich.logging import RichHandler

from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.io import HDF5FileProcessor, LagrFileProcessor
from dragon3_pipelines.visualization import (
    GalacticOrbitVisualizer,
    HDF5Visualizer,
    LagrVisualizer,
    PlotPurger,
)
from dragon3_pipelines.analysis import (
    CurrentMassLagrangianProcessor,
    GalacticEnergyAngularMomentumProcessor,
    GalacticOrbitProcessor,
    ParticleTracker,
)
from dragon3_pipelines.analysis.hdf5_scan import ttot_matches_sample

# Setup logger
try:
    logger
except NameError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])


class SimulationPlotter:
    """模拟处理类，管理整个模拟处理流程"""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)
        self.lagr_file_processor = LagrFileProcessor(config_manager)
        self.hdf5_visualizer = HDF5Visualizer(config_manager)
        self.lagr_visualizer = LagrVisualizer(config_manager)
        self.galactic_orbit_visualizer = GalacticOrbitVisualizer(config_manager)
        self.particle_tracker = ParticleTracker(config_manager)
        self.current_lagrangian_processor = CurrentMassLagrangianProcessor(config_manager)
        self.galactic_orbit_processor = GalacticOrbitProcessor(config_manager)
        self.galactic_energy_angular_momentum_processor = GalacticEnergyAngularMomentumProcessor(
            config_manager
        )

    def plot_hdf5_file(self, hdf5_file_path: str, simu_name: str) -> None:
        """处理单个HDF5文件（包含多个snapshot）

        Args:
            hdf5_file_path: Path to HDF5 file (contains multiple snapshots)
            simu_name: Name of the simulation
        """
        # 获取HDF5文件的代表时间
        t_nbody_in_filename = self.hdf5_file_processor.get_hdf5_file_time_from_filename(
            hdf5_file_path
        )
        if t_nbody_in_filename < self.config.skip_until_of[simu_name]:
            logger.debug("skipped")
            return

        # 加载数据
        df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)
        sample_every_nb_time = (
            getattr(self.config, "hdf5", {})
            .get("file_selection", {})
            .get("sample_every_nb_time", 1.0)
        )

        # 处理每个时间点
        for ttot in df_dict["scalars"]["TTOT"].unique():
            if ttot < self.config.skip_until_of[simu_name]:
                continue
            if not ttot_matches_sample(float(ttot), sample_every_nb_time):
                continue
            logger.debug(f"{ttot=}")

            # 获取该时间点的数据
            single_df_at_t, binary_df_at_t, is_valid = self.hdf5_file_processor.get_snapshot_at_t(
                df_dict, ttot
            )
            if not is_valid:
                logger.info(
                    f"Warning: {simu_name} {hdf5_file_path} {ttot=} data validation failed, skipping"
                )
                continue

            scalar_row_at_t = df_dict["scalars"].loc[ttot]
            if hasattr(scalar_row_at_t, "iloc") and not hasattr(scalar_row_at_t, "dtype"):
                scalar_row_at_t = scalar_row_at_t.iloc[0]

            # 位置散点图
            self.hdf5_visualizer.single.create_position_plot_jpg(single_df_at_t, simu_name)
            self.hdf5_visualizer.single.create_position_plot_wide_pc_jpg(single_df_at_t, simu_name)
            self.hdf5_visualizer.single.create_position_plot_orbital_xT_xR_wide_pc_jpg(
                single_df_at_t, scalar_row_at_t, simu_name
            )
            self.hdf5_visualizer.single.create_position_plot_orbital_xT_xL_wide_pc_jpg(
                single_df_at_t, scalar_row_at_t, simu_name
            )
            self.hdf5_visualizer.single.create_position_plot_hightlight_compact_objects_jpg(
                single_df_at_t, simu_name
            )
            self.hdf5_visualizer.single.create_position_plot_hightlight_compact_objects_wide_pc_jpg(
                single_df_at_t, simu_name
            )

            # 质量-距离关系图
            self.hdf5_visualizer.single.create_mass_distance_plot_density(single_df_at_t, simu_name)
            # CMD图
            self.hdf5_visualizer.single.create_CMD_plot_density(single_df_at_t, simu_name)
            # 彩色CMD图
            self.hdf5_visualizer.single.create_color_CMD_jpg(single_df_at_t, simu_name)
            if self.config.galactic_energy_angular_momentum.get("enabled", True):
                galactic_e_lz_path = (
                    self.hdf5_visualizer.single.galactic_energy_angular_momentum_plot_jpg_path(
                        single_df_at_t, simu_name
                    )
                )
                if not (self.config.skip_existing_plot and os.path.exists(galactic_e_lz_path)):
                    galactic_e_lz_df = (
                        self.galactic_energy_angular_momentum_processor.compute_snapshot(
                            single_df_at_t, scalar_row_at_t
                        )
                    )
                    self.hdf5_visualizer.single.create_galactic_energy_angular_momentum_plot_jpg(
                        galactic_e_lz_df, simu_name
                    )
                else:
                    logger.debug(f"Skip existing plot: {galactic_e_lz_path}")
            # 速度-位置 # 不知为何非常非常慢，先不弄
            # self.hdf5_visualizer.single.create_vx_x_plot_density(single_df_at_t, simu_name)

            # 双星
            if binary_df_at_t is not None and not binary_df_at_t.empty:
                # 质量比-主星质量图
                self.hdf5_visualizer.binary.create_mass_ratio_m1_plot_density(
                    binary_df_at_t, simu_name
                )
                self.hdf5_visualizer.binary.create_mass_ratio_m1_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # 半长轴-主星质量图
                self.hdf5_visualizer.binary.create_semi_m1_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_semi_m1_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # 偏心率-半长轴图
                self.hdf5_visualizer.binary.create_ecc_semi_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_ecc_semi_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                self.hdf5_visualizer.binary.create_ecc_semi_plot_jpg_compact_object_only_loglog(
                    binary_df_at_t, simu_name
                )
                # 绑定能-半长轴图
                self.hdf5_visualizer.binary.create_ebind_semi_plot_density(
                    binary_df_at_t, simu_name
                )
                self.hdf5_visualizer.binary.create_ebind_semi_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # GW时间-半长轴图
                self.hdf5_visualizer.binary.create_taugw_semi_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # 总质量-距离关系图
                self.hdf5_visualizer.binary.create_mtot_distance_plot_density(
                    binary_df_at_t, simu_name
                )
                self.hdf5_visualizer.binary.create_mtot_distance_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # 速度-位置
                self.hdf5_visualizer.binary.create_bin_vx_x_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_bin_vx_x_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )
                # 半长轴-距离
                self.hdf5_visualizer.binary.create_semi_distance_plot_density(
                    binary_df_at_t, simu_name
                )
                self.hdf5_visualizer.binary.create_semi_distance_plot_jpg_compact_object_only(
                    binary_df_at_t, simu_name
                )

            # 清理内存
            plt.close("all")
            gc.collect()

    def plot_lagr(self, simu_name: str) -> None:
        """处理Lagrangian半径数据"""
        l7df_sns = self.lagr_file_processor.load_sns_friendly_data(simu_name)
        self.lagr_visualizer.create_lagr_radii_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_avmass_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_total_mass_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_velocity_dispersion_plot(l7df_sns, simu_name)
        plt.close("all")
        gc.collect()

    def plot_current_mass_lagr(self, simu_name: str) -> None:
        """处理基于HDF5 snapshot当前质量的Lagrangian半径数据"""
        l7df_sns = self.current_lagrangian_processor.load_sns_friendly_data(simu_name, update=True)
        if l7df_sns.empty:
            logger.info(f"No current-mass Lagrangian data for {simu_name}, skipping plots")
            return
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="rlagr",
            filename_suffix="current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_rlagr,
        )
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="rlagr",
            filename_suffix="loglog_current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_rlagr_logx,
        )
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="avmass",
            filename_suffix="current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_avmass,
        )
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="avmass",
            filename_suffix="loglog_current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_avmass_logx,
        )
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="sigma",
            filename_suffix="current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_sigma,
        )
        self.lagr_visualizer.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="sigma",
            filename_suffix="loglog_current_mass",
            extra_ax_handler=self.lagr_visualizer._extra_ax_handler_sigma_logx,
        )
        plt.close("all")
        gc.collect()

    def plot_galactic_orbit(self, simu_name: str) -> None:
        """处理星团在星系中的轨道数据"""
        orbit_df = self.galactic_orbit_processor.load_plot_data(simu_name, update=True)
        if orbit_df.empty:
            logger.info(f"No galactic orbit data for {simu_name}, skipping plots")
            return
        self.galactic_orbit_visualizer.create_projection_plot(orbit_df, simu_name)
        self.galactic_orbit_visualizer.create_interactive_3d_html(orbit_df, simu_name)
        plt.close("all")
        gc.collect()

    def plot_all_simulations(self) -> None:
        """处理所有模拟"""
        for simu_name in self.config.pathof.keys():
            # 先画lagr
            self.plot_lagr(simu_name)
            if self.config.current_lagrangian.get("enabled", True):
                self.plot_current_mass_lagr(simu_name)
            if self.config.galactic_orbit.get("enabled", True):
                self.plot_galactic_orbit(simu_name)

            # 获取所有HDF5文件
            hdf5_files = self.hdf5_file_processor.get_all_hdf5_paths(simu_name)

            # 创建带固定参数的部分函数
            process_file_partial = functools.partial(self.plot_hdf5_file, simu_name=simu_name)

            # 使用进程池并行处理
            ctx = multiprocessing.get_context("forkserver")
            with ctx.Pool(
                processes=self.config.processes_count, maxtasksperchild=self.config.tasks_per_child
            ) as pool:
                with Progress() as progress:
                    task = progress.add_task(f"{simu_name} HDF5 Files", total=len(hdf5_files))
                    for _ in pool.imap(process_file_partial, hdf5_files):
                        progress.advance(task)


def _build_purge_parser() -> argparse.ArgumentParser:
    """Build the purge subcommand parser."""
    parser = argparse.ArgumentParser(prog="python -m dragon3_pipelines purge")
    parser.add_argument(
        "target", nargs="?", help="Purge target, e.g. single or binary.create_ecc_semi_plot_density"
    )
    parser.add_argument("--simu", dest="simu_name", help="Limit purge to one simulation")
    parser.add_argument(
        "--all-sims",
        action="store_true",
        help="Match every configured simulation prefix",
    )
    parser.add_argument("--plot-dir", help="Override configured plot directory")
    parser.add_argument(
        "--filename-suffix", help="Match a custom filename suffix, or '*' for any suffix"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only preview matched files")
    parser.add_argument(
        "--yes", action="store_true", help="Delete without interactive prompt after preview"
    )
    parser.add_argument("--list-targets", action="store_true", help="List supported purge targets")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def _build_main_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI help parser."""
    parser = argparse.ArgumentParser(
        prog="python -m dragon3_pipelines",
        description=(
            "Run Dragon3 plot generation by default, or use a subcommand for maintenance tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-k",
        "--skip-until",
        metavar="VALUE",
        help="Start processing from N-body time VALUE, or use 'last' to resume from existing plots",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(
        title="subcommands",
        metavar="command",
        dest="command",
    )
    subparsers.add_parser("purge", help="Preview or delete generated plot files")
    subparsers.add_parser("help", help="Show this help, or help for a command")

    parser.epilog = (
        "Examples:\n"
        "  python -m dragon3_pipelines\n"
        "  python -m dragon3_pipelines --skip-until=last\n"
        "  python -m dragon3_pipelines help purge\n"
        "  python -m dragon3_pipelines purge --list-targets\n"
        "\n"
        "The installed script 'dragon3-plot' accepts the same arguments."
    )
    return parser


def _print_help_topic(topic: str | None) -> int:
    """Print top-level help or subcommand help."""
    if topic in (None, "dragon3_pipelines", "dragon3-plot"):
        _build_main_parser().print_help()
        return 0
    if topic == "purge":
        _build_purge_parser().print_help()
        return 0

    print(f"Unknown help topic: {topic}")
    print("Use 'python -m dragon3_pipelines --help' to list available commands.")
    return 2


def _normalize_pipeline_opts(opts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Normalize CLI option aliases before passing them to ConfigManager."""
    normalized_opts = []
    for opt, arg in opts:
        if opt == "-k":
            normalized_opts.append(("--skip-until", arg))
        else:
            normalized_opts.append((opt, arg))
    return normalized_opts


def _main_purge(argv: list[str]) -> int:
    """Run the purge CLI subcommand."""
    if argv in (["-h"], ["--help"], ["help"]):
        _build_purge_parser().print_help()
        return 0
    parser = _build_purge_parser()
    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.list_targets:
        for target in PlotPurger.list_targets():
            print(target)
        return 0

    if args.target is None:
        parser.error("target is required unless --list-targets is used")
    if args.simu_name and args.all_sims:
        parser.error("use either --simu or --all-sims, not both")
    if not args.simu_name and not args.all_sims:
        parser.error("use --simu for one simulation or --all-sims for every configured prefix")

    config = ConfigManager(opts=[])
    purger = PlotPurger(config)
    simu_name = None if args.all_sims else args.simu_name

    if args.dry_run:
        result = purger.preview(
            args.target,
            simu_name=simu_name,
            plot_dir=args.plot_dir,
            filename_suffix=args.filename_suffix,
        )
        print(purger.format_preview(result.matched_paths))
        return 0

    result = purger.purge(
        args.target,
        simu_name=simu_name,
        plot_dir=args.plot_dir,
        filename_suffix=args.filename_suffix,
        yes=args.yes,
    )
    if result.cancelled:
        print("Purge cancelled")
        return 1
    print(f"Deleted {len(result.deleted_paths)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        return _print_help_topic(None)
    if argv and argv[0] == "help":
        if len(argv) > 2:
            print(f"Unexpected arguments: {' '.join(argv[2:])}")
            return 2
        return _print_help_topic(argv[1] if len(argv) == 2 else None)
    if argv and argv[0] == "purge":
        return _main_purge(argv[1:])

    try:
        long_options = ["skip-until=", "debug"]
        opts, args = getopt.getopt(argv, "k:", long_options)
        if args:
            print(f"Unexpected arguments: {' '.join(args)}")
            return 2
        if "--debug" in dict(opts):
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s: %(message)s")
        else:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
        opts = _normalize_pipeline_opts(opts)
    except getopt.GetoptError as err:
        print(err)
        return 2

    config = ConfigManager(opts=opts)

    # 初始化处理器
    plotter = SimulationPlotter(config)

    # 处理所有模拟
    plotter.plot_all_simulations()

    return 0


if __name__ == "__main__":
    sys.exit(main())
