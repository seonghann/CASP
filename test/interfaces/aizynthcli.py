""" Module containing classes and routines for the CLI
"""
from __future__ import annotations
import argparse
import json
import os
import warnings
import logging
import importlib
import tempfile
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from test.aizynthfinder import AiZynthFinder
from test.utils.files import cat_hdf_files, split_file, start_processes
from test.utils.logging import logger, setup_logger

if TYPE_CHECKING:
    from test.utils.type_utils import StrDict


def _do_clustering(
    finder: AiZynthFinder, results: StrDict, detailed_results: bool
) -> None:
    t0 = time.perf_counter_ns()
    results["cluster_labels"] = finder.routes.cluster(n_clusters=0)
    if not detailed_results:
        return

    results["cluster_time"] = (time.perf_counter_ns() - t0) * 1e-9
    results["distance_matrix"] = finder.routes.distance_matrix().tolist()


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("aizynthcli")
    parser.add_argument(
        "--smiles",
        required=True,
        help="the target molecule smiles or the path of a file containing the smiles",
    )
    parser.add_argument(
        "--config", required=True, help="the filename of a configuration file"
    )
    parser.add_argument("--policy", default="", help="the name of the policy to use")
    parser.add_argument(
        "--stocks", nargs="+", default=[], help="the name of the stocks to use"
    )
    parser.add_argument(
        "--output", help="the name of the output file (JSON or HDF5 file)"
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        default=False,
        help="if provided, detailed logging to file is enabled",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        help="if given, the input is split over a number of processes",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="if provided, perform automatic clustering",
    )
    return parser.parse_args()


def _select_stocks(finder: AiZynthFinder, args: argparse.Namespace) -> None:
    stocks = list(args.stocks)
    try:
        module = importlib.import_module("custom_stock")
    except ModuleNotFoundError:
        pass
    else:
        if hasattr(module, "stock"):
            finder.stock.load(module.stock, "custom_stock")  # type: ignore
            stocks.append("custom_stock")
    finder.stock.select(stocks or finder.stock.items)


def _process_single_smiles(
    smiles: str, finder: AiZynthFinder, output_name: str, do_clustering: bool
) -> None:
    output_name = output_name or "trees.json"
    finder.target_smiles = smiles
    finder.prepare_tree()
    finder.tree_search(show_progress=True)
    finder.build_routes()

    with open(output_name, "w") as fileobj:
        json.dump(finder.routes.dicts, fileobj, indent=2)
    logger().info(f"Trees saved to {output_name}")

    scores = ", ".join("%.4f" % score for score in finder.routes.scores)
    logger().info(f"Scores for best routes: {scores}")

    stats = finder.extract_statistics()
    if do_clustering:
        _do_clustering(finder, stats, detailed_results=False)
    stats_str = "\n".join(
        f"{key.replace('_', ' ')}: {value}" for key, value in stats.items()
    )
    logger().info(stats_str)


def _process_multi_smiles(
    filename: str, finder: AiZynthFinder, output_name: str, do_clustering: bool
) -> None:
    output_name = output_name or "output.hdf5"
    with open(filename, "r") as fileobj:
        smiles = [line.strip() for line in fileobj.readlines()]

    results = defaultdict(list)
    for smi in smiles:
        finder.target_smiles = smi
        finder.prepare_tree()
        search_time = finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()

        logger().info(f"Done with {smi} in {search_time:.3} s")
        if do_clustering:
            _do_clustering(finder, stats, detailed_results=True)
        for key, value in stats.items():
            results[key].append(value)
        results["top_scores"].append(
            ", ".join("%.4f" % score for score in finder.routes.scores)
        )
        results["trees"].append(finder.routes.dicts)

    data = pd.DataFrame.from_dict(results)
    with warnings.catch_warnings():  # This wil suppress a PerformanceWarning
        warnings.simplefilter("ignore")
        data.to_hdf(output_name, key="table", mode="w")
    logger().info(f"Output saved to {output_name}")


def _multiprocess_smiles(args: argparse.Namespace) -> None:
    def create_cmd(index, filename):
        cmd_args = [
            "aizynthcli",
            "--smiles",
            filename,
            "--config",
            args.config,
            "--output",
            hdf_files[index - 1],
        ]
        if args.policy:
            cmd_args.extend(["--policy", args.policy])
        if args.stocks:
            cmd_args.append("--stocks")
            cmd_args.extend(args.stocks)
        if args.cluster:
            cmd_args.append("--cluster")
        return cmd_args

    if not os.path.exists(args.smiles):
    
        raise ValueError(
            "For multiprocessing execution the --smiles argument needs to be a filename"
        )

    setup_logger(logging.INFO)
    filenames = split_file(args.smiles, args.nproc)
    hdf_files = [tempfile.mktemp(suffix=".hdf") for _ in range(args.nproc)]
    start_processes(filenames, "aizynthcli", create_cmd)

    if not all(os.path.exists(filename) for filename in hdf_files):
        raise FileNotFoundError(
            "Not all output files produced. Please check the individual log files: 'aizynthcli*.log'"
        )
    cat_hdf_files(hdf_files, args.output or "output.hdf5")


def main() -> None:
    """Entry point for the aizynthcli command"""
    args = _get_arguments()
    if args.nproc:
        return _multiprocess_smiles(args)

    multi_smiles = os.path.exists(args.smiles)

    file_level_logging = logging.DEBUG if args.log_to_file else None
    setup_logger(logging.INFO, file_level_logging)
    
    finder = AiZynthFinder(configfile=args.config)
    _select_stocks(finder, args)
    finder.expansion_policy.select(args.policy or finder.expansion_policy.items[0])
    try:
        finder.filter_policy.select(args.policy)
    except KeyError:
        pass
    
    if multi_smiles:
        _process_multi_smiles(args.smiles, finder, args.output, args.cluster)
    else:
        _process_single_smiles(args.smiles, finder, args.output, args.cluster)


if __name__ == "__main__":
    main()
