"""Module containing routines to work with files and processes."""
from __future__ import annotations
import tempfile
import subprocess
import time
import warnings
from typing import TYPE_CHECKING

import more_itertools
import pandas as pd

from test.utils.logging import logger

if TYPE_CHECKING:
    from test.utils.type_utils import List, Sequence, Any, Callable


def cat_hdf_files(input_files: List[str], output_name: str) -> None:
    """
    Concatenate hdf5 files with the key "table"

    :param input_files: the paths to the files to concatenate
    :param output_name: the name of the concatenated file
    """
    data = pd.read_hdf(input_files[0], key="table")
    for filename in input_files[1:]:
        new_data = pd.read_hdf(filename, key="table")
        data = data.append(new_data)

    with warnings.catch_warnings():  # This wil suppress a PerformanceWarning
        warnings.simplefilter("ignore")
        data.to_hdf(output_name, key="table")


def split_file(filename: str, nparts: int) -> List[str]:
    """
    Split the content of a text file into a given number of temporary files

    :param filename: the path to the file to split
    :param nparts: the number of parts to create
    :return: list of filenames of the parts
    """
    with open(filename, "r") as fileobj:
        lines = fileobj.read().splitlines()

    filenames = []
    for chunk in more_itertools.divide(nparts, lines):
        filenames.append(tempfile.mktemp())
        with open(filenames[-1], "w") as fileobj:
            fileobj.write("\n".join(chunk))
    return filenames


def start_processes(
    inputs: Sequence[Any], log_prefix: str, cmd_callback: Callable, poll_freq: int = 5
) -> None:
    """
    Start a number of background processes and wait for them
    to complete.

    The standard output and standard error is saved to a log file.

    The command to start for each process is given by the ``cmd_callback``
    function that takes the index of the process and an item of the input
    as arguments.

    :param inputs: a sequence of input to the processes
    :param log_prefix: the prefix to the log file of each processes
    :param cmd_callback: function that creates the process commands
    :param poll_freq: the polling frequency for checking if processes are completed
    """
    processes = []
    output_fileobjs = []
    for index, iinput in enumerate(inputs, 1):
        output_fileobjs.append(open(f"{log_prefix}{index}.log", "w"))
        cmd = cmd_callback(index, iinput)
        processes.append(
            subprocess.Popen(cmd, stdout=output_fileobjs[-1], stderr=subprocess.STDOUT)
        )
        logger().info(f"Started background task with pid={processes[-1].pid}")

    logger().info("Waiting for background tasks to complete...")
    not_finished = True
    while not_finished:
        time.sleep(poll_freq)
        not_finished = False
        for process, fileobj in zip(processes, output_fileobjs):
            fileobj.flush()
            if process.poll() is None:
                not_finished = True

    for fileobj in output_fileobjs:
        fileobj.close()
