#!/usr/bin/env python
#
# usage: cmp-state-dumps [-h] [-d]
# Compare all files in the state_dumps directory and outputs a summary
# options:
#   -h, --help    show this help message and exit
#   -d, --delete  removes matching files
#
# Uses a pool of worker threads that compare each state dump.
# possible improvements: use a pool of workers for file removing.

import argparse
import glob
import re
import multiprocessing as mp
import os
from collections import defaultdict

POOL_SIZE = 16

STATE_DUMPS_PATH = "state_dumps"
VM_DIRECTORY = "vm"
NATIVE_DIRECTORY = "native"

LOG_PATH = "state_dumps/matching.log"


def compare(vm_dump_path: str):
    native_dump_path = re.sub(VM_DIRECTORY, NATIVE_DIRECTORY, vm_dump_path, count=1)

    if not (m := re.findall(r"/(0x.*).json", vm_dump_path)):
        raise Exception("bad path")
    tx = m[0]

    if not (m := re.findall(r"block(\d+)", vm_dump_path)):
        raise Exception("bad path")
    block = m[0]

    try:
        with open(native_dump_path) as f:
            native_dump = f.read()
        with open(vm_dump_path) as f:
            vm_dump = f.read()
    except:  # noqa: E722
        return ("MISS", block, tx)

    native_dump = re.sub(r".*reverted.*", "", native_dump, count=1)
    native_dump = re.sub(r".*cairo_native.*", "", native_dump)
    vm_dump = re.sub(r".*reverted.*", "", vm_dump, count=1)
    vm_dump = re.sub(r".*cairo_native.*", "", vm_dump)

    if native_dump == vm_dump:
        return ("MATCH", block, tx, vm_dump_path, native_dump_path)
    else:
        return ("DIFF", block, tx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cmp-state-dumps",
        description="Compare all files in the state_dumps directory and outputs a summary",
    )
    parser.add_argument(
        "-d", "--delete", action="store_true", help="removes matching files"
    )
    config = parser.parse_args()

    files = glob.glob(f"{STATE_DUMPS_PATH}/{VM_DIRECTORY}/*/*.json")
    files.sort(key=os.path.getmtime)

    print(f"Starting comparison with {POOL_SIZE} workers")

    stats = defaultdict(int)
    with mp.Pool(POOL_SIZE) as pool, open(LOG_PATH, mode="a") as log:
        for status, *info in pool.imap(compare, files):
            stats[status] += 1

            if status != "MATCH":
                (block, tx) = info
                print(status, block, tx)

            elif status == "MATCH" and config.delete:
                (block, tx, vm_dump_path, native_dump_path) = info

                log.write(f"{block} {tx}\n")
                log.flush()
                os.remove(native_dump_path)
                os.remove(vm_dump_path)

    print("Finished comparison")

    print()
    for key, count in stats.items():
        print(key, count)

    if stats["DIFF"] != 0 or stats["MISS"] != 0:
        exit(1)
    else:
        exit(0)
