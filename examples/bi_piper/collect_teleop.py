"""Convenience entry point for passive shared-CAN HDF5 collection."""

import sys

from lerobot.scripts.lerobot_collect_bi_piper import main

if __name__ == "__main__":
    raise SystemExit(main(["teleop", *sys.argv[1:]]))
