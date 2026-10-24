"""Convenience entry point for two-pass replay HDF5 collection."""

import sys

from lerobot.scripts.lerobot_collect_bi_piper import main

if __name__ == "__main__":
    raise SystemExit(main(["replay", *sys.argv[1:]]))
