# scripts/bump_version.py

import re
import os
import sys

VERSION_FILE = './../src/PTLF/_version.py'

def read_version():
    with open(VERSION_FILE, 'r') as f:
        content = f.read()
    match = re.search(r"__version__ = ['\"](\d+)\.(\d+)\.(\d+)['\"]", content)
    if not match:
        raise RuntimeError("Version string not found")
    return tuple(map(int, match.groups()))

def bump_patch_version():
    x, y, z = read_version()
    new_version = f"{x}.{y}.{z + 1}"
    with open(VERSION_FILE, 'w') as f:
        f.write(f"__version__ = '{new_version}'\n")
    print(f"✅ Bumped version to {new_version}")

if __name__ == "__main__":
    bump_patch_version()
