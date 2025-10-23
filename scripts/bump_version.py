# scripts/bump_version.py

import os
import sys

VERSION_FILE = './src/PTLF/_version.py'

def read_version():
    with open(VERSION_FILE, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                # Extract version between quotes
                version_str = line.split('=')[1].strip().strip('"\'')
                break
        else:
            raise RuntimeError("Version string not found")

    parts = version_str.split('.')
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid version format: {version_str}")

    return tuple(map(int, parts))

def bump_patch_version():
    x, y, z = read_version()
    new_version = f"{x}.{y}.{z + 1}"

    with open(VERSION_FILE, 'w') as f:
        f.write(f"__version__ = '{new_version}'\n")

    print(f"✅ Bumped version to {new_version}")

if __name__ == "__main__":
    bump_patch_version()
