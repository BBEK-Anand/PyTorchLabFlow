# scripts/bump_version.py

import os

VERSION_FILE = './src/PTLF/_version.py'

def read_version():
    """Read and parse the __version__ value from the version file."""
    with open(VERSION_FILE, 'r') as f:
        for line in f:
            if line.strip().startswith('__version__'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    version_str = parts[1].strip().strip('"\'')
                    break
        else:
            raise RuntimeError("Version string not found in _version.py")

    parts = version_str.split('.')
    if not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid version format: {version_str}")

    return [int(p) for p in parts]

def bump_patch_version():
    """Increment the last numeric part of the version."""
    parts = read_version()
    parts[-1] += 1  # bump the last number
    new_version = '.'.join(map(str, parts))

    with open(VERSION_FILE, 'w') as f:
        f.write(f"__version__ = '{new_version}'\n")

    print(f"✅ Bumped version to {new_version}")

if __name__ == "__main__":
    bump_patch_version()
