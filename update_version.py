import re
import os

version_file = "src/PTLF/_version.py"

with open(version_file, "r") as f:
    content = f.read()

match = re.search(r"^__version__ = ['\"](\d+)\.(\d+)\.(\d+)['\"]", content, re.M)
if not match:
    raise RuntimeError("Invalid version format in _version.py (expected x.y.z)")

major, minor, patch = map(int, match.groups())
patch += 1
new_version = f"{major}.{minor}.{patch}"

with open(version_file, "w") as f:
    f.write(f"__version__ = '{new_version}'\n")

print(f"✅ Version bumped to: {new_version}")
