# Copyright 2024 BBEK Anand
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


import os
import re
from setuptools import setup, find_packages

def get_version():
    version_file = os.path.join('src', 'PTLF', '_version.py')
    with open(version_file) as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if not match:
        raise RuntimeError("Unable to find version string in _version.py.")
    return match.group(1)

setup(
    name='PyTorchLabFlow',
    version=get_version(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'torch',
        'matplotlib',
        'pandas',
        'tqdm'
    ],
    author='BBEK-Anand',
    author_email='',
    description='PyTorchLabFlow is a lightweight framework that simplifies PyTorch experiment management...',
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BBEK-Anand/PyTorchLabFlow',
    license="Apache-2.0",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
