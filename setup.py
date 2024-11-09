from setuptools import setup, find_packages

setup(
    name='PyTorchLabFlow',
    version='0.1.8.6',
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
    description='PyTorchLabFlow is a lightweight framework that simplifies PyTorch experiment management, reducing setup time with reusable components for training, logging, and checkpointing. It streamlines workflows, making it ideal for fast and efficient model development.',
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BBEK-Anand/PyTorchLabFlow',
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
