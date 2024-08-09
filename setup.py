from setuptools import setup, find_packages

setup(
    name='PyTorchLabFlow',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'librosa',
        'pandas',
        'tqdm'
    ],
    author='BBEK-Anand',
    author_email='',
    description='A pipeline for managing PyTorch models, datasets, optimizers, and more.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BBEK-Anand/PyTorchLabFlow',
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
