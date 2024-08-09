from setuptools import setup

setup(
    name='Torch-PipeLine',
    version='0.1.0',
    packages=['.'],
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
    url='https://github.com/BBEK-Anand/Torch-PipeLine',
    classifiers=[
        'Programming Language :: Python :: 3',

        'Operating System :: OS Independent',
    ],
    python_requires='>=3.1',
)
