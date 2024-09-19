# Copyright (c) 2024 BBEK-Anand
# Licensed under the MIT License
"""
PyTorchLabFlow 
==============

This is your go-to offlinesolution for managingPyTorch experimentswith ease. Run experiments securely on your local machine, no data sharing with 3rd parties. Need more power? Seamlesslytrasfer your setup to a high-end system without any reconfiguration. Wheather you are on a laptop or a workstation, PyTorchLabFlow ensures flexibility and privacy, allowingyou to experiment anywhere, anytime, without internet dependency. [Read more at github](https://github.com/BBEK-Anand/PyTorchLabFlow)

Functions:
----------
    - setup_project : Create the directory structure for a new machine learning project.
    - get_ppls : Retrieves pipeline information based on the specified mode and configuration.
    - verify : Verifies the existence or uniqueness of a pipeline based on the given mode and configuration.
    - up2date : Updates the root configuration file with the latest information from individual experiment JSON files.
    - set_default_config : Set the default configuration for the project.
    - test_mods : Configures and initializes a testing pipeline for evaluating a machine learning model.
    - train_new : Initializes and sets up a new pipeline for training a machine learning model.
    - re_train : Re-trains an existing pipeline or initializes a new pipeline with the provided configuration.
    - use_ppl : Configures and initializes a pipeline for an existing model or creates a new pipeline if necessary.
    - performance_plot : Plots performance metrics (accuracy and loss) over epochs for one or more pipelines.
    - multi_train : Train or re-train multiple pipelines up to the specified number of epochs.
    - get_model : Retrieves the model class or its name for a given pipeline or a list of pipelines.
    - archive : Archive or restore a project pipeline's files.
    - delete : Delete project files from the archive.
    - setup_trasfer : makes directories ready for transfer.
    - transfer :  Transfer pipeline files between active and transfer directories.
API :
-----
    - PipeLine : Class for managing the entire machine learning pipeline, including model setup, training, validation, and saving configurations and weights.

For more use help(function)

"""
from .pipeline import *