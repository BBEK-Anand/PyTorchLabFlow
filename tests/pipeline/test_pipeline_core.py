# test_pipeline_core.py
import unittest
import os
import sys
import shutil
import json
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from PTLF.lab import lab_setup
from PTLF.experiment import PipeLine

# Helper function to create dummy module files
def create_dummy_modules():
    """Creates dummy python files for models, datasets, etc."""
    libs_path = Path("Libs")
    
    # Dummy Model
    with open(libs_path / "models.py", "w") as f:
        f.write("from torch import nn\n")
        f.write("class DummyModel(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        f.write("        self.linear = nn.Linear(10, 1)\n")
        f.write("    def forward(self, x):\n")
        f.write("        return self.linear(x)\n")

    # Dummy Dataset
    with open(libs_path / "datasets.py", "w") as f:
        f.write("from torch.utils.data import Dataset\n")
        f.write("import torch\n")
        f.write("class DummyDataset(Dataset):\n")
        f.write("    def __len__(self):\n")
        f.write("        return 20\n")
        f.write("    def __getitem__(self, idx):\n")
        f.write("        return torch.randn(10), torch.tensor([float(idx % 2)])\n")

    # Dummy Loss, Accuracy, Optimizer
    with open(libs_path / "losses.py", "w") as f:
        f.write("from torch import nn\n")
        f.write("def DummyLoss(): return nn.BCEWithLogitsLoss()\n")
    
    with open(libs_path / "accuracies.py", "w") as f:
        f.write("import torch\n")
        f.write("class DummyAcc:\n")
        f.write("    def __call__(self, logits, labels):\n")
        f.write("        return torch.tensor(1.0)\n")
        f.write("    def to(self, device): return self\n")


class TestPipelineCore(unittest.TestCase):
    """Tests core methods of the PipeLine class."""

    @classmethod
    def setUpClass(cls):
        """Set up a temporary project structure once for all tests in this class."""
        cls.test_dir = Path("temp_core_test")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir()
        os.chdir(cls.test_dir)
        
        lab_setup(project_name=".", create_root=False)
        create_dummy_modules()
        
        # Add the temp project directory to Python path to allow imports
        sys.path.insert(0, str(Path(".").resolve()))

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary project directory."""
        sys.path.pop(0)
        os.chdir("..")
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Create a new PipeLine instance for each test."""
        self.pipeline = PipeLine()
        
        # Clean up config files before each test to ensure isolation
        if Path("internal/Configs/test_pipeline.json").exists():
            Path("internal/Configs/test_pipeline.json").unlink()
        if Path("internal/Weights/test_pipeline.pth").exists():
            Path("internal/Weights/test_pipeline.pth").unlink()
        if Path("internal/Histories/test_pipeline.csv").exists():
            Path("internal/Histories/test_pipeline.csv").unlink()


    def test_load_component(self):
        """Verify that Python classes can be loaded dynamically."""
        # Act
        model_class = self.pipeline.load_component("Libs.models.DummyModel")
        
        # Assert
        self.assertIsNotNone(model_class)
        self.assertTrue(isinstance(model_class, nn.Module))

    def test_setup_new_config(self):
        """Verify that setup correctly creates and saves a new configuration."""
        # Act
        self.pipeline.setup(
            name="test_pipeline",
            model_loc="Libs.models.DummyModel",
            dataset_loc="Libs.datasets.DummyDataset",
            loss_loc="Libs.losses.DummyLoss",
            accuracy_loc="Libs.accuracies.DummyAcc",
            optimizer_loc="torch.optim.Adam",
            config_path="internal/Configs/test_pipeline.json"
        )
        
        # Assert
        config_path = Path("internal/Configs/test_pipeline.json")
        weights_path = Path("internal/Weights/test_pipeline.pth")
        history_path = Path("internal/Histories/test_pipeline.csv")
        
        self.assertTrue(config_path.exists())
        self.assertTrue(weights_path.exists())
        self.assertTrue(history_path.exists())
        
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(config["ppl_name"], "test_pipeline")
        self.assertEqual(config["model_loc"], "Libs.models.DummyModel")

    def test_setup_use_existing_config(self):
        """Verify that setup can load an existing configuration file."""
        # Arrange: create a config file first
        config_data = {
            "ppl_name": "existing_pipeline",
            "model_loc": "Libs.models.DummyModel",
            "dataset_loc": "Libs.datasets.DummyDataset",
            "loss_loc": "Libs.losses.DummyLoss",
            "accuracy_loc": "Libs.accuracies.DummyAcc",
            "optimizer_loc": "torch.optim.Adam",
            "best": {"val_loss": 1.23},
            "last": {"epoch": 0}
        }
        config_path = Path("internal/Configs/existing_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f)
            
        # Act
        self.pipeline.setup(config_path=str(config_path), use_config=True)

        # Assert
        self.assertEqual(self.pipeline.name, "existing_pipeline")
        self.assertEqual(self.pipeline.cnfg["best"]["val_loss"], 1.23)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.loss)

    def test_prepare_data(self):
        """Verify that data loaders are created correctly."""
        # Arrange
        self.pipeline.setup(
            name="data_prep_pipeline",
            model_loc="Libs.models.DummyModel",
            dataset_loc="Libs.datasets.DummyDataset",
            loss_loc="Libs.losses.DummyLoss",
            accuracy_loc="Libs.accuracies.DummyAcc",
            optimizer_loc="torch.optim.Adam",
            train_data_src="some/path", # path is not used by DummyDataset, just needs to be non-null
            train_batch_size=4,
            valid_data_src="some/other/path",
            valid_batch_size=4,
            config_path="internal/Configs/data_prep.json"
        )
        
        # Act
        self.pipeline.prepare_data()

        # Assert
        self.assertIsNotNone(self.pipeline.trainDataLoader)
        self.assertIsNotNone(self.pipeline.validDataLoader)
        self.assertEqual(self.pipeline.trainDataLoader.batch_size, 4)
        self.assertEqual(len(self.pipeline.trainDataLoader.dataset), 20)
        self.assertTrue(self.pipeline._configured)

    def test_update_improves_loss(self):
        """Verify that `update` saves the model when validation loss improves."""
        # Arrange
        self.pipeline.setup(
            name="update_pipeline",
            model_loc="Libs.models.DummyModel",
            dataset_loc="Libs.datasets.DummyDataset",
            loss_loc="Libs.losses.DummyLoss",
            accuracy_loc="Libs.accuracies.DummyAcc",
            optimizer_loc="torch.optim.Adam",
            config_path="internal/Configs/update_pipeline.json",
        )
        self.pipeline.prepare_data(train_data_src=".", valid_data_src=".", train_batch_size=4, valid_batch_size=4)
        initial_mtime = os.path.getmtime(self.pipeline.weights_path)

        # Act
        update_data = {
            'epoch': 1, 'train_accuracy': 0.8, 'train_loss': 0.5,
            'val_accuracy': 0.7, 'val_loss': 0.6 # Lower than initial inf
        }
        self.pipeline.update(update_data)
        
        # Assert
        new_mtime = os.path.getmtime(self.pipeline.weights_path)
        self.assertGreater(new_mtime, initial_mtime) # File should have been re-saved
        self.assertEqual(self.pipeline.cnfg['best']['epoch'], 1)
        self.assertEqual(self.pipeline.cnfg['best']['val_loss'], 0.6)

if __name__ == '__main__':
    unittest.main()