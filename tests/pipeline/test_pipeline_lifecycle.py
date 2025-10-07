# test_pipeline_lifecycle.py
import unittest
import os
import sys
import shutil
from pathlib import Path
from PTLF.lab import setup_project
from PTLF.experiment import (train_new, archive, delete, 
                                transfer, setup_transfer, get_ppls, up2date)
from test_pipeline_core import create_dummy_modules

class TestPipelineLifecycle(unittest.TestCase):
    """Tests for archiving, deleting, and transferring pipelines."""

    def setUp(self):
        """Set up a fresh project with a pipeline for each test."""
        self.test_dir = Path("temp_lifecycle_test")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        os.chdir(self.test_dir)
        
        setup_project(project_name=".", create_root=False)
        create_dummy_modules()
        sys.path.insert(0, str(Path(".").resolve()))

        # Create a pipeline to be managed
        train_new(
            name="lifecycle_ppl", model_loc="Libs.models.DummyModel", dataset_loc="Libs.datasets.DummyDataset",
            loss_loc="Libs.losses.DummyLoss", accuracy_loc="Libs.accuracies.DummyAcc", optimizer_loc="torch.optim.Adam"
        )
        up2date()

    def tearDown(self):
        """Clean up the project directory."""
        sys.path.pop(0)
        os.chdir("..")
        shutil.rmtree(self.test_dir)
        
    def test_archive_and_restore(self):
        """Verify that a pipeline can be archived and then restored."""
        # --- Archive ---
        # Assert initial state
        self.assertIn("lifecycle_ppl", get_ppls(config="internal"))
        self.assertNotIn("lifecycle_ppl", get_ppls(config="archive"))
        
        # Act
        archive(ppl="lifecycle_ppl")
        
        # Assert archived state
        self.assertNotIn("lifecycle_ppl", get_ppls(config="internal"))
        self.assertIn("lifecycle_ppl", get_ppls(config="archive"))
        self.assertTrue(Path("internal/Archived/Configs/lifecycle_ppl.json").exists())
        self.assertFalse(Path("internal/Configs/lifecycle_ppl.json").exists())

        # --- Restore ---
        # Act
        archive(ppl="lifecycle_ppl", reverse=True)

        # Assert restored state
        self.assertIn("lifecycle_ppl", get_ppls(config="internal"))
        self.assertNotIn("lifecycle_ppl", get_ppls(config="archive"))
        self.assertFalse(Path("internal/Archived/Configs/lifecycle_ppl.json").exists())
        self.assertTrue(Path("internal/Configs/lifecycle_ppl.json").exists())

    def test_delete(self):
        """Verify that a pipeline can be permanently deleted from the archive."""
        # Arrange: First, archive the pipeline
        archive(ppl="lifecycle_ppl")
        self.assertIn("lifecycle_ppl", get_ppls(config="archive"))
        self.assertTrue(Path("internal/Archived/Configs/lifecycle_ppl.json").exists())

        # Act
        delete(ppl="lifecycle_ppl")

        # Assert
        self.assertNotIn("lifecycle_ppl", get_ppls(config="archive"))
        self.assertFalse(Path("internal/Archived/Configs/lifecycle_ppl.json").exists())
        self.assertFalse(Path("internal/Archived/Weights/lifecycle_ppl.pth").exists())
        self.assertFalse(Path("internal/Archived/Histories/lifecycle_ppl.csv").exists())

    def test_transfer_export_move(self):
        """Verify exporting a pipeline with 'move' mode."""
        # Arrange
        setup_transfer()
        self.assertIn("lifecycle_ppl", get_ppls(config="internal"))
        
        # Act
        transfer(ppl="lifecycle_ppl", type="export", mode="move")

        # Assert
        self.assertNotIn("lifecycle_ppl", get_ppls(config="internal"))
        self.assertIn("lifecycle_ppl", get_ppls(config="transfer"))
        self.assertTrue(Path("internal/Transfer/Configs/lifecycle_ppl.json").exists())
        self.assertFalse(Path("internal/Configs/lifecycle_ppl.json").exists())
        
    def test_transfer_import_copy(self):
        """Verify importing a pipeline with 'copy' mode."""
        # Arrange: First export the pipeline
        setup_transfer()
        transfer(ppl="lifecycle_ppl", type="export", mode="move")
        self.assertIn("lifecycle_ppl", get_ppls(config="transfer"))
        self.assertNotIn("lifecycle_ppl", get_ppls(config="internal"))

        # Act
        transfer(ppl="lifecycle_ppl", type="import", mode="copy")

        # Assert
        self.assertIn("lifecycle_ppl", get_ppls(config="transfer")) # Should still be in transfer
        self.assertIn("lifecycle_ppl", get_ppls(config="internal")) # Should now be in internal
        self.assertTrue(Path("internal/Transfer/Configs/lifecycle_ppl.json").exists())
        self.assertTrue(Path("internal/Configs/lifecycle_ppl.json").exists())

if __name__ == '__main__':
    unittest.main()