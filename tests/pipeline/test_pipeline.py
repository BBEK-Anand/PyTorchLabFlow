# import unittest
# import os
# import shutil
# import json
# from unittest.mock import patch, MagicMock, ANY

# import torch
# import torch.nn as nn
# import pandas as pd
# from pathlib import Path

# # 1. IMPORTANT: Adjust this import to match your project's structure
# from PTLF.experiment import PipeLine

# # --- Dummy Components for Testing ---
# # These simulate the objects that your 'load_component' function would return.
# class DummyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#     def forward(self, x):
#         return self.linear(x)

# class DummyDataset(torch.utils.data.Dataset):
#     def __len__(self):
#         return 10
#     def __getitem__(self, idx):
#         # The forward pass expects a tuple of inputs and a single tensor for labels
#         return ((torch.randn(1),), torch.randn(1))

# def dummy_optimizer(params):
#     # A mock optimizer that has a 'step' method
#     optimizer = torch.optim.SGD(params, lr=0.1)
#     optimizer.step = MagicMock()
#     return optimizer

# def dummy_loss():
#     return nn.MSELoss()

# def dummy_metric(logits, labels):
#     return torch.tensor(1.0)


# # In tests/pipeline/test_pipeline.py

# # ... (imports)

# class TestPipeLine(unittest.TestCase):
    
#     # ... (dummy classes)

#     # 👇 FIX: Correct the patch paths to point where the objects are *used* (in the _pipeline module)
#     @patch('PTLF._pipeline.get_caller')
#     @patch('PTLF._pipeline.get_invalid_loc_queries')
#     @patch('PTLF._pipeline.hash_args')
#     @patch('PTLF._pipeline.load_component')
#     @patch('PTLF._pipeline.Db')
#     @patch('PTLF._pipeline.get_shared_data')
#     def setUp(self, mock_get_shared_data, MockDb, mock_load_component,
#               mock_hash_args, mock_get_invalid, mock_get_caller):
#         """Set up a temporary environment and mocks for each test."""
        
#         # Mock the global settings FIRST
#         self.mock_settings = {
#             'data_path': 'temp_test_data_pipeline', # Use a relative path managed by the test
#             'metrics': {'accuracy': {}},
#             'strategy': {'mode': 'min', 'monitor': 'val_loss'},
#             'logid': 'test_log_001'
#         }
#         mock_get_shared_data.return_value = self.mock_settings

#         # NOW, create the directory that the Db class will look for
#         self.test_dir = Path(self.mock_settings['data_path'])
#         self.test_dir.mkdir(exist_ok=True)
#         (self.test_dir / "Configs").mkdir(exist_ok=True)
#         (self.test_dir / "Quicks").mkdir(exist_ok=True)
#         (self.test_dir / "Histories").mkdir(exist_ok=True)
#         (self.test_dir / "Weights").mkdir(exist_ok=True)
#         # Assign mocks from arguments
#         self.mock_db_instance = MockDb.return_value
#         self.mock_load_component = mock_load_component
#         self.mock_hash_args = mock_hash_args
#         self.mock_get_invalid = mock_get_invalid
#         self.mock_get_caller = mock_get_caller
        
#         # Configure return values for mocks
#         self.mock_hash_args.return_value = 'dummy_hash_123'
#         self.mock_get_invalid.return_value = []
#         self.mock_get_caller.return_value = 'test_script.py'

#         # Dummy args for reuse
#         self.dummy_args = {
#             'model': {'loc': 'DummyModel'}, 'loss': {'loc': 'DummyLoss'},
#             'optimizer': {'loc': 'DummyOptimizer', 'args': {}},
#             'dataset': {'loc': 'DummyDataset', 'args': {}},
#             'metrics': {'accuracy': {'loc': 'DummyMetric'}},
#             'train_data_src': 'path/train', 'val_data_src': 'path/val',
#             'train_batch_size': 2, 'val_batch_size': 2
#         }

#         # The PipeLine instance to be tested - this should now succeed
#         self.pipeline = PipeLine()
        
#         # Configure side effect for load_component mock
#         self.mock_load_component.side_effect = self._mocked_load_component

#     def tearDown(self):
#         """Clean up the temporary environment after each test."""
#         shutil.rmtree(self.test_dir)

#     # ... (rest of your test methods are fine and don't need changes) ...

#     def _mocked_load_component(self, **kwargs):
#         """Helper to return dummy components based on the 'loc' argument."""
#         loc = kwargs.get('loc')
#         if 'Model' in loc: return DummyModel()
#         if 'Dataset' in loc: return DummyDataset()
#         if 'Optimizer' in loc: return dummy_optimizer(params=DummyModel().parameters())
#         if 'Loss' in loc: return dummy_loss()
#         if 'Metric' in loc: return dummy_metric
#         return MagicMock()

#     # --- Test Cases for Each Method ---

#     def test_init_default(self):
#         """Test default initialization."""
#         self.assertIsNone(self.pipeline.pplid)
#         self.assertIsNone(self.pipeline.cnfg)
#         self.assertFalse(self.pipeline._prepared)
#         self.assertIsNotNone(self.pipeline.settings)

#     # @patch.object(PipeLine, 'load')
#     # def test_init_with_pplid(self, mock_load):
#     #     """Test initialization with a pplid, which should trigger load."""
#     #     PipeLine(pplid="existing_ppl_123")
#     #     mock_load.assert_called_once_with(pplid="existing_ppl_123")
    
#     def test_get_path(self):
#         """Test path generation for different artifact types."""
#         self.pipeline.pplid = "test_ppl"
#         base = self.test_dir
#         self.assertEqual(self.pipeline.get_path('config'), str(base / "Configs/test_ppl.json"))
#         self.assertEqual(self.pipeline.get_path('history'), str(base / "Histories/test_ppl.csv"))
#         self.assertEqual(self.pipeline.get_path('quick'), str(base / "Quicks/test_ppl.json"))
#         self.assertEqual(self.pipeline.get_path('weight', epoch=5), str(base / "Weights/test_ppl/test_ppl_e5.pth"))
        
#         with self.assertRaises(ValueError):
#             self.pipeline.get_path('weight') # Should fail without epoch

#     def test_verify_by_pplid(self):
#         """Test verification of a pipeline by its ID."""
#         self.mock_db_instance.query.return_value = [(1,)]
#         self.assertEqual(self.pipeline.verify(pplid="exists"), "exists")
#         self.mock_db_instance.query.assert_called_with("SELECT 1 FROM ppls WHERE pplid = ? LIMIT 1", ("exists",))

#         self.mock_db_instance.query.return_value = []
#         self.assertFalse(self.pipeline.verify(pplid="does_not_exist"))

#     def test_verify_by_args(self):
#         """Test verification of a pipeline by its arguments hash."""
#         self.mock_db_instance.query.return_value = [("existing_ppl",)]
#         result = self.pipeline.verify(args=self.dummy_args)
#         self.assertEqual(result, "existing_ppl")
#         self.mock_hash_args.assert_called_with(self.dummy_args)
    
#     @patch.object(PipeLine, '_check_args', return_value=None)
#     def test_new_pipeline_creation(self, mock_check_args):
#         """Test the 'new' method to create a pipeline."""
#         self.mock_db_instance.query.side_effect = [
#                 [],              # First call to verify(pplid=...) returns nothing (good)
#                 [('new_ppl',)]   # Second call to verify(args=...) returns the new pplid
#             ]
        
#         self.pipeline.new(pplid="new_ppl", args=self.dummy_args)
        
#         mock_check_args.assert_called_once_with(self.dummy_args)
#         self.assertTrue((self.test_dir / "Configs/new_ppl.json").exists())
#         self.assertTrue((self.test_dir / "Histories/new_ppl.csv").exists())
#         self.assertTrue((self.test_dir / "Quicks/new_ppl.json").exists())
        
#         self.mock_db_instance.execute.assert_called_with(
#             "INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)",
#             ("new_ppl", "dummy_hash_123")
#         )

#     def test_load_pipeline(self):
#         """Test loading an existing pipeline's configuration."""
#         pplid = "existing_ppl"
#         config_path = self.test_dir / "Configs" / f"{pplid}.json"
#         config_data = {'pplid': pplid, 'args': self.dummy_args}
#         with open(config_path, 'w') as f:
#             json.dump(config_data, f)

#         self.mock_db_instance.query.return_value = [(1,)] # Simulate it exists
#         self.pipeline.load(pplid=pplid)
        
#         self.assertEqual(self.pipeline.pplid, pplid)
#         self.assertEqual(self.pipeline.cnfg, config_data)

#     def test_prepare(self):
#         """Test the prepare method for component initialization."""
#         pplid = "prep_ppl"
#         self.pipeline.cnfg = {'pplid': pplid, 'args': self.dummy_args, 'last': {'epoch': 0}}
#         self.pipeline.pplid = pplid

#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({'best': {'val_loss': 1000.0}}, f)

#         self.pipeline.prepare()
        
#         self.assertTrue(self.pipeline._prepared)
#         self.assertIn('model', self.pipeline.comps)
#         self.assertIsNotNone(self.pipeline.trainDataLoader)
#         self.mock_load_component.assert_any_call(**self.dummy_args['model'])
    
#     def test_update(self):
#         """Test the update method after a training epoch."""
#         pplid = "update_ppl"
#         self.pipeline.pplid = pplid
#         self.pipeline.cnfg = {'pplid': pplid}
#         self.pipeline._prepared = True
#         self.pipeline.__best = 1.0
        
#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({'last': {}, 'best': {}}, f)
        
#         epoch_data = {
#             'epoch': 1, 'val_loss': 0.5, 'train_loss': 0.8,
#             'train_accuracy': 0.9, 'val_accuracy': 0.85,
#             'train_duration': 10.0, 'val_duration': 2.0
#         }
        
#         is_best = self.pipeline.update(epoch_data)
        
#         self.assertTrue(is_best)
#         self.assertTrue((self.test_dir / "Weights" / pplid / f"{pplid}_e1.pth").exists())
        
#         with open(quick_path, 'r') as f:
#             quick_data = json.load(f)
#             self.assertEqual(quick_data['best']['val_loss'], 0.5)

#     @patch.object(PipeLine, '_forward')
#     def test_train_loop(self, mock_forward):
#         """Test the main training loop logic by mocking the _forward pass."""
#         pplid = "train_ppl"
#         self.pipeline.pplid = pplid
#         self.pipeline._prepared = True
#         self.pipeline.cnfg = {'last': {'epoch': 0}, 'best': {'epoch': 0}}
#         self.pipeline.P = MagicMock() # Mock the 'P' attribute for should_running
#         self.pipeline.P.should_running = True
        
#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({
#                 'last': {'epoch': 0}, 
#                 'best': {'epoch': 0, 'val_loss': 1000}
#             }, f)
        
#         self.mock_db_instance.query.return_value = []
#         mock_forward.return_value = {'loss': 0.5, 'accuracy': 0.9, 'duration': 1}
        
#         with patch.object(self.pipeline, 'update', return_value=True) as spy_update:
#             self.pipeline.train(num_epochs=2, verbose=['accuracy'])
#             self.assertEqual(mock_forward.call_count, 4) # 2 epochs * (train + valid)
#             self.assertEqual(spy_update.call_count, 2)

#     def test_running_methods(self):
#         """Test is_running, should_running, and stop_running."""
#         pplid = "running_ppl"
#         self.pipeline.pplid = pplid

#         # Not running
#         self.mock_db_instance.query.return_value = []
#         self.assertFalse(self.pipeline.is_running())
#         self.assertTrue(self.pipeline.should_running)

#         # Running
#         self.mock_db_instance.query.return_value = [('log123',)]
#         self.assertEqual(self.pipeline.is_running(), 'log123')
        
#         # Flagged to stop
#         self.mock_db_instance.query.side_effect = [
#             [('log123',)],   # for is_running
#             [('stop',)]      # for should_running
#         ]
#         self.pipeline.stop_running()
#         self.mock_db_instance.execute.assert_called_with(
#             "UPDATE runnings SET parity = ? WHERE logid = ?", ('stop', 'log123')
#         )
#         self.assertFalse(self.pipeline.should_running)

# if __name__ == '__main__':
#     unittest.main()