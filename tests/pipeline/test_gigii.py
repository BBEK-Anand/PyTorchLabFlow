# import unittest
# import os
# import shutil
# import json
# import sys
# from unittest.mock import patch, MagicMock, ANY

# import torch
# import torch.nn as nn
# import pandas as pd
# from pathlib import Path

# # This import path should match your project structure
# from PTLF.experiment import PipeLine

# # --- Dummy Components moved to the top level of the module ---
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
#         return ((torch.randn(1),), torch.randn(1))

# def dummy_optimizer(params):
#     optimizer = torch.optim.SGD(params, lr=0.1)
#     optimizer.step = MagicMock()
#     return optimizer

# def dummy_loss():
#     return nn.MSELoss()

# def dummy_metric(logits, labels):
#     return torch.tensor(1.0)


# # --- FIX: Moved the patch decorators from setUp to the class itself ---
# # This ensures the mocks are active for every test method.
# @patch('PTLF._pipeline.get_shared_data')
# @patch('PTLF._pipeline.Db')
# @patch('PTLF._pipeline.load_component')
# @patch('PTLF._pipeline.hash_args')
# @patch('PTLF._pipeline.get_invalid_loc_queries')
# @patch('PTLF._pipeline.get_caller')
# class TestPipeLine(unittest.TestCase):
#     """Unit tests for the PipeLine class, with dependencies mocked."""

#     def setUp(self, mock_get_caller, mock_get_invalid, mock_hash_args, 
#               mock_load_component, MockDb, mock_get_shared_data):
#         """Set up a temporary, isolated environment for each test."""
        
#         self.mock_settings = {
#             'data_path': 'temp_test_data_pipeline',
#             'metrics': {'accuracy': {}},
#             'strategy': {'mode': 'min', 'monitor': 'val_loss'},
#             'logid': 'test_log_001'
#         }
#         mock_get_shared_data.return_value = self.mock_settings

#         self.test_dir = Path(self.mock_settings['data_path'])
#         for subdir in ["", "Configs", "Quicks", "Histories", "Weights"]:
#             (self.test_dir / subdir).mkdir(exist_ok=True)
        
#         self.mock_db_instance = MockDb.return_value
#         self.mock_load_component = mock_load_component
#         self.mock_hash_args = mock_hash_args
#         self.mock_get_invalid = mock_get_invalid
#         self.mock_get_caller = mock_get_caller
        
#         self.mock_hash_args.return_value = 'dummy_hash_123'
#         self.mock_get_invalid.return_value = []
#         self.mock_get_caller.return_value = 'test_script.py'
#         self.mock_load_component.side_effect = self._mocked_load_component

#         self.dummy_args = {
#             'model': {'loc': 'DummyModel'}, 'loss': {'loc': 'DummyLoss'},
#             'optimizer': {'loc': 'DummyOptimizer', 'args': {}},
#             'dataset': {'loc': 'DummyDataset', 'args': {}},
#             'metrics': {'accuracy': {'loc': 'DummyMetric'}},
#             'train_data_src': 'path/train', 'val_data_src': 'path/val',
#             'train_batch_size': 2, 'val_batch_size': 2
#         }

#         self.pipeline = PipeLine()

#     def tearDown(self):
#         shutil.rmtree(self.test_dir, ignore_errors=True)

#     def _mocked_load_component(self, **kwargs):
#         loc_map = {
#             'Model': DummyModel(), 'Dataset': DummyDataset(),
#             'Optimizer': dummy_optimizer(params=DummyModel().parameters()),
#             'Loss': dummy_loss(), 'Metric': dummy_metric
#         }
#         for key, component in loc_map.items():
#             if key in kwargs.get('loc', ''):
#                 return component
#         return MagicMock()

#     def test_init_default(self):
#         self.assertIsNone(self.pipeline.pplid)
#         self.assertIsNone(self.pipeline.cnfg)
#         self.assertFalse(self.pipeline._prepared)

#     def test_get_path(self):
#         self.pipeline.pplid = "test_ppl"
#         base = self.test_dir
#         self.assertEqual(self.pipeline.get_path('config'), str(base / "Configs/test_ppl.json"))
#         self.assertEqual(self.pipeline.get_path('history'), str(base / "Histories/test_ppl.csv"))
#         self.assertEqual(self.pipeline.get_path('quick'), str(base / "Quicks/test_ppl.json"))
#         self.assertEqual(self.pipeline.get_path('weight', epoch=5), str(base / "Weights/test_ppl/test_ppl_e5.pth"))
        
#         with self.assertRaisesRegex(ValueError, "Epoch must be specified"):
#             self.pipeline.get_path('weight')

#     def test_verify_by_pplid(self):
#         self.mock_db_instance.query.return_value = [(1,)]
#         self.assertEqual(self.pipeline.verify(pplid="exists"), "exists")

#         self.mock_db_instance.query.return_value = []
#         self.assertFalse(self.pipeline.verify(pplid="does_not_exist"))

#     def test_verify_by_args(self):
#         self.mock_db_instance.query.return_value = [("existing_ppl",)]
#         result = self.pipeline.verify(args=self.dummy_args)
#         self.assertEqual(result, "existing_ppl")
#         self.mock_hash_args.assert_called_with(self.dummy_args)
    
#     @patch.object(PipeLine, '_check_args')
#     def test_new_pipeline_creation(self, mock_check_args):
#         (self.test_dir / "ppls.csv").touch()
#         self.mock_db_instance.query.return_value = []
        
#         self.pipeline.new(pplid="new_ppl", args=self.dummy_args)
        
#         mock_check_args.assert_called_once_with(self.dummy_args)
#         self.assertTrue((self.test_dir / "Configs/new_ppl.json").exists())
        
#         self.mock_db_instance.execute.assert_any_call(
#             "INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)",
#             ("new_ppl", "dummy_hash_123")
#         )

#     def test_load_pipeline(self):
#         pplid = "existing_ppl"
#         config_path = self.test_dir / "Configs" / f"{pplid}.json"
#         config_data = {'pplid': pplid, 'args': self.dummy_args}
#         with open(config_path, 'w') as f:
#             json.dump(config_data, f)

#         self.mock_db_instance.query.return_value = [(1,)]
#         self.pipeline.load(pplid=pplid)
        
#         self.assertEqual(self.pipeline.pplid, pplid)
#         self.assertEqual(self.pipeline.cnfg, config_data)

#     def test_prepare(self):
#         pplid = "prep_ppl"
#         self.pipeline.cnfg = {'pplid': pplid, 'args': self.dummy_args, 'last': {'epoch': 0}}
#         self.pipeline.pplid = pplid

#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({'best': {'val_loss': 1000.0}}, f)

#         self.pipeline.prepare()
        
#         self.assertTrue(self.pipeline._prepared)
#         self.assertIn('model', self.pipeline.comps)

#     def test_update_improves_metric(self):
#         pplid = "update_ppl"
#         self.pipeline.pplid = pplid
#         self.pipeline.cnfg = {'pplid': pplid}
#         self.pipeline.__best = 1.0
#         self.pipeline.comps['model'] = DummyModel()

#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({'last': {}, 'best': {}}, f)
        
#         epoch_data = {
#             'epoch': 1, 'val_loss': 0.5, 'train_loss': 0.8, 'train_accuracy': 0.9,
#             'val_accuracy': 0.85, 'train_duration': 10.0, 'val_duration': 2.0
#         }
        
#         is_best = self.pipeline.update(epoch_data)
        
#         self.assertTrue(is_best)
#         self.assertTrue((self.test_dir / "Weights" / pplid / f"{pplid}_e1.pth").exists())

#     @patch('PTLF._pipeline.PipeLine.should_running', new_callable=unittest.mock.PropertyMock)
#     @patch.object(PipeLine, '_forward')
#     def test_train_loop(self, mock_forward, mock_should_running):
#         mock_should_running.return_value = True
#         pplid = "train_ppl"
#         self.pipeline.pplid = pplid
#         self.pipeline._prepared = True
#         self.pipeline.cnfg = {'last': {'epoch': 3}, 'best': {'epoch': 1}}
        
#         quick_path = self.test_dir / "Quicks" / f"{pplid}.json"
#         with open(quick_path, 'w') as f:
#             json.dump({'last': {'epoch': 3}, 'best': {'epoch': 1, 'val_loss': 1000}}, f)
        
#         self.mock_db_instance.query.return_value = []
#         mock_forward.return_value = {'loss': 0.5, 'accuracy': 0.9, 'duration': 1}
        
#         with patch.object(self.pipeline, 'update', return_value=False) as spy_update:
#             self.pipeline.train(num_epochs=5, self_patience=3)
#             self.assertEqual(mock_forward.call_count, 6)
#             self.assertEqual(spy_update.call_count, 3)

#     def test_running_methods(self):
#         pplid = "running_ppl"
#         self.pipeline.pplid = pplid

#         self.mock_db_instance.query.return_value = []
#         self.assertFalse(self.pipeline.is_running())
#         self.assertTrue(self.pipeline.should_running)

#         self.mock_db_instance.query.return_value = [('log123',)]
#         self.assertEqual(self.pipeline.is_running(), 'log123')
        
#         self.mock_db_instance.query.side_effect = [[('log123',)], [('stop',)]]
#         self.pipeline.stop_running()
#         self.mock_db_instance.execute.assert_called_with(
#             "UPDATE runnings SET parity = ? WHERE logid = ?", ('stop', 'log123')
#         )
#         self.assertFalse(self.pipeline.should_running)

#     def test_save_config_raises_error_on_mismatch(self):
#         self.pipeline.pplid = "test_ppl"
#         self.pipeline.cnfg = {'pplid': "test_ppl", 'args': self.dummy_args}
        
#         self.mock_db_instance.query.return_value = [("a_different_ppl",)]
        
#         with self.assertRaisesRegex(ValueError, "can not save config"):
#             self.pipeline._save_config()

# if __name__ == '__main__':
#     unittest.main()

