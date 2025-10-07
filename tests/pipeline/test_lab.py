# import unittest
# import os
# import shutil
# import json
# import sqlite3
# from pathlib import Path
# from unittest.mock import patch, MagicMock

# # Adjust this import to match your project structure
# from PTLF.lab import create_project, setup_databases, lab_setup, export_settigns

# class TestLabFunctions(unittest.TestCase):
#     """Unit tests for the lab setup and management functions."""

#     def setUp(self):
#         """Set up a temporary directory for test artifacts."""
#         self.test_dir = Path("./temp_test_lab_project")
#         self.test_dir.mkdir(exist_ok=True)
#         self.settings = {
#             "project_name": "TestLab",
#             "project_dir": str(self.test_dir.resolve()),
#             "component_dir": str((self.test_dir / "components").resolve()),
#         }

#     def tearDown(self):
#         """Clean up the temporary directory after each test."""
#         shutil.rmtree(self.test_dir)
#         # Clean up any stray sys.path modifications if necessary
#         if 'PTLF.context' in locals():
#             if self.settings["component_dir"] in sys.path:
#                 sys.path.remove(self.settings["component_dir"])


#     @patch('PTLF.lab.setup_databases')
#     @patch('PTLF.lab.set_shared_data')
#     def test_create_project(self, mock_set_shared_data, mock_setup_databases):
#         """
#         Test that create_project correctly builds the directory structure and saves settings.
#         """
#         # --- Act ---
#         settings_path_str = create_project(self.settings)
#         settings_path = Path(settings_path_str)

#         # --- Assert ---
#         # Verify paths and directories
#         project_data_path = self.test_dir / "TestLab"
#         self.assertTrue(project_data_path.is_dir())
#         self.assertTrue((project_data_path / "Configs").is_dir())
#         self.assertTrue((project_data_path / "Archived" / "Configs").is_dir())
#         self.assertTrue((project_data_path / "Transfer" / "Configs").is_dir())
#         self.assertTrue((self.test_dir / "components").is_dir())
#         self.assertEqual(settings_path.name, "TestLab.json")

#         # Verify the content of the saved settings file
#         with open(settings_path, 'r') as f:
#             saved_settings = json.load(f)
        
#         self.assertEqual(saved_settings['project_name'], "TestLab")
#         self.assertTrue(os.path.isabs(saved_settings['project_dir']))
#         self.assertEqual(saved_settings['data_path'], str(project_data_path.resolve()))
        
#         # Verify that dependencies were called correctly
#         mock_setup_databases.assert_called_once()
#         mock_set_shared_data.assert_called_once()


#     def test_setup_databases(self):
#         """
#         Test that setup_databases creates all required DB files and tables correctly.
#         """
#         # --- Arrange ---
#         # Update settings with a data_path for this test
#         data_path = self.test_dir / "TestLabData"
#         data_path.mkdir()
#         self.settings['data_path'] = str(data_path)
        
#         # --- Act ---
#         setup_databases(self.settings)

#         # --- Assert ---
#         # 1. Check if all database files exist
#         logs_db_path = data_path / "logs.db"
#         ppls_db_path = data_path / "ppls.db"
#         archived_db_path = data_path / "Archived" / "ppls.db"
        
#         self.assertTrue(logs_db_path.exists())
#         self.assertTrue(ppls_db_path.exists())
#         self.assertTrue(archived_db_path.exists())

#         # 2. Connect and verify schemas
#         # Verify logs.db
#         with sqlite3.connect(logs_db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs';")
#             self.assertIsNotNone(cursor.fetchone())
#             cursor.execute("SELECT logid FROM logs WHERE logid='log0';")
#             self.assertEqual(cursor.fetchone()[0], 'log0')

#         # Verify ppls.db
#         with sqlite3.connect(ppls_db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             tables = {row[0] for row in cursor.fetchall()}
#             self.assertEqual(tables, {'ppls', 'edges', 'runnings'})

#         # Verify Archived/ppls.db
#         with sqlite3.connect(archived_db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ppls';")
#             self.assertIsNotNone(cursor.fetchone())


#     @patch('PTLF.lab.register_libs_path')
#     @patch('PTLF.lab.set_shared_data')
#     @patch('PTLF.lab.Db')
#     def test_lab_setup(self, MockDb, mock_set_shared_data, mock_register_libs_path):
#         """
#         Test that lab_setup correctly reads settings, logs a new session, and configures context.
#         """
#         # --- Arrange ---
#         # Create a dummy settings file and a dummy logs.db
#         settings_path = self.test_dir / "settings.json"
#         db_path = self.test_dir / "logs.db"
        
#         self.settings['data_path'] = str(self.test_dir)
#         with open(settings_path, 'w') as f:
#             json.dump(self.settings, f)
        
#         # Mock the Db class to control its behavior
#         mock_db_instance = MockDb.return_value
#         # Simulate that there is 1 existing log entry ('log0')
#         mock_cursor = MagicMock()
#         mock_cursor.fetchone.return_value = [1] 
#         mock_db_instance.execute.return_value = mock_cursor

#         # --- Act ---
#         lab_setup(str(settings_path))

#         # --- Assert ---
#         # Verify it tries to connect to the correct DB
#         MockDb.assert_called_with(db_path=str(db_path))

#         # Verify it correctly counts existing logs and inserts a new one
#         mock_db_instance.execute.assert_any_call("SELECT COUNT(*) FROM logs")
        
#         # It should insert the *next* log entry, which is 'log1'
#         mock_db_instance.execute.assert_any_call(
#             "INSERT INTO logs (logid, called_at) VALUES (?, ?)",
#             ('log1', ANY) # ANY allows us to ignore the dynamic get_caller() result
#         )

#         # Verify context setup functions are called
#         mock_set_shared_data.assert_called_once_with(self.settings, 'log1')
#         mock_register_libs_path.assert_called_once_with(self.settings["component_dir"])


#     def test_lab_setup_raises_error_for_missing_file(self):
#         """
#         Test that lab_setup raises a ValueError if the settings file doesn't exist.
#         """
#         with self.assertRaises(ValueError):
#             lab_setup("non_existent_path.json")


#     @patch('PTLF.lab.get_shared_data')
#     def test_export_settings(self, mock_get_shared_data):
#         """
#         Test that export_settings writes the settings to the correct parent directory.
#         """
#         # --- Arrange ---
#         data_path = self.test_dir / "TestLabData"
#         data_path.mkdir()
        
#         mock_settings = {
#             "project_name": "ExportTest",
#             "data_path": str(data_path),
#             "some_other_key": "value"
#         }
#         mock_get_shared_data.return_value = mock_settings
        
#         # --- Act ---
#         exported_path_str = export_settigns()
#         exported_path = Path(exported_path_str)

#         # --- Assert ---
#         # Expected path is in the parent of data_path
#         expected_path = self.test_dir / "ExportTest.json"
#         self.assertEqual(exported_path.resolve(), expected_path.resolve())
#         self.assertTrue(exported_path.exists())

#         # Check content
#         with open(exported_path, 'r') as f:
#             content = json.load(f)
#         self.assertEqual(content, mock_settings)


# # if __name__ == '__main__':
# #     unittest.main()