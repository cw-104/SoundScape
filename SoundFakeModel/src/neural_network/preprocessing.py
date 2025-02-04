from backend.Isolate import isolate_file
import pandas as pd
from typing import Dict, Any
from functools import partial
from enum import Enum
import os

headers = ["File Path", "Separated File Path", "Label", "Results"]
REAL_LABLE = 'real'
FAKE_LABLE = 'fake'

class PreprocessedData:
    
    @staticmethod
    def create_row(file_path : str, separated_file_path : str, label : str, results):
        return {
            "File Path": file_path,
            "Separated File Path": separated_file_path,
            "Label": label,
            "Results": results
        }
    
    @staticmethod
    def create_unprocessed_row(file_path : str, label : str):
        return {
            "File Path": file_path,
            "Separated File Path": None,
            "Label": label,
            "Results": None
        }

    class _internal_csv_manager:
        @staticmethod
        def create_csv(csv_file_path: str, columns: list):
            """
            Create the CSV file with specified columns. If not already exists
            """
            if os.path.exists(csv_file_path):
                return
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_file_path, index=False)

        @staticmethod
        def append_rows(csv_file_path: str, rows: [Dict[str, Any]]):
            """
            Append data to the CSV file.
            rows: list of dict of processed_data key=col name: value = data to store
            """
            new_rows = pd.DataFrame(rows) 
            new_rows.to_csv(csv_file_path, mode='a', header=False, index=False)
        

        @staticmethod
        def get_row(csv_file_path: str, row_index: int) -> Dict[str, Any]:
            """
            Read a specific row from the CSV file by index.
            """
            df = pd.read_csv(csv_file_path)
            if row_index < len(df):
                return df.iloc[row_index].to_dict()
            else:
                raise IndexError("Row index out of range.")


        @staticmethod
        def search_get(csv_file_path: str, column_name: str, value: Any) -> list:
            """
            Gets all rows that match the query.
            """
            df = pd.read_csv(csv_file_path)
            results = []
            for index, row in df.iterrows():
                if row[column_name] == value:
                    row = row.to_dict()
                    row['index']  = index
                    results.append(row)
            return results

        @staticmethod
        def delete_row(csv_file_path: str, row_index: int):
            """
            Deletes a row from the CSV file by index.
            """
            df = pd.read_csv(csv_file_path)
            if row_index < len(df):
                df = df.drop(index=row_index)
                df.to_csv(csv_file_path, index=False)
            else:
                raise IndexError("Row index out of range.")
    def __init__(self, csv_file_path):
        """
        manage the saving and recalling of preprocessed data
        save to csv file
        | File Path | | Separated File Path | Label | results |
        """
        self.csv_file_path = csv_file_path
        self._internal_csv_manager.create_csv(csv_file_path, headers)
        self.read_row = self.read_row = partial(self._internal_csv_manager.get_row, self.csv_file_path)
        self.search = lambda column_name, value: self._internal_csv_manager.search_get(self.csv_file_path, column_name=column_name, value=value)
        self.delete_row = partial(self._internal_csv_manager.delete_row, self.csv_file_path)

    def add_entries(self, rows):
        """
        Add data to the csv file
        """
        self._internal_csv_manager.append_rows(self.csv_file_path, rows)

    def add_entry(self, row):
        """
        Add data to the csv file
        """
        self._internal_csv_manager.append_rows(self.csv_file_path, [row])
    

    def overwrite_row(self, row_index, file_path, separated_file_path, label, results):
        """
        Overwrite a row in the csv file
        """
        data = {
            "File Path": file_path,
            "Separated File Path": separated_file_path,
            "Label": label,
            "Results": results
        }
        self._internal_csv_manager.delete_row(self.csv_file_path, row_index)
        self._internal_csv_manager.append_row(self.csv_file_path, data)
        
    def contains(self, file_path):
        """
        Check if the file path is in the csv file
        """
        results = self.search("File Path", file_path)
        return len(results) > 0    
