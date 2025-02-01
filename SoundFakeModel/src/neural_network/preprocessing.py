from backend.Isolate import isolate_file
import pandas as pd
from typing import Dict, Any
from functools import partial


def pre_isolate(audio_file_path):
    """
    Pre-process the audio file to isolate the voice,
    make training more efficient
    """
    isolated_path = isolate_file(audio_file_path, mp3=True)
    return isolated_path

from enum import Enum
class Classification(Enum):
    FAKE = 'fake'
    REAL = 'real'
class PreprocessedData:
    class _internal_csv_manager:
        @staticmethod
        def create_csv(csv_file_path: str, columns: list):
            """
            Create the CSV file with specified columns.
            """
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_file_path, index=False)

        @staticmethod
        def append_row(csv_file_path: str, row: Dict[str, Any]):
            """
            Append data to the CSV file.
            row: dict of processed_data key=col name: value = data to store
            """
            df = pd.read_csv(csv_file_path)
            df.to_csv(csv_file_path, mode='a', header=False, index=False)

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
        def search_get(csv_file_path: str, column: str, value: Any) -> list:
            """
            Gets all rows that match the query.
            """
            df = pd.read_csv(csv_file_path)
            matching_rows = df[df[column] == value]
            return matching_rows.to_dict(orient='records')

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
        self._internal_csv_manager.create_csv(csv_file_path, self._data_row.headers)
        self.read_row = self.read_row = partial(self._internal_csv_manager.get_row, self.csv_file_path)
        self.search = partial(self._internal_csv_manager.search_get, self.csv_file_path)
        self.delete_row = partial(self._internal_csv_manager.delete_row, self.csv_file_path)

    def add_data(self, file_path, separated_file_path, label, results):
        """
        Add data to the csv file
        """
        data = {
            "File Path": file_path,
            "Separated File Path": separated_file_path,
            "Label": label,
            "Results": results
        }
        self._internal_csv_manager.append_row(self.csv_file_path, data)
    


    class _data_row:
        headers = ["File Path", "Separated File Path", "Label", "Results"]
        from enum import Enum

        def create_row(file_path : str, separated_file_path : str, label : Classification, results):
            return {
                "File Path": file_path,
                "Separated File Path": separated_file_path,
                "Label": label,
                "Results": results
            }

