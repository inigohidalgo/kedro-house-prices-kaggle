import os
import tempfile
import logging
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger()


class KaggleDataSet:
    def __init__(self, credentials):
        required_keys = ["user", "key"]
        for key in required_keys:
            if key not in credentials:
                raise KeyError(f"KaggleDataSet requires key {key} in credentials")
            else:
                kaggle_env_key = f"KAGGLE_{key}"
                log.debug(f"Registering {kaggle_env_key} environment variable")
                os.environ[kaggle_env_key] = credentials[key]
        log.debug("Importing KaggleApi from kaggle_api_extended")
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()

        self.credentials = credentials

    def load_competition_dataset(self, competition_name, dataset_name, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self._download_single_competition_file(
                competition_name, dataset_name, temp_dir
            )
            if dataset_name.endswith(".csv"):
                dataset = pd.read_csv(dataset_path, **kwargs)
            elif dataset_name.endswith(".xlsx"):
                dataset = pd.read_excel(dataset_path, **kwargs)
            elif dataset_name.endswith(".json"):
                dataset = pd.read_json(dataset_path, **kwargs)

            return dataset

    def _download_all_competition_files(
        self, competition_name, path=None
    ) -> List[Path]:

        competition_files = self.api.competition_list_files(competition_name)
        if path is None:
            path = Path()
        else:
            path = Path(path)
        self.api.competition_download_files(competition_name, path, quiet=True)
        log.debug(f"Files {', '.join(competition_files)} downloaded to {path}")
        return [path / file_name for file_name in competition_files]

    def _download_single_competition_file(
        self, competition_name, file_name, path=None
    ) -> Path:
        if path is None:
            path = Path()
        else:
            path = Path(path)
        self.api.competition_download_file(
            competition_name, file_name, path, quiet=True
        )
        log.debug(f"File {file_name} downloaded to {path}")

        return path / file_name
