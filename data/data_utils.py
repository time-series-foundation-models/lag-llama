# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import random
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from pandas.tseries.frequencies import to_offset

from data.read_new_dataset import (
    MetaData,
    TrainDatasets,
    create_train_dataset_without_last_k_timesteps,
    get_ett_dataset,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


def _count_timesteps(
    left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset
) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """
    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if isinstance(left, pd.Period):
        left = left.to_timestamp()
    if isinstance(right, pd.Period):
        right = right.to_timestamp()
    assert (
        right >= left
    ), f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )


def create_train_dataset_last_k_percentage(raw_train_dataset, freq, k=100):
    # Get training data
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        number_of_values = int(len(s_train["target"]) * k / 100)
        train_start_index = len(s_train["target"]) - number_of_values
        s_train["target"] = s_train["target"][train_start_index:]
        train_data.append(s_train)

    train_data = ListDataset(train_data, freq=freq)

    return train_data


def create_train_and_val_datasets_with_dates(
    name,
    dataset_path,
    data_id,
    history_length,
    prediction_length=None,
    num_val_windows=None,
    val_start_date=None,
    train_start_date=None,
    freq=None,
    last_k_percentage=None,
):
    """
    Train Start date is assumed to be the start of the series if not provided
    Freq is not given is inferred from the data
    We can use ListDataset to just group multiple time series - https://github.com/awslabs/gluonts/issues/695
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        raw_dataset = get_ett_dataset(name, path)
    elif name in (
        "cpu_limit_minute",
        "cpu_usage_minute",
        "function_delay_minute",
        "instances_minute",
        "memory_limit_minute",
        "memory_usage_minute",
        "platform_delay_minute",
        "requests_minute",
    ):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) is not str]
        test_data = [x for x in data["test"] if type(x["target"][0]) is not str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) is not str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(
            full_dataset, freq=metadata.freq, k=24
        )
        raw_dataset = TrainDatasets(
            metadata=metadata, train=train_ds, test=full_dataset
        )
    else:
        raw_dataset = get_dataset(name, path=Path(dataset_path))

    if prediction_length is None:
        prediction_length = raw_dataset.metadata.prediction_length
    if freq is None:
        freq = raw_dataset.metadata.freq
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    raw_train_dataset = raw_dataset.train

    if not num_val_windows and not val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")
    if num_val_windows and val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")

    max_train_end_date = None

    # Get training data
    total_train_points = 0
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = train_end_index - number_of_values
        else:
            train_start_index = 0
        s_train["target"] = series["target"][train_start_index:train_end_index]
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + to_offset(freq) * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        s_val["start"] = series["start"] + val_start_index * timestep_delta
        s_val["target"] = series["target"][val_start_index:]
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
        total_train_points
        + total_val_points
        - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
    )


def create_test_dataset(name, dataset_path, history_length, freq=None, data_id=None):
    """
    For now, only window per series is used.
    make_evaluation_predictions automatically only predicts for the last "prediction_length" timesteps
    NOTE / TODO: For datasets where the test set has more series (possibly due to more timestamps), \
    we should check if we only use the last N series where N = series per single timestamp, or if we should do something else.
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        dataset = get_ett_dataset(name, path)
    elif name in (
        "cpu_limit_minute",
        "cpu_usage_minute",
        "function_delay_minute",
        "instances_minute",
        "memory_limit_minute",
        "memory_usage_minute",
        "platform_delay_minute",
        "requests_minute",
    ):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) is not str]
        test_data = [x for x in data["test"] if type(x["target"][0]) is not str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) is not str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(
            full_dataset, freq=metadata.freq, k=24
        )
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        dataset = get_dataset(name, path=Path(dataset_path))

    if freq is None:
        freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    data = []
    total_points = 0
    for i, series in enumerate(dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": series["start"] + offset,
                    "item_id": i,
                    "data_id": data_id,
                }
            )
        else:
            series_copy = copy.deepcopy(series)
            series_copy["item_id"] = i
            series_copy["data_id"] = data_id
            data.append(series_copy)
        total_points += len(data[-1]["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points


offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}


def to_gluonts(entry):
    dataset_freq = pd.infer_freq(entry["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    return {
        "start": pd.Period(
            entry["timestamp"][0],
            freq=dataset_freq,
        ).to_timestamp(),
        "target": entry["target"],
        "item_id": entry["id"],
    }
