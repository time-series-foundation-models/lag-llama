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

import json
import os
import random
import warnings
from hashlib import sha1
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, interleave_datasets

from gluonts.dataset.field_names import FieldName


from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    LeavesMissingValues,
    LastValueImputation,
    MissingValueImputation,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.time_feature import time_features_from_frequency_str

from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.dataset_list import CHRONOS_TRAINING_DATASETS, CHRONOS_TRAINING_DATASET_SIZE
from helpers.utils import set_seed
from lag_llama.gluon.estimator import LagLlamaEstimator

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


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

    # If there's no "target" column, randomly select a float32 column
    if "target" not in entry:
        float32_columns = [
            col
            for col in entry.keys()
            if isinstance(entry[col], (list, np.ndarray)) and col != "timestamp"
        ]
        if float32_columns:
            target_column = random.choice(float32_columns)
            target = entry[target_column]
        else:
            raise ValueError("No suitable float32 column found for target")
    else:
        target = entry["target"]

    return {
        "start": pd.Period(entry["timestamp"][0], freq=dataset_freq),
        "target": target,
        "item_id": entry["id"],
    }


class ChronosDataset:
    """
    Dataset wrapper, using transforms to turn data from a time series
    into a gluonts-compatible dataset list.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    transformation:
        The estimator's transformation
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        path: str,
        datasets: list,
        transformation,
        lags_seq,
        probabilities: Optional[List[float]] = None,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "causal",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        num_shards: int = 64,
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        datasets = [load_dataset(path, dataset, split="train") for dataset in datasets]
        for dataset in datasets:
            dataset.with_format("numpy")

        if probabilities is None:
            # use the CHRONOS_TRAINING_DATASETS normalized by the sum
            self.probabilities = [
                size / sum(CHRONOS_TRAINING_DATASET_SIZE)
                for size in CHRONOS_TRAINING_DATASET_SIZE
            ]

        else:
            assert len(probabilities) == len(datasets)
            self.probabilities = probabilities

        datasets = interleave_datasets(datasets, probabilities=probabilities)
        datasets = datasets.to_iterable_dataset(num_shards=num_shards)
        datasets = datasets.map(to_gluonts)
        self.datasets = datasets.select_columns(
            [FieldName.ITEM_ID, "timestamp", FieldName.TARGET, FieldName.START]
        )

        self.mode = mode
        self.min_past = min_past or prediction_length
        self.transformation = transformation
        self.lags_seq = lags_seq

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0

        self.model_type = model_type
        self.imputation_method = (
            imputation_method or LeavesMissingValues()
            if model_type == "seq2seq"
            else LastValueImputation()
        )
        self.np_dtype = np_dtype

    def to_gluonts(entry):
        dataset_freq = pd.infer_freq(entry["timestamp"])
        dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

        # If there's no "target" column, randomly select a float32 column
        if "target" not in entry or entry["target"] is None:
            float32_columns = [
                col
                for col in entry.keys()
                if isinstance(entry[col], (list, np.ndarray)) and col != "timestamp"
            ]
            if float32_columns:
                target_column = random.choice(float32_columns)
                target = entry[target_column]
            else:
                raise ValueError("No suitable float32 column found for target")
        else:
            target = entry["target"]
        return {
            FieldName.START: pd.Period(entry["timestamp"][0], freq=dataset_freq),
            FieldName.TARGET: np.asarray(target, dtype=np.float32),
            FieldName.ITEM_ID: entry["id"],
            "timestamp": np.asarray(entry["timestamp"]),
        }

    def _create_instance_splitter(
        self,
        mode: str = "training",
        past_length=1024,
        prediction_length=1,
        min_past=None,
    ):
        assert mode in ["training", "test", "validation"]
        if min_past is None:
            min_past = prediction_length

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=min_past,
                min_future=prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=prediction_length,
            time_series_fields=["timestamp"],
            dummy_value=np.nan,
        )

    def add_features(entry, time_features):
        entry[f"past_{FieldName.TIME_FEAT}"] = (
            np.vstack(
                [
                    feat(pd.to_datetime(entry["past_timestamp"]))
                    for feat in time_features
                ]
            )
            .astype(np.float32)
            .T
        )

        entry[f"future_{FieldName.TIME_FEAT}"] = (
            np.vstack(
                [
                    feat(pd.to_datetime(entry["future_timestamp"]))
                    for feat in time_features
                ]
            )
            .astype(np.float32)
            .T
        )

        past_nan = np.isnan(entry[f"past_{FieldName.TARGET}"])
        future_nan = np.isnan(entry[f"future_{FieldName.TARGET}"])
        entry[f"past_{FieldName.OBSERVED_VALUES}"] = np.invert(past_nan)
        entry[f"future_{FieldName.OBSERVED_VALUES}"] = np.invert(future_nan)
        entry[f"past_{FieldName.TARGET}"][past_nan] = 0.0
        entry[f"future_{FieldName.TARGET}"][future_nan] = 0.0

        return entry

    def create_training_data(self, data):
        split_transform = self._create_instance_splitter(
            "training",
            past_length=self.context_length + max(self.lags_seq),
            prediction_length=self.prediction_length,
        )

        def split(entry, split_transform):
            return next(iter(split_transform.apply([entry], is_train=True)))

        datasets = self.datasets.map(
            split, fn_kwargs={"split_transform": split_transform}
        )
        time_features = time_features_from_frequency_str("s")
        datasets = datasets.map(
            self.add_features, fn_kwargs={"time_features": time_features}
        )
        return datasets.select_columns(
            [
                "past_target",
                "future_target",
                "past_time_feat",
                "future_time_feat",
                "past_observed_values",
                "future_observed_values",
            ]
        )

    def create_test_data(self, data):
        split_transform = self._create_instance_splitter(
            "test",
            past_length=self.context_length + max(self.lags_seq),
            prediction_length=self.prediction_length,
        )

        def split(entry, split_transform):
            return next(iter(split_transform.apply([entry], is_train=False)))

        datasets = self.datasets.map(
            split, fn_kwargs={"split_transform": split_transform}
        )
        time_features = time_features_from_frequency_str("s")
        datasets = datasets.map(
            self.add_features, fn_kwargs={"time_features": time_features}
        )
        return datasets.select_columns(
            [
                "past_target",
                "future_target",
                "past_time_feat",
                "future_time_feat",
                "past_observed_values",
                "future_observed_values",
            ]
        )

    def create_validation_data(self, data):
        split_transform = self._create_instance_splitter(
            "validation",
            past_length=self.context_length + max(self.lags_seq),
            prediction_length=self.prediction_length,
        )

        def split(entry, split_transform):
            return next(iter(split_transform.apply([entry], is_train=True)))

        datasets = self.datasets.map(
            split, fn_kwargs={"split_transform": split_transform}
        )
        time_features = time_features_from_frequency_str("s")
        datasets = datasets.map(
            self.add_features, fn_kwargs={"time_features": time_features}
        )
        return datasets.select_columns(
            [
                "past_target",
                "future_target",
                "past_time_feat",
                "future_time_feat",
                "past_observed_values",
                "future_observed_values",
            ]
        )


def train_model(
    training_network,
    trainer_kwargs,
    training_data,
    batch_size,
    validation_data=None,
    ckpt_path: Optional[str] = None,
):
    monitor = "train_loss" if validation_data is None else "val_loss"
    checkpoint = ModelCheckpoint(monitor=monitor, mode="min", verbose=True)
    custom_callbacks = trainer_kwargs.pop("callbacks", [])

    trainer = L.Trainer(
        **{
            "accelerator": "auto",
            "callbacks": [checkpoint] + custom_callbacks,
            "log_every_n_steps": 1,
            "limit_val_batches": 0.01,
            **trainer_kwargs,
        }
    )

    training_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    trainer.fit(
        model=training_network,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
        ckpt_path=ckpt_path,
    )

    return training_network, trainer


def train(args):
    # Set seed
    set_seed(args.seed)
    L.seed_everything(args.seed)

    # # Print GPU stats
    # print_gpu_stats()

    # Create a directory to store the results in
    # This string is made independent of hyperparameters here, as more hyperparameters / arguments may be added later
    # The name should be created in the calling bash script
    # This way, when that same script is executed again, automatically the model training is resumed from a checkpoint if available
    experiment_name = args.experiment_name
    fulldir_experiments = os.path.join(
        args.results_dir, experiment_name, str(args.seed)
    )
    if os.path.exists(fulldir_experiments):
        print(fulldir_experiments, "already exists.")
    os.makedirs(fulldir_experiments, exist_ok=True)

    # Create directory for checkpoints
    checkpoint_dir = os.path.join(fulldir_experiments, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    elif args.get_ckpt_path_from_experiment_name:
        fulldir_experiments_for_ckpt_path = os.path.join(
            args.results_dir, args.get_ckpt_path_from_experiment_name, str(args.seed)
        )
        full_experiment_name_original = (
            args.get_ckpt_path_from_experiment_name + "-seed-" + str(args.seed)
        )
        experiment_id_original = sha1(
            full_experiment_name_original.encode("utf-8")
        ).hexdigest()[:8]
        checkpoint_dir_wandb = os.path.join(
            fulldir_experiments_for_ckpt_path,
            "lag-llama",
            experiment_id_original,
            "checkpoints",
        )
        file = os.listdir(checkpoint_dir_wandb)[-1]
        if file:
            ckpt_path = os.path.join(checkpoint_dir_wandb, file)
        if not ckpt_path:
            raise Exception("ckpt_path not found from experiment name")
        # Delete the EarlyStoppingCallback and save it in the current checkpoint_dir
        new_ckpt_path = checkpoint_dir + "/pretrained_ckpt.ckpt"
        print("Moving", ckpt_path, "to", new_ckpt_path)
        ckpt_loaded = torch.load(ckpt_path)
        del ckpt_loaded["callbacks"][
            "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"
        ]
        ckpt_loaded["callbacks"][
            "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
        ]["best_model_path"] = new_ckpt_path
        ckpt_loaded["callbacks"][
            "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
        ]["dirpath"] = checkpoint_dir
        del ckpt_loaded["callbacks"][
            "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
        ]["last_model_path"]
        torch.save(ckpt_loaded, checkpoint_dir + "/pretrained_ckpt.ckpt")
        ckpt_path = checkpoint_dir + "/pretrained_ckpt.ckpt"
    else:
        ckpt_path = None
        if not args.evaluate_only:
            ckpt_path = checkpoint_dir + "/last.ckpt"
            if not os.path.isfile(ckpt_path):
                ckpt_path = None
        else:
            if args.evaluate_only:
                full_experiment_name_original = (
                    experiment_name + "-seed-" + str(args.seed)
                )
                experiment_id_original = sha1(
                    full_experiment_name_original.encode("utf-8")
                ).hexdigest()[:8]
                checkpoint_dir_wandb = os.path.join(
                    fulldir_experiments,
                    "lag-llama",
                    experiment_id_original,
                    "checkpoints",
                )
                file = os.listdir(checkpoint_dir_wandb)[-1]
                if file:
                    ckpt_path = os.path.join(checkpoint_dir_wandb, file)
            elif args.evaluate_only:
                for file in os.listdir(checkpoint_dir):
                    if "best" in file:
                        ckpt_path = checkpoint_dir + "/" + file
                        break

    if ckpt_path:
        print("Checkpoint", ckpt_path, "retrieved from experiment directory")
    else:
        print("No checkpoints found. Training from scratch.")

    # MLflow logging
    # NOTE: Caution when using `full_experiment_name` after this
    if args.eval_prefix and (args.evaluate_only):
        experiment_name = args.eval_prefix + "_" + experiment_name
    full_experiment_name = experiment_name + "-seed-" + str(args.seed)

    # Set up TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir=args.results_dir,
        name=full_experiment_name,
        version=str(args.seed),
        default_hp_metric=False,
    )
    # Log hyperparameters
    logger.log_hyperparams(vars(args))

    # Callbacks
    swa_callbacks = StochasticWeightAveraging(
        swa_lrs=args.swa_lrs,
        swa_epoch_start=args.swa_epoch_start,
        annealing_epochs=args.annealing_epochs,
        annealing_strategy=args.annealing_strategy,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=int(args.early_stopping_patience),
        verbose=True,
        mode="min",
    )
    model_checkpointing = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, lr_monitor, model_checkpointing]
    if args.swa:
        print("Using SWA")
        callbacks.append(swa_callbacks)

    # # Create train and test datasets
    # if not args.single_dataset:
    #     train_dataset_names = args.all_datasets
    #     for test_dataset in args.test_datasets:
    #         train_dataset_names.remove(test_dataset)
    #     print("Training datasets:", train_dataset_names)
    #     print("Test datasets:", args.test_datasets)
    #     data_id_to_name_map = {}
    #     name_to_data_id_map = {}
    #     for data_id, name in enumerate(train_dataset_names):
    #         data_id_to_name_map[data_id] = name
    #         name_to_data_id_map[name] = data_id
    #     test_data_id = -1
    #     for name in args.test_datasets:
    #         data_id_to_name_map[test_data_id] = name
    #         name_to_data_id_map[name] = test_data_id
    #         test_data_id -= 1
    # else:
    #     print("Training and test on", args.single_dataset)
    #     data_id_to_name_map = {}
    #     name_to_data_id_map = {}
    #     data_id_to_name_map[0] = args.single_dataset
    #     name_to_data_id_map[args.single_dataset] = 0

    # # Get prediction length and set it if we are in the single dataset
    # if args.single_dataset and args.use_dataset_prediction_length:
    #     _, prediction_length, _ = create_test_dataset(
    #         args.single_dataset, args.dataset_path, 0
    #     )
    #     args.prediction_length = prediction_length

    # Cosine Annealing LR
    if args.use_cosine_annealing_lr:
        cosine_annealing_lr_args = {
            "T_max": args.cosine_annealing_lr_t_max,
            "eta_min": args.cosine_annealing_lr_eta_min,
        }
    else:
        cosine_annealing_lr_args = {}

    # Create the estimator
    estimator = LagLlamaEstimator(
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=1,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_embd_per_head=args.n_embd_per_head,
        n_head=args.n_head,
        max_context_length=args.max_context_length,
        rope_scaling=None,
        scaling=args.data_normalization,
        lr=args.lr,
        weight_decay=args.weight_decay,
        distr_output=args.distr_output,
        # augmentations
        aug_prob=args.aug_prob,
        freq_mask_rate=args.freq_mask_rate,
        freq_mixing_rate=args.freq_mixing_rate,
        jitter_prob=args.jitter_prob,
        jitter_sigma=args.jitter_sigma,
        scaling_prob=args.scaling_prob,
        scaling_sigma=args.scaling_sigma,
        rotation_prob=args.rotation_prob,
        permutation_prob=args.permutation_prob,
        permutation_max_segments=args.permutation_max_segments,
        permutation_seg_mode=args.permutation_seg_mode,
        magnitude_warp_prob=args.magnitude_warp_prob,
        magnitude_warp_sigma=args.magnitude_warp_sigma,
        magnitude_warp_knot=args.magnitude_warp_knot,
        time_warp_prob=args.time_warp_prob,
        time_warp_sigma=args.time_warp_sigma,
        time_warp_knot=args.time_warp_knot,
        window_slice_prob=args.window_slice_prob,
        window_slice_reduce_ratio=args.window_slice_reduce_ratio,
        window_warp_prob=args.window_warp_prob,
        window_warp_window_ratio=args.window_warp_window_ratio,
        window_warp_scales=args.window_warp_scales,
        # others
        num_batches_per_epoch=args.num_batches_per_epoch,
        num_parallel_samples=args.num_parallel_samples,
        time_feat=args.time_feat,
        dropout=args.dropout,
        lags_seq=args.lags_seq,
        # data_id_to_name_map=data_id_to_name_map,
        use_cosine_annealing_lr=args.use_cosine_annealing_lr,
        cosine_annealing_lr_args=cosine_annealing_lr_args,
        track_loss_per_series=args.single_dataset is not None,
        ckpt_path=ckpt_path,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.gpu],
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=fulldir_experiments,
        ),
    )

    # Save the args as config to the directory
    config_filepath = fulldir_experiments + "/args.json"

    def path_to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    # Convert args to a dictionary and handle Path objects
    args_dict = {k: path_to_str(v) for k, v in vars(args).items()}
    with open(config_filepath, "w") as config_savefile:
        json.dump(args_dict, config_savefile, indent=4)

    # Save the number of parameters to the directory for easy retrieval
    num_parameters = sum(
        p.numel() for p in estimator.create_lightning_module().parameters()
    )
    num_parameters_path = fulldir_experiments + "/num_parameters.txt"
    with open(num_parameters_path, "w") as num_parameters_savefile:
        num_parameters_savefile.write(str(num_parameters))

    # Log num_parameters
    logger.experiment.add_scalar("num_parameters", num_parameters, 0)

    # Create samplers
    # Here we make a window slightly bigger so that instance sampler can sample from each window
    # An alternative is to have exact size and use different instance sampler (e.g. ValidationSplitSampler)
    # We change ValidationSplitSampler to add min_past
    history_length = estimator.context_length + max(estimator.lags_seq)
    prediction_length = args.prediction_length
    window_size = history_length + prediction_length
    print(
        "Context length:",
        estimator.context_length,
        "Prediction Length:",
        estimator.prediction_length,
        "max(lags_seq):",
        max(estimator.lags_seq),
        "Therefore, window size:",
        window_size,
    )

    # # Remove max(estimator.lags_seq) if the dataset is too small
    # if args.use_single_instance_sampler:
    #     estimator.train_sampler = SingleInstanceSampler(
    #         min_past=estimator.context_length + max(estimator.lags_seq),
    #         min_future=estimator.prediction_length,
    #     )
    #     estimator.validation_sampler = SingleInstanceSampler(
    #         min_past=estimator.context_length + max(estimator.lags_seq),
    #         min_future=estimator.prediction_length,
    #     )
    # else:
    #     estimator.train_sampler = ExpectedNumInstanceSampler(
    #         num_instances=1.0,
    #         min_past=estimator.context_length + max(estimator.lags_seq),
    #         min_future=estimator.prediction_length,
    #     )
    #     estimator.validation_sampler = ExpectedNumInstanceSampler(
    #         num_instances=1.0,
    #         min_past=estimator.context_length + max(estimator.lags_seq),
    #         min_future=estimator.prediction_length,
    #     )

    ## Batch size
    batch_size = args.batch_size

    if args.evaluate_only:
        pass
    else:
        # if not args.single_dataset:
        #     # Create training and validation data
        #     all_datasets, val_datasets, dataset_num_series = [], [], []
        #     dataset_train_num_points, dataset_val_num_points = [], []

        #     for data_id, name in enumerate(train_dataset_names):
        #         data_id = name_to_data_id_map[name]
        #         (
        #             train_dataset,
        #             val_dataset,
        #             total_train_points,
        #             total_val_points,
        #             _,
        #             _,
        #             _,
        #         ) = create_train_and_val_datasets_with_dates(
        #             name,
        #             args.dataset_path,
        #             data_id,
        #             history_length,
        #             prediction_length,
        #             num_val_windows=args.num_validation_windows,
        #             last_k_percentage=args.single_dataset_last_k_percentage,
        #         )
        #         print(
        #             "Dataset:",
        #             name,
        #             "Total train points:",
        #             total_train_points,
        #             "Total val points:",
        #             total_val_points,
        #         )
        #         all_datasets.append(train_dataset)
        #         val_datasets.append(val_dataset)
        #         dataset_num_series.append(len(train_dataset))
        #         dataset_train_num_points.append(total_train_points)
        #         dataset_val_num_points.append(total_val_points)

        #     # Add test splits of test data to validation dataset, just for tracking purposes
        #     test_datasets_num_series = []
        #     test_datasets_num_points = []
        #     test_datasets = []

        #     if args.stratified_sampling:
        #         if args.stratified_sampling == "series":
        #             train_weights = dataset_num_series
        #             val_weights = (
        #                 dataset_num_series + test_datasets_num_series
        #             )  # If there is just 1 series (airpassengers or saugeenday) this will fail
        #         elif args.stratified_sampling == "series_inverse":
        #             train_weights = [1 / x for x in dataset_num_series]
        #             val_weights = [
        #                 1 / x for x in dataset_num_series + test_datasets_num_series
        #             ]  # If there is just 1 series (airpassengers or saugeenday) this will fail
        #         elif args.stratified_sampling == "timesteps":
        #             train_weights = dataset_train_num_points
        #             val_weights = dataset_val_num_points + test_datasets_num_points
        #         elif args.stratified_sampling == "timesteps_inverse":
        #             train_weights = [1 / x for x in dataset_train_num_points]
        #             val_weights = [
        #                 1 / x for x in dataset_val_num_points + test_datasets_num_points
        #             ]
        #     else:
        #         train_weights = val_weights = None

        #     train_data = CombinedDataset(all_datasets, weights=train_weights)
        #     val_data = CombinedDataset(
        #         val_datasets + test_datasets, weights=val_weights
        #     )
        # else:
        #     (
        #         train_data,
        #         val_data,
        #         total_train_points,
        #         total_val_points,
        #         _,
        #         _,
        #         _,
        #     ) = create_train_and_val_datasets_with_dates(
        #         args.single_dataset,
        #         args.dataset_path,
        #         0,
        #         history_length,
        #         prediction_length,
        #         num_val_windows=args.num_validation_windows,
        #         last_k_percentage=args.single_dataset_last_k_percentage,
        #     )
        #     print(
        #         "Dataset:",
        #         args.single_dataset,
        #         "Total train points:",
        #         total_train_points,
        #         "Total val points:",
        #         total_val_points,
        #     )
        train_data = ChronosDataset(
            path=args.dataset_path,
            datasets=CHRONOS_TRAINING_DATASETS[:2],
            probabilities=CHRONOS_TRAINING_DATASET_SIZE[:2],
            transformation=estimator.create_transformation(),
            lags_seq=estimator.lags_seq,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            model_type="causal",
            imputation_method=LastValueImputation(),
            mode="training",
        )

        val_data = ChronosDataset(
            path=args.dataset_path,
            datasets=CHRONOS_TRAINING_DATASETS[:2],
            probabilities=CHRONOS_TRAINING_DATASET_SIZE[:2],
            transformation=estimator.create_transformation(),
            lags_seq=estimator.lags_seq,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            model_type="causal",
            imputation_method=LastValueImputation(),
            mode="validation",
        )

        # Batch size search since when we scale up, we might not be able to use the same batch size for all models
        # if args.search_batch_size:
        #     estimator.num_batches_per_epoch = 10
        #     estimator.limit_val_batches = 10
        #     estimator.trainer_kwargs["max_epochs"] = 1
        #     estimator.trainer_kwargs["callbacks"] = []
        #     estimator.trainer_kwargs["logger"] = None
        #     fulldir_batchsize_search = os.path.join(
        #         fulldir_experiments, "batch-size-search"
        #     )
        #     os.makedirs(fulldir_batchsize_search, exist_ok=True)
        #     while batch_size >= 1:
        #         try:
        #             print("Trying batch size:", batch_size)
        #             batch_size_search_dir = os.path.join(
        #                 fulldir_batchsize_search, "batch-size-search-" + str(batch_size)
        #             )
        #             os.makedirs(batch_size_search_dir, exist_ok=True)
        #             estimator.batch_size = batch_size
        #             estimator.trainer_kwargs["default_root_dir"] = (
        #                 fulldir_batchsize_search
        #             )
        #             # Train
        #             train_output = estimator.train_model(
        #                 training_data=train_data,
        #                 validation_data=val_data,
        #                 shuffle_buffer_length=None,
        #                 ckpt_path=None,
        #             )
        #             break
        #         except RuntimeError as e:
        #             if "out of memory" in str(e):
        #                 gc.collect()
        #                 torch.cuda.empty_cache()
        #                 if batch_size == 1:
        #                     print(
        #                         "Batch is already at the minimum. Cannot reduce further. Exiting..."
        #                     )
        #                     exit(0)
        #                 else:
        #                     print("Caught OutOfMemoryError. Reducing batch size...")
        #                     batch_size //= 2
        #                     continue
        #             else:
        #                 print(e)
        #                 exit(1)
        #     estimator.num_batches_per_epoch = args.num_batches_per_epoch
        #     estimator.limit_val_batches = args.limit_val_batches
        #     estimator.trainer_kwargs["max_epochs"] = args.max_epochs
        #     estimator.trainer_kwargs["callbacks"] = callbacks
        #     estimator.trainer_kwargs["logger"] = logger
        #     estimator.trainer_kwargs["default_root_dir"] = fulldir_experiments
        #     if batch_size > 1:
        #         batch_size //= 2
        #     estimator.batch_size = batch_size
        #     print("\nUsing a batch size of", batch_size, "\n")
        #     logger.log_hyperparams({"batch_size": batch_size})

        # Train using lightning trainer

        # train_output = estimator.train_model(
        #     training_data=train_data,
        #     validation_data=val_data,
        #     shuffle_buffer_length=None,
        #     ckpt_path=ckpt_path,
        # )
        trained_model, trainer = train_model(
            training_network=estimator.create_lightning_module(),
            trainer_kwargs=estimator.trainer_kwargs,
            training_data=train_data,
            batch_size=batch_size,
            validation_data=val_data,
            ckpt_path=ckpt_path,
        )

        # Set checkpoint path before evaluating
        best_model_path = trainer.checkpoint_callback.best_model_path
        estimator.ckpt_path = best_model_path

    # print("Using checkpoint:", estimator.ckpt_path, "for evaluation")
    # # Make directory to store metrics
    # metrics_dir = os.path.join(fulldir_experiments, "metrics")
    # os.makedirs(metrics_dir, exist_ok=True)

    # # Evaluate
    # evaluation_datasets = (
    #     args.test_datasets + train_dataset_names
    #     if not args.single_dataset
    #     else [args.single_dataset]
    # )

    # for name in evaluation_datasets:  # [test_dataset]:
    #     print("Evaluating on", name)
    #     test_data, prediction_length, _ = create_test_dataset(
    #         name, args.dataset_path, window_size
    #     )
    #     print("# of Series in the test data:", len(test_data))

    #     # Adapt evaluator to new dataset
    #     estimator.prediction_length = prediction_length
    #     # Batch size loop just in case. This is mandatory as it involves sampling etc.
    #     # NOTE: In case can't do sampling with even batch size of 1, then keep reducing num_parallel_samples until we can (keeping batch size at 1)
    #     while batch_size >= 1:
    #         try:
    #             # Batch size
    #             print("Trying batch size:", batch_size)
    #             estimator.batch_size = batch_size
    #             predictor = estimator.create_predictor(
    #                 estimator.create_transformation(),
    #                 estimator.create_lightning_module(),
    #             )
    #             # Make evaluations
    #             forecast_it, ts_it = make_evaluation_predictions(
    #                 dataset=test_data, predictor=predictor, num_samples=args.num_samples
    #             )
    #             forecasts = list(forecast_it)
    #             tss = list(ts_it)
    #             break
    #         except RuntimeError as e:
    #             if "out of memory" in str(e):
    #                 gc.collect()
    #                 torch.cuda.empty_cache()
    #                 if batch_size == 1:
    #                     print(
    #                         "Batch is already at the minimum. Cannot reduce further. Exiting..."
    #                     )
    #                     exit(0)
    #                 else:
    #                     print("Caught OutOfMemoryError. Reducing batch size...")
    #                     batch_size //= 2
    #                     continue
    #             else:
    #                 print(e)
    #                 exit(1)

    #     if args.plot_test_forecasts:
    #         print("Plotting forecasts")
    #         figure = plot_forecasts(forecasts, tss, prediction_length)
    #         logger.experiment.add_figure(f"Forecast_plot_of_{name}", figure, 0)

    #     # Get metrics
    #     evaluator = Evaluator(
    #         num_workers=args.num_workers, aggregation_strategy=aggregate_valid
    #     )
    #     agg_metrics, _ = evaluator(
    #         iter(tss), iter(forecasts), num_series=len(test_data)
    #     )
    #     # Save metrics
    #     metrics_savepath = metrics_dir + "/" + name + ".json"
    #     with open(metrics_savepath, "w") as metrics_savefile:
    #         json.dump(agg_metrics, metrics_savefile)

    #     # Log metrics. For now only CRPS is logged.
    #     logger.experiment.add_scalar(
    #         f"test/{name}/CRPS", agg_metrics["mean_wQuantileLoss"], 0
    #     )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Experiment args
    parser.add_argument("-e", "--experiment_name", type=str, required=True)

    # Data arguments
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="autogluon/chronos_datasets",
        help="Enter the datasets folder path here",
    )
    parser.add_argument(
        "--all_datasets", type=str, nargs="+", default=CHRONOS_TRAINING_DATASETS
    )
    parser.add_argument("-t", "--test_datasets", type=str, nargs="+", default=[])
    parser.add_argument(
        "--stratified_sampling",
        type=str,
        choices=["series", "series_inverse", "timesteps", "timesteps_inverse"],
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Model hyperparameters
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--max_context_length", type=int, default=2048)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument(
        "--num_encoder_layer", type=int, default=4, help="Only for lag-transformer"
    )
    parser.add_argument("--n_embd_per_head", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument(
        "--lags_seq", type=str, nargs="+", default=["Q", "M", "W", "D", "H", "T", "S"]
    )

    # Data normalization
    parser.add_argument(
        "--data_normalization", default=None, choices=["mean", "std", "robust", "none"]
    )

    ## Augmentation hyperparameters
    # Augmentation probability
    parser.add_argument("--aug_prob", type=float, default=0)

    # Frequency Masking
    parser.add_argument(
        "--freq_mask_rate", type=float, default=0.1, help="Rate of frequency masking"
    )

    # Frequency Mixing
    parser.add_argument(
        "--freq_mixing_rate", type=float, default=0.1, help="Rate of frequency mixing"
    )

    # Jitter
    parser.add_argument(
        "--jitter_prob",
        type=float,
        default=0,
        help="Probability of applying Jitter augmentation",
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.03,
        help="Standard deviation for Jitter augmentation",
    )

    # Scaling
    parser.add_argument(
        "--scaling_prob",
        type=float,
        default=0,
        help="Probability of applying Scaling augmentation",
    )
    parser.add_argument(
        "--scaling_sigma",
        type=float,
        default=0.1,
        help="Standard deviation for Scaling augmentation",
    )

    # Rotation
    parser.add_argument(
        "--rotation_prob",
        type=float,
        default=0,
        help="Probability of applying Rotation augmentation",
    )

    # Permutation
    parser.add_argument(
        "--permutation_prob",
        type=float,
        default=0,
        help="Probability of applying Permutation augmentation",
    )
    parser.add_argument(
        "--permutation_max_segments",
        type=int,
        default=5,
        help="Maximum segments for Permutation augmentation",
    )
    parser.add_argument(
        "--permutation_seg_mode",
        type=str,
        default="equal",
        choices=["equal", "random"],
        help="Segment mode for Permutation augmentation",
    )

    # MagnitudeWarp
    parser.add_argument(
        "--magnitude_warp_prob",
        type=float,
        default=0,
        help="Probability of applying MagnitudeWarp augmentation",
    )
    parser.add_argument(
        "--magnitude_warp_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for MagnitudeWarp augmentation",
    )
    parser.add_argument(
        "--magnitude_warp_knot",
        type=int,
        default=4,
        help="Number of knots for MagnitudeWarp augmentation",
    )

    # TimeWarp
    parser.add_argument(
        "--time_warp_prob",
        type=float,
        default=0,
        help="Probability of applying TimeWarp augmentation",
    )
    parser.add_argument(
        "--time_warp_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for TimeWarp augmentation",
    )
    parser.add_argument(
        "--time_warp_knot",
        type=int,
        default=4,
        help="Number of knots for TimeWarp augmentation",
    )

    # WindowSlice
    parser.add_argument(
        "--window_slice_prob",
        type=float,
        default=0,
        help="Probability of applying WindowSlice augmentation",
    )
    parser.add_argument(
        "--window_slice_reduce_ratio",
        type=float,
        default=0.9,
        help="Reduce ratio for WindowSlice augmentation",
    )

    # WindowWarp
    parser.add_argument(
        "--window_warp_prob",
        type=float,
        default=0,
        help="Probability of applying WindowWarp augmentation",
    )
    parser.add_argument(
        "--window_warp_window_ratio",
        type=float,
        default=0.1,
        help="Window ratio for WindowWarp augmentation",
    )
    parser.add_argument(
        "--window_warp_scales",
        nargs="+",
        type=float,
        default=[0.5, 2.0],
        help="Scales for WindowWarp augmentation",
    )

    # Argument to include time-features
    parser.add_argument(
        "--time_feat",
        help="include time features",
        action="store_true",
    )

    # Training arguments
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-m", "--max_epochs", type=int, default=10000)
    parser.add_argument("-n", "--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--shuffle_buffer_length", type=int, default=32)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--early_stopping_patience", default=50)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Evaluation arguments
    parser.add_argument("--num_parallel_samples", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)

    # GPU ID
    parser.add_argument("--gpu", type=int, default=0)

    # Directory to save everything in
    parser.add_argument("-r", "--results_dir", type=str, required=True)

    # Other arguments
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Only evaluate, do not train"
    )
    parser.add_argument(
        "--use_kv_cache",
        help="KV caching during infernce. Only for Lag-LLama.",
        action="store_true",
        default=True,
    )

    # SWA arguments
    parser.add_argument(
        "--swa", action="store_true", help="Using Stochastic Weight Averaging"
    )
    parser.add_argument("--swa_lrs", type=float, default=1e-2)
    parser.add_argument("--swa_epoch_start", type=float, default=0.8)
    parser.add_argument("--annealing_epochs", type=int, default=10)
    parser.add_argument(
        "--annealing_strategy", type=str, default="cos", choices=["cos", "linear"]
    )

    # Training/validation iterator type switching
    parser.add_argument(
        "--use_single_instance_sampler", action="store_true", default=True
    )

    # Plot forecasts
    parser.add_argument("--plot_test_forecasts", action="store_true", default=True)

    # Search search_batch_size
    parser.add_argument("--search_batch_size", action="store_true", default=False)

    # Number of validation windows
    parser.add_argument("--num_validation_windows", type=int, default=14)

    # Training KWARGS
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-8)

    # Override arguments with a dictionary file with args
    parser.add_argument("--cfg", action=ActionConfigFile)

    # Evaluation utils
    parser.add_argument("--eval_prefix", type=str)

    # Checkpoints args
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--get_ckpt_path_from_experiment_name", type=str)

    # Single dataset setup: used typically for finetuning
    parser.add_argument("--single_dataset", type=str)
    parser.add_argument(
        "--use_dataset_prediction_length", action="store_true", default=False
    )
    parser.add_argument("--single_dataset_last_k_percentage", type=float)

    # CosineAnnealingLR
    parser.add_argument("--use_cosine_annealing_lr", action="store_true", default=False)
    parser.add_argument("--cosine_annealing_lr_t_max", type=int, default=10000)
    parser.add_argument("--cosine_annealing_lr_eta_min", type=float, default=1e-2)

    # Distribution output
    parser.add_argument(
        "--distr_output", type=str, default="studentT", choices=["studentT"]
    )
    args = parser.parse_args()

    # print args for logging
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    train(args)
