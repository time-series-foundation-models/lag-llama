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

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import argparse
import gc
import json
import os
from hashlib import sha1

import lightning
import torch
import wandb
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.evaluation._base import aggregate_valid
from gluonts.transform import ExpectedNumInstanceSampler
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
    LearningRateMonitor
)
from lightning.pytorch.loggers import WandbLogger

from data.data_utils import (
    CombinedDataset,
    SingleInstanceSampler,
    create_test_dataset,
    create_train_and_val_datasets_with_dates,
)

from data.dataset_list import ALL_DATASETS
from utils.utils import plot_forecasts, set_seed


from lag_llama.gluon.estimator import LagLlamaEstimator


def train(args):
    # Set seed
    set_seed(args.seed)
    lightning.seed_everything(args.seed)

    # # Print GPU stats
    # print_gpu_stats()

    # Create a directory to store the results in
    # This string is made independent of hyperparameters here, as more hyperparameters / arguments may be added later
    # The name should be created in the calling bash script
    # This way, when that same script is executed again, automatically the model training is resumed from a checkpoint if available
    experiment_name = args.experiment_name
    fulldir_experiments = os.path.join(args.results_dir, experiment_name, str(args.seed))
    if os.path.exists(fulldir_experiments): print(fulldir_experiments, "already exists.")
    os.makedirs(fulldir_experiments, exist_ok=True)

    # Create directory for checkpoints
    checkpoint_dir = os.path.join(fulldir_experiments, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    elif args.get_ckpt_path_from_experiment_name:
        fulldir_experiments_for_ckpt_path = os.path.join(args.results_dir, args.get_ckpt_path_from_experiment_name, str(args.seed))
        full_experiment_name_original = args.get_ckpt_path_from_experiment_name + "-seed-" + str(args.seed)
        experiment_id_original = sha1(full_experiment_name_original.encode("utf-8")).hexdigest()[:8]
        checkpoint_dir_wandb = os.path.join(fulldir_experiments_for_ckpt_path, "lag-llama", experiment_id_original, "checkpoints")
        file = os.listdir(checkpoint_dir_wandb)[-1]
        if file: ckpt_path = os.path.join(checkpoint_dir_wandb, file)
        if not ckpt_path: raise Exception("ckpt_path not found from experiment name")
        # Delete the EarlyStoppingCallback and save it in the current checkpoint_dir
        new_ckpt_path = checkpoint_dir + "/pretrained_ckpt.ckpt"
        print("Moving", ckpt_path, "to", new_ckpt_path)
        ckpt_loaded = torch.load(ckpt_path)
        del ckpt_loaded['callbacks']["EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"]
        ckpt_loaded['callbacks']["ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["best_model_path"] = new_ckpt_path
        ckpt_loaded['callbacks']["ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["dirpath"] = checkpoint_dir
        del ckpt_loaded['callbacks']["ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["last_model_path"]
        torch.save(ckpt_loaded, checkpoint_dir + "/pretrained_ckpt.ckpt")
        ckpt_path = checkpoint_dir + "/pretrained_ckpt.ckpt"
    else:
        ckpt_path = None
        if not args.evaluate_only:
            ckpt_path = checkpoint_dir + "/last.ckpt"
            if not os.path.isfile(ckpt_path): ckpt_path = None
        else:
            if args.evaluate_only:
                full_experiment_name_original = experiment_name + "-seed-" + str(args.seed)
                experiment_id_original = sha1(full_experiment_name_original.encode("utf-8")).hexdigest()[:8]
                checkpoint_dir_wandb = os.path.join(fulldir_experiments, "lag-llama", experiment_id_original, "checkpoints")
                file = os.listdir(checkpoint_dir_wandb)[-1]
                if file: ckpt_path = os.path.join(checkpoint_dir_wandb, file)
            elif args.evaluate_only:
                for file in os.listdir(checkpoint_dir):
                    if "best" in file:
                        ckpt_path = checkpoint_dir + "/" + file
                        break

    if ckpt_path:
        print("Checkpoint", ckpt_path, "retrieved from experiment directory")
    else:
        print("No checkpoints found. Training from scratch.")

    # W&B logging
    # NOTE: Caution when using `full_experiment_name` after this
    if args.eval_prefix and (args.evaluate_only): experiment_name = args.eval_prefix + "_" + experiment_name
    full_experiment_name = experiment_name + "-seed-" + str(args.seed)
    experiment_id = sha1(full_experiment_name.encode("utf-8")).hexdigest()[:8]
    logger = WandbLogger(name=full_experiment_name, \
                        save_dir=fulldir_experiments, group=experiment_name, \
                        tags=args.wandb_tags, entity=args.wandb_entity, \
                        project=args.wandb_project, allow_val_change=True, \
                        config=vars(args), id=experiment_id, \
                        mode=args.wandb_mode, settings=wandb.Settings(code_dir="."))
    
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
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [early_stop_callback,
                lr_monitor,
                model_checkpointing
                ]
    if args.swa:
        print("Using SWA")
        callbacks.append(swa_callbacks)

    # Create train and test datasets
    if not args.single_dataset:
        train_dataset_names = args.all_datasets
        for test_dataset in args.test_datasets:
            train_dataset_names.remove(test_dataset)
        print("Training datasets:", train_dataset_names)
        print("Test datasets:", args.test_datasets)
        data_id_to_name_map = {}
        name_to_data_id_map = {}
        for data_id, name in enumerate(train_dataset_names):
            data_id_to_name_map[data_id] = name
            name_to_data_id_map[name] = data_id
        test_data_id = -1
        for name in args.test_datasets:
            data_id_to_name_map[test_data_id] = name
            name_to_data_id_map[name] = test_data_id
            test_data_id -= 1
    else:
        print("Training and test on", args.single_dataset)
        data_id_to_name_map = {}
        name_to_data_id_map = {}
        data_id_to_name_map[0] = args.single_dataset
        name_to_data_id_map[args.single_dataset] = 0

    # Get prediction length and set it if we are in the single dataset
    if args.single_dataset and args.use_dataset_prediction_length:
        _, prediction_length, _ = create_test_dataset(
            args.single_dataset, args.dataset_path, 0
        )
        args.prediction_length = prediction_length

    # Cosine Annealing LR
    if args.use_cosine_annealing_lr:
        cosine_annealing_lr_args = {"T_max": args.cosine_annealing_lr_t_max, \
                                    "eta_min": args.cosine_annealing_lr_eta_min}
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
        max_context_length=2048,
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
        data_id_to_name_map=data_id_to_name_map,
        use_cosine_annealing_lr=args.use_cosine_annealing_lr,
        cosine_annealing_lr_args=cosine_annealing_lr_args,
        track_loss_per_series=args.single_dataset != None,
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
    with open(config_filepath, "w") as config_savefile:
        json.dump(vars(args), config_savefile, indent=4)

    # Save the number of parameters to the directory for easy retrieval
    num_parameters = sum(
        p.numel() for p in estimator.create_lightning_module().parameters()
    )
    num_parameters_path = fulldir_experiments + "/num_parameters.txt"
    with open(num_parameters_path, "w") as num_parameters_savefile:
        num_parameters_savefile.write(str(num_parameters))
    # Log num_parameters
    logger.log_metrics({"num_parameters": num_parameters})

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

    # Remove max(estimator.lags_seq) if the dataset is too small
    if args.use_single_instance_sampler:
        estimator.train_sampler = SingleInstanceSampler(
            min_past=estimator.context_length + max(estimator.lags_seq),
            min_future=estimator.prediction_length,
        )
        estimator.validation_sampler = SingleInstanceSampler(
            min_past=estimator.context_length + max(estimator.lags_seq),
            min_future=estimator.prediction_length,
        )
    else:
        estimator.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=estimator.context_length + max(estimator.lags_seq),
            min_future=estimator.prediction_length,
        )
        estimator.validation_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=estimator.context_length + max(estimator.lags_seq),
            min_future=estimator.prediction_length,
        )

    ## Batch size
    batch_size = args.batch_size

    if args.evaluate_only:
        pass
    else:
        if not args.single_dataset:
            # Create training and validation data
            all_datasets, val_datasets, dataset_num_series = [], [], []
            dataset_train_num_points, dataset_val_num_points = [], []

            for data_id, name in enumerate(train_dataset_names):
                data_id = name_to_data_id_map[name]
                (
                    train_dataset,
                    val_dataset,
                    total_train_points,
                    total_val_points,
                    total_val_windows,
                    max_train_end_date,
                    total_points,
                ) = create_train_and_val_datasets_with_dates(
                    name,
                    args.dataset_path,
                    data_id,
                    history_length,
                    prediction_length,
                    num_val_windows=args.num_validation_windows,
                    last_k_percentage=args.single_dataset_last_k_percentage
                )
                print(
                    "Dataset:",
                    name,
                    "Total train points:", total_train_points,
                    "Total val points:", total_val_points,
                )
                all_datasets.append(train_dataset)
                val_datasets.append(val_dataset)
                dataset_num_series.append(len(train_dataset))
                dataset_train_num_points.append(total_train_points)
                dataset_val_num_points.append(total_val_points)

            # Add test splits of test data to validation dataset, just for tracking purposes
            test_datasets_num_series = []
            test_datasets_num_points = []
            test_datasets = []

            if args.stratified_sampling:
                if args.stratified_sampling == "series":
                    train_weights = dataset_num_series
                    val_weights = dataset_num_series + test_datasets_num_series # If there is just 1 series (airpassengers or saugeenday) this will fail
                elif args.stratified_sampling == "series_inverse":
                    train_weights = [1/x for x in dataset_num_series]
                    val_weights = [1/x for x in dataset_num_series + test_datasets_num_series] # If there is just 1 series (airpassengers or saugeenday) this will fail
                elif args.stratified_sampling == "timesteps":
                    train_weights = dataset_train_num_points
                    val_weights = dataset_val_num_points + test_datasets_num_points
                elif args.stratified_sampling == "timesteps_inverse":
                    train_weights = [1 / x for x in dataset_train_num_points]
                    val_weights = [1 / x for x in dataset_val_num_points + test_datasets_num_points]
            else:
                train_weights = val_weights = None
                
            train_data = CombinedDataset(all_datasets, weights=train_weights)
            val_data = CombinedDataset(val_datasets+test_datasets, weights=val_weights)
        else:
            (
                train_data,
                val_data,
                total_train_points,
                total_val_points,
                total_val_windows,
                max_train_end_date,
                total_points,
            ) = create_train_and_val_datasets_with_dates(
                args.single_dataset,
                args.dataset_path,
                0,
                history_length,
                prediction_length,
                num_val_windows=args.num_validation_windows,
                last_k_percentage=args.single_dataset_last_k_percentage
            )
            print(
                "Dataset:",
                args.single_dataset,
                "Total train points:", total_train_points,
                "Total val points:", total_val_points,
            )

        # Batch size search since when we scale up, we might not be able to use the same batch size for all models
        if args.search_batch_size:
            estimator.num_batches_per_epoch = 10
            estimator.limit_val_batches = 10
            estimator.trainer_kwargs["max_epochs"] = 1
            estimator.trainer_kwargs["callbacks"] = []
            estimator.trainer_kwargs["logger"] = None
            fulldir_batchsize_search = os.path.join(
                fulldir_experiments, "batch-size-search"
            )
            os.makedirs(fulldir_batchsize_search, exist_ok=True)
            while batch_size >= 1:
                try:
                    print("Trying batch size:", batch_size)
                    batch_size_search_dir = os.path.join(
                        fulldir_batchsize_search, "batch-size-search-" + str(batch_size)
                    )
                    os.makedirs(batch_size_search_dir, exist_ok=True)
                    estimator.batch_size = batch_size
                    estimator.trainer_kwargs[
                        "default_root_dir"
                    ] = fulldir_batchsize_search
                    # Train
                    train_output = estimator.train_model(
                        training_data=train_data,
                        validation_data=val_data,
                        shuffle_buffer_length=None,
                        ckpt_path=None,
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        if batch_size == 1:
                            print(
                                "Batch is already at the minimum. Cannot reduce further. Exiting..."
                            )
                            exit(0)
                        else:
                            print("Caught OutOfMemoryError. Reducing batch size...")
                            batch_size //= 2
                            continue
                    else:
                        print(e)
                        exit(1)
            estimator.num_batches_per_epoch = args.num_batches_per_epoch
            estimator.limit_val_batches = args.limit_val_batches
            estimator.trainer_kwargs["max_epochs"] = args.max_epochs
            estimator.trainer_kwargs["callbacks"] = callbacks
            estimator.trainer_kwargs["logger"] = logger
            estimator.trainer_kwargs["default_root_dir"] = fulldir_experiments
            if batch_size > 1: batch_size //= 2
            estimator.batch_size = batch_size
            print("\nUsing a batch size of", batch_size, "\n")
            wandb.config.update({"batch_size": batch_size}, allow_val_change=True)


        # Train
        train_output = estimator.train_model(
            training_data=train_data,
            validation_data=val_data,
            shuffle_buffer_length=None,
            ckpt_path=ckpt_path,
        )

        # Set checkpoint path before evaluating
        best_model_path = train_output.trainer.checkpoint_callback.best_model_path
        estimator.ckpt_path = best_model_path


    print("Using checkpoint:", estimator.ckpt_path, "for evaluation")
    # Make directory to store metrics
    metrics_dir = os.path.join(fulldir_experiments, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Evaluate
    evaluation_datasets = args.test_datasets + train_dataset_names if not args.single_dataset else [args.single_dataset]

    for name in evaluation_datasets:  # [test_dataset]:
        print("Evaluating on", name)
        test_data, prediction_length, total_points = create_test_dataset(
            name, args.dataset_path, window_size
        )
        print("# of Series in the test data:", len(test_data))

        # Adapt evaluator to new dataset
        estimator.prediction_length = prediction_length
        # Batch size loop just in case. This is mandatory as it involves sampling etc.
        # NOTE: In case can't do sampling with even batch size of 1, then keep reducing num_parallel_samples until we can (keeping batch size at 1)
        while batch_size >= 1:
            try:
                # Batch size
                print("Trying batch size:", batch_size)
                estimator.batch_size = batch_size
                predictor = estimator.create_predictor(
                    estimator.create_transformation(),
                    estimator.create_lightning_module(),
                )
                # Make evaluations
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=test_data, predictor=predictor, num_samples=args.num_samples
                )
                forecasts = list(forecast_it)
                tss = list(ts_it)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        print(
                            "Batch is already at the minimum. Cannot reduce further. Exiting..."
                        )
                        exit(0)
                    else:
                        print("Caught OutOfMemoryError. Reducing batch size...")
                        batch_size //= 2
                        continue
                else:
                    print(e)
                    exit(1)

        if args.plot_test_forecasts:
            print("Plotting forecasts")
            figure = plot_forecasts(forecasts, tss, prediction_length)
            wandb.log({f"Forecast plot of {name}": wandb.Image(figure)})

        # Get metrics
        evaluator = Evaluator(
            num_workers=args.num_workers, aggregation_strategy=aggregate_valid
        )
        agg_metrics, _ = evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        # Save metrics
        metrics_savepath = metrics_dir + "/" + name + ".json"
        with open(metrics_savepath, "w") as metrics_savefile:
            json.dump(agg_metrics, metrics_savefile)

        # Log metrics. For now only CRPS is logged.
        wandb_metrics = {}
        wandb_metrics["test/" + name + "/" + "CRPS"] = agg_metrics["mean_wQuantileLoss"]
        logger.log_metrics(wandb_metrics)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("-e", "--experiment_name", type=str, required=True)

    # Data arguments
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="datasets",
        help="Enter the datasets folder path here"
    )
    parser.add_argument("--all_datasets", type=str, nargs="+", default=ALL_DATASETS)
    parser.add_argument("-t", "--test_datasets", type=str, nargs="+", default=[])
    parser.add_argument(
        "--stratified_sampling",
        type=str,
        choices=["series", "series_inverse", "timesteps", "timesteps_inverse"],
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Model hyperparameters
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--max_prediction_length", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--num_encoder_layer", type=int, default=4, help="Only for lag-transformer")
    parser.add_argument("--n_embd_per_head", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--lags_seq", type=str, nargs="+", default=["Q", "M", "W", "D", "H", "T", "S"])

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
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-m", "--max_epochs", type=int, default=10000)
    parser.add_argument("-n", "--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--early_stopping_patience", default=50)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Evaluation arguments
    parser.add_argument("--num_parallel_samples", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)

    # GPU ID
    parser.add_argument("--gpu", type=int, default=0)

    # Directory to save everything in
    parser.add_argument("-r", "--results_dir", type=str, required=True)

    # W&B
    parser.add_argument("-w", "--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lag-llama-test")
    parser.add_argument("--wandb_tags", nargs="+")
    parser.add_argument(
        "--wandb_mode", type=str, default="online", choices=["offline", "online"]
    )

    # Other arguments
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Only evaluate, do not train"
    )
    parser.add_argument(
        "--use_kv_cache",
        help="KV caching during infernce. Only for Lag-LLama.",
        action="store_true",
        default=True
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
    parser.add_argument("--use_single_instance_sampler", action="store_true", default=True)

    # Plot forecasts
    parser.add_argument("--plot_test_forecasts", action="store_true", default=True)

    # Search search_batch_size
    parser.add_argument("--search_batch_size", action="store_true", default=False)

    # Number of validation windows
    parser.add_argument("--num_validation_windows", type=int, default=14)

    # Training KWARGS
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-8)

    # Override arguments with a dictionary file with args
    parser.add_argument('--args_from_dict_path', type=str)

    # Evaluation utils
    parser.add_argument("--eval_prefix", type=str)

    # Checkpoints args
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--get_ckpt_path_from_experiment_name", type=str)

    # Single dataset setup: used typically for finetuning
    parser.add_argument("--single_dataset", type=str)
    parser.add_argument("--use_dataset_prediction_length", action="store_true", default=False)
    parser.add_argument("--single_dataset_last_k_percentage", type=float)

    # CosineAnnealingLR
    parser.add_argument("--use_cosine_annealing_lr", action="store_true", default=False)
    parser.add_argument("--cosine_annealing_lr_t_max", type=int, default=10000)
    parser.add_argument("--cosine_annealing_lr_eta_min", type=float, default=1e-2)

    # Distribution output
    parser.add_argument('--distr_output', type=str, default="studentT", choices=["studentT"])

    args = parser.parse_args()

    if args.args_from_dict_path:
        with open(args.args_from_dict_path, "r") as read_file: loaded_args = json.load(read_file)
        for key, value in loaded_args.items():
            setattr(args, key, value)

    # print args for logging
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    train(args)
