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

import os
import random
from itertools import islice

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_gpu_stats():
    # Print GPU stats
    device = torch.cuda.current_device()
    memory_stats = torch.cuda.memory_stats(device=device)
    t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated_memory_gb = memory_stats["allocated_bytes.all.current"] / (1024**3)
    print(f"Total Memory: {t:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")


def plot_forecasts(forecasts, tss, prediction_length):
    plt.figure(figsize=(20, 15))
    plt.rcParams.update({"font.size": 15})

    # Create custom legend handles
    forecast_line = mlines.Line2D([], [], color="g", label="Forecast")
    target_line = mlines.Line2D([], [], color="blue", label="Target")

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx + 1)
        forecast.plot(color="g")
        # ax.plot(forecast, color='g', label="Forecast")
        # ts[-3 * dataset.metadata.prediction_length:][0].plot(label="target")
        ts[-3 * prediction_length :][0].plot(label="target", ax=ax)
        plt.xticks(rotation=60)
        ax.set_title(forecast.item_id)
        # ax.legend()  # Add legend to each subplot
        ax.legend(handles=[forecast_line, target_line])

    plt.gcf().tight_layout()
    return plt.gcf()
