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

import requests
# import json


ALL_DATASETS = set(
    [
        "australian_electricity_demand",
        "electricity_hourly",
        "london_smart_meters_without_missing",
        "solar_10_minutes",
        "wind_farms_without_missing",
        "pedestrian_counts",
        "uber_tlc_hourly",
        "traffic",
        "kdd_cup_2018_without_missing",
        "saugeenday",
        "sunspot_without_missing",
        "exchange_rate",
        "cpu_limit_minute",
        "cpu_usage_minute",
        "function_delay_minute",
        "instances_minute",
        "memory_limit_minute",
        "memory_usage_minute",
        "platform_delay_minute",
        "requests_minute",
        "ett_h1",
        "ett_h2",
        "ett_m1",
        "ett_m2",
        "beijing_pm25",
        "AirQualityUCI",
        "beijing_multisite",
        "weather",
    ]
)

ALL_CHRONOS_DATASETS = set(
    [
        "dominick",
        "electricity_15min",
        "ercot",
        "exchange_rate",
        "m4_daily",
        "m4_hourly",
        "m4_monthly",
        "m4_quarterly",
        "m4_weekly",
        "m4_yearly",
        "m5",
        "mexico_city_bikes",
        "monash_australian_electricity",
        "monash_car_parts",
        "monash_cif_2016",
        "monash_covid_deaths",
        "monash_electricity_hourly",
        "monash_electricity_weekly",
        "monash_fred_md",
        "monash_hospital",
        "monash_kdd_cup_2018",
        "monash_london_smart_meters",
        "monash_m1_monthly",
        "monash_m1_quarterly",
        "monash_m1_yearly",
        "monash_m3_monthly",
        "monash_m3_quarterly",
        "monash_m3_yearly",
        "monash_nn5_weekly",
        "monash_pedestrian_counts",
        "monash_rideshare",
        "monash_saugeenday",
        "monash_temperature_rain",
        "monash_tourism_monthly",
        "monash_tourism_quarterly",
        "monash_tourism_yearly",
        "monash_traffic",
        "monash_weather",
        "nn5",
        "solar",
        "solar_1h",
        "taxi_1h",
        "taxi_30min",
        "uber_tlc_daily",
        "uber_tlc_hourly",
        "ushcn_daily",
        "weatherbench_daily",
        "weatherbench_hourly_10m_u_component_of_wind",
        "weatherbench_hourly_10m_v_component_of_wind",
        "weatherbench_hourly_2m_temperature",
        "weatherbench_hourly_geopotential",
        "weatherbench_hourly_potential_vorticity",
        "weatherbench_hourly_relative_humidity",
        "weatherbench_hourly_specific_humidity",
        "weatherbench_hourly_temperature",
        "weatherbench_hourly_toa_incident_solar_radiation",
        "weatherbench_hourly_total_cloud_cover",
        "weatherbench_hourly_total_precipitation",
        "weatherbench_hourly_u_component_of_wind",
        "weatherbench_hourly_v_component_of_wind",
        "weatherbench_hourly_vorticity",
        "weatherbench_weekly",
        "wiki_daily_100k",
        "wind_farms_daily",
        "wind_farms_hourly",
    ]
)

CHRONOS_ZERO_SHOT_DATASETS = set(
    [
        "monash_traffic",
        "monash_australian_electricity",
        "ercot",
        "ETTm",
        "ETTh",
        "exchange_rate",
        "nn5",
        "monash_nn5_weekly",
        "monash_weather",
        "monash_covid_deaths",
        "monash_fred_md",
        "m4_quarterly",
        "m4_yearly",
        "dominick",
        "m5",
        "monash_tourism_monthly",
        "monash_tourism_quarterly",
        "monash_tourism_yearly",
        "monash_car_parts",
        "monash_hospital",
        "monash_cif_2016",
        "monash_m1_yearly",
        "monash_m1_quarterly",
        "monash_m1_monthly",
        "monash_m3_monthly",
        "monash_m3_yearly",
        "monash_m3_quarterly",
    ]
)

CHRONOS_TRAINING_DATASETS = sorted(
    list(ALL_CHRONOS_DATASETS - CHRONOS_ZERO_SHOT_DATASETS)
)


# make a call to https://datasets-server.huggingface.co/size?dataset=autogluon/chronos_datasets
# to get the size of each dataset
def get_dataset_sizes(datasets="autogluon/chronos_datasets"):
    url = f"https://datasets-server.huggingface.co/size?dataset={datasets}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to retrieve dataset sizes. Status code: {response.status_code}")
        return None


CHRONOS_TRAINING_DATASET_SIZE = [
    281757285,
    65546675,
    1336971,
    52772258,
    2556691,
    103206946,
    31139117,
    333563,
    8780105,
    597389119,
    12377357,
    781873,
    222678,
    44967856,
    1507924920,
    124515417,
    2265297,
    4541244,
    84747,
    1878515,
    221089890,
    18924392742,
    4459525170,
    4459131344,
    4417880446,
    4066338801,
    4294284095,
    4444863474,
    3690593237,
    4444952494,
    3698713695,
    4089257085,
    4012970585,
    4458570074,
    4458487590,
    4448083821,
    2243012083,
    592554033,
    598834,
    12333116,
]

# dataset_sizes = get_dataset_sizes()
# if dataset_sizes and "size" in dataset_sizes and "configs" in dataset_sizes["size"]:
#     config_dict = {
#         config["config"]: config["num_bytes_parquet_files"]
#         for config in dataset_sizes["size"]["configs"]
#     }

#     for dataset in CHRONOS_TRAINING_DATASETS:
#         if dataset in config_dict:
#             CHRONOS_TRAINING_DATASET_SIZE.append(config_dict[dataset])
#         else:
#             print(f"Warning: Size information not found for dataset {dataset}")
#             CHRONOS_TRAINING_DATASET_SIZE.append(None)
