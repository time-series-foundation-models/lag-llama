mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

EXP_NAME="pretraining_lag_llama_2"
FILENAME="experiments/seeds/${EXP_NAME}"
CONFIGPATH="configs/lag_llama.json"

echo $EXP_NAME

# NUM_SEEDS used only if it is a new experiment
NUM_SEEDS=1

# Create seeds
if [ -f $FILENAME ]; then
    echo "${FILENAME} already exists."

    SEEDS=()
    while read -r LINE; do
        SEEDS+=("$LINE")
    done < $FILENAME
    echo "Found ${#SEEDS[@]} seeds for training."
else
    # Write seeds
    echo "${FILENAME} created. Writing seeds."
    touch $FILENAME
    for (( i = 0; i < $NUM_SEEDS; i++ )) 
    do 
        SEED=$((RANDOM + 1))
        echo $SEED >> $FILENAME
    done

    # Read them
    SEEDS=()
    while read -r LINE; do
        SEEDS+=("$LINE")
    done < $FILENAME
fi

# Train
for SEED in "${SEEDS[@]}"
do
    EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

    python run.py \
    -e $EXP_NAME -d "datasets" --seed $SEED \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --wandb_entity "ashok-arjun" --wandb_project "lag-llama" --wandb_tags "lag-llama-pretraining" \
    --all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" \
    --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" \
    --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
    --lr 0.0001
done
