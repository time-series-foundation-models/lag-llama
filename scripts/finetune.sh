# This script can only be executed once you have trained a model. The experiment name used for the trained model should be specified in line 4

CONFIGPATH="configs/lag_llama.json"
PRETRAINING_EXP_NAME="pretraining_lag_llama"
PERCENTAGE=100 # Change to lesser value to limit the history. Use 20, 40, 60, 80 to reproduce experiments in the paper.

for FINETUNE_DATASET in "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25"
do
    EXP_NAME="${PRETRAINING_EXP_NAME}_finetune_on_${FINETUNE_DATASET}"

    # We reuse the same seeds as used for pretraining
    FILENAME="experiments/seeds/${PRETRAINING_EXP_NAME}"
    echo $PRETRAINING_EXP_NAME

    # Get the seeds
    if [ -f $FILENAME ]; then
        echo "${FILENAME} found. Reading seeds."
        SEEDS=()
        while read -r LINE; do
            SEEDS+=("$LINE")
        done < $FILENAME
        echo "Found ${#SEEDS[@]} seeds for finetuning."
    else
        echo "${FILENAME} does not exist. Cannot perform finetuning."
        exit 0
    fi

    # Iterate through all training dataset
    for SEED in "${SEEDS[@]}"
    do
        EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

        python run.py \
        -e $EXPERIMENT_NAME -d "datasets" --seed $SEED \
        -r "experiments/results" \
        --batch_size 512 -m 1000 -n 128 \
        --wandb_entity "enter-wandb-entity" --wandb_project "enter-wandb-project" --wandb_tags "enter-wandb-tags-or-remove-this-argument" \
        --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
        --single_dataset $FINETUNE_DATASET \
        --get_ckpt_path_from_experiment_name $PRETRAINING_EXP_NAME --lr 0.00001 --use_dataset_prediction_length --num_validation_windows 1 \
        --single_dataset_last_k_percentage $PERCENTAGE
    done
done