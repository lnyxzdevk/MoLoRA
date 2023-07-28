#!/bin/bash

# Define common arguments
base_model='meta-llama/Llama-2-13b-hfwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww'
model_type='llama'
data_path='crowdworks/EvolInstruct-cluster'
prompt_template_name='alpaca'

# Define cluster numbers
clusters=(1 2 3 4)

# Loop through cluster numbers and execute the Python command
for cluster_number in "${clusters[@]}"
do
    output_dir="./models/llama-13b-hf-c${cluster_number}"
    python cluster_finetune.py --base_model $base_model --model_type $model_type --output_dir $output_dir --cluster_number $cluster_number --data_path $data_path --prompt_template_name $prompt_template_name --load_8bit
done
