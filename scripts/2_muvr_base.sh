#!/bin/bash

# models=("BLIP" "BLIP2" "SigLIP" "EVA-CLIP" "OpenCLIP" "OAI-CLIP" "Meta-CLIP")
models=("OpenCLIP" "EVA-CLIP")

modes=("t" "v" "tv")

tag_modes=(0.0)

base_input="/MUVR/features/VLMs_1_15"

for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        for tag_mode in "${tag_modes[@]}"; do
            echo "Processing model: $model mode: $mode"
            
            input_folder="$base_input/$model"
            
            python evaluate_all_vlm_mul.py \
                --dataset_hdf5 "$input_folder" \
                --topic "all" \
                --mode "$mode" \
                --model_name "$model" \
                --tag_mode $tag_mode \
                --gpu_id 0 \
                --output_dir './VLM_results'
            
            echo "--------------------------------------------------"
        done
    done
done

echo "All tasks completed!"


