#!/bin/bash

# pip install clip transformers open_clip_torch salesforce-lavis timm

# models=("BLIP" "BLIP2" "SigLIP" "OpenCLIP" "OAI-CLIP" "Meta-CLIP" "EVA-CLIP")
models=("EVA-CLIP" "OpenCLIP")

topics=("news" "region" "animal" "dance" "others")

base_input="/MUVR/videos/all_videos"
base_output="/MUVR/features/VLMs_1_15"

for model in "${models[@]}"; do
    for topic in "${topics[@]}"; do
        echo "Processing model: $model with topic: $topic"
        
        input_folder="$base_input/$topic"
        output_folder="$base_output/$model/$topic"

        python extract_all_features.py \
            --model "$model" \
            --input_folder "$input_folder" \
            --output_folder "$output_folder" \
            --num_segments 1 \
            --frames_per_segment 15 \
            --topic "$topic" \
            --gpu_ids 0,1,2,3,4,5,6,7
        
        echo "--------------------------------------------------"
    done
done

echo "All tasks completed!"


    
    

