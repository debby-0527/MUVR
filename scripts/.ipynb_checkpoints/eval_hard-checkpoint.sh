#!/bin/bash
# 定义模型列表
models=("EVA-CLIP")
# models=("OpenCLIP")

# modes=("tv")
modes=("tag")

tag_modes=(0.3)
# tag_modes=(0.5 100)

# 基础路径
base_input="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/VLMs_1_15"

# 遍历所有模型和主题组合
for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        for tag_mode in "${tag_modes[@]}"; do
            echo "Processing model: $model mode: $mode"
            
            # 构建输入和输出文件夹路径
            input_folder="$base_input/$model"
            
            # 执行命令
            python evaluate_all_vlm_mul.py \
                --dataset_hdf5 "$input_folder" \
                --topic "animal" \
                --mode "$mode" \
                --model_name "$model" \
                --tag_mode $tag_mode \
                --gpu_id 0
            
            echo "--------------------------------------------------"
        done
    done
done

echo "All tasks completed!"


