#!/bin/bash
# 定义模型列表
# models=("BLIP" "BLIP2" "SigLIP" "EVA-CLIP" "OpenCLIP" "OAI-CLIP" "Meta-CLIP")
models=("EVA-CLIP" "OpenCLIP")
# models=("EVA-CLIP")

modes=("tv")
# modes=("v" "tv")
# modes=("tag")
tag_modes=(0.0)
# tag_modes=(0.5 100)

# ref_modes=("weight5" "weight10" "weight15")
ref_modes=("max" "mean" "top5" "query5" "weight5")

# 基础路径
base_input="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/VLMs_1_15"

# 遍历所有模型和主题组合
for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        for tag_mode in "${tag_modes[@]}"; do
            for ref_mode in "${ref_modes[@]}"; do
                echo "Processing model: $model mode: $mode ref: $ref_mode"
                
                # 构建输入和输出文件夹路径
                input_folder="$base_input/$model"
                
                # 执行命令
                python evaluate_all_vlm_mul_ref.py \
                    --dataset_hdf5 "$input_folder" \
                    --topic "all" \
                    --mode "$mode" \
                    --model_name "$model" \
                    --tag_mode $tag_mode \
                    --ref_mode $ref_mode \
                    --gpu_id 1 \
                    --output_dir './VLM_results_0430_ref_15_'+"$ref_mode"
                
                echo "--------------------------------------------------"
            done
        done
    done
done

echo "All tasks completed!"


