#!/bin/bash
# 扩充VLM类型，推理检验，输出错误视频并修复，抽特征复现，批量VLM。。。
# 基于结果构建数据集，基于数据集测试VLLM，取最好视觉结果构建数据集，对比表格，对比大模型纠正能力。。。

# pip install clip transformers open_clip_torch salesforce-lavis
# 定义模型列表
# models=("BLIP" "BLIP2" "SigLIP" "OpenCLIP" "OAI-CLIP" "Meta-CLIP" "EVA-CLIP")
models=("EVA-CLIP" "OpenCLIP")

# 定义主题列表
topics=("news" "geng" "animal" "region" "dance")

# 基础路径
base_input="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/video_6_336/all_file"
base_output="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/VLMs_1_120"

# 遍历所有模型和主题组合
for model in "${models[@]}"; do
    for topic in "${topics[@]}"; do
        echo "Processing model: $model with topic: $topic"
        
        # 构建输入和输出文件夹路径
        input_folder="$base_input/$topic"
        output_folder="$base_output/$model/$topic"
        
        # 执行命令
        python extract_all_features.py \
            --model "$model" \
            --input_folder "$input_folder" \
            --output_folder "$output_folder" \
            --num_segments 1 \
            --frames_per_segment 120 \
            --topic "$topic" \
            --gpu_ids 0,1,2,3,4,5,6,7
        
        echo "--------------------------------------------------"
    done
done

echo "All tasks completed!"


    
    

