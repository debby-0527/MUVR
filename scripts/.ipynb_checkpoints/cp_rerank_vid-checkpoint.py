import json
import os
import shutil
from pathlib import Path

def copy_videos_from_jsonl(jsonl_path, video_source_dir, target_dir):
    """
    从JSONL文件中读取Query和Target视频路径，并复制到目标文件夹
    
    :param jsonl_path: JSONL文件路径
    :param video_source_dir: 视频源文件夹路径
    :param target_dir: 目标文件夹路径
    """
    # 确保目标文件夹存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 处理Query视频
                if 'Query' in data:
                    query_video = f"{video_source_dir}/{data['Query']}.mp4"
                    if os.path.exists(query_video):
                        shutil.copy2(query_video, target_dir)
                        print(f"Copied Query video: {query_video}")
                    else:
                        print(f"Query video not found: {query_video}")
                
                # 处理Target视频
                if 'Target' in data:
                    target_video = f"{video_source_dir}/{data['Target']}.mp4"
                    if os.path.exists(target_video):
                        shutil.copy2(target_video, target_dir)
                        print(f"Copied Target video: {target_video}")
                    else:
                        print(f"Target video not found: {target_video}")
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
                continue

# 使用示例
if __name__ == "__main__":
    for split in ['news', 'geng', 'region', 'animal', 'dance']:
        jsonl_path = f"/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/{split}_20.jsonl"  # 替换为你的JSONL文件路径
        video_source_dir = f"/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/video_6_336/all_file/{split}"    # 替换为视频源文件夹路径
        target_dir = "/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/video_6_336/qa_file"   # 替换为目标文件夹路径
        
        copy_videos_from_jsonl(jsonl_path, video_source_dir, target_dir)