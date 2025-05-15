import os
import json
import argparse
from tqdm import tqdm

def process_data(jsonl_path, json_path, output_path):
    # 读取JSON文件并建立索引
    json_data = {}
    with open(json_path, 'r') as f:
        try:
            # 加载整个JSON数组
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            raise

        for item in tqdm(data, desc='Loading JSON data'):
            # 过滤Clip不符合要求的条目
            if item.get('Clip', '') != 'Clip: 00.00.00-00.00.00':
                continue
            
            # 提取frames_path的关键部分
            frames_full = item['frames_path']
            frame_key = os.path.basename(frames_full)
            
            # 建立索引
            json_data[frame_key] = {
                'prompt_en': item['prompt_en'],
                'is_query': item['is_query']
            }

    # 处理JSONL文件
    processed = []
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc='Processing JSONL'):
            data = json.loads(line)
            query = data['Query']
            
            if query in json_data:
                data['prompt_en'] = json_data[query]['prompt_en']
                processed.append(data)

    # 保存结果
    with open(output_path, 'w') as f:
        for item in tqdm(processed, desc='Writing output'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge prompt_en data')
    parser.add_argument('--jsonl', type=str, default="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/animal_tag_0.005.jsonl")
    parser.add_argument('--json', type=str, default="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations/animal/queries_en.json")
    parser.add_argument('--output', type=str, default="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/animal_tag_0.005_clip.jsonl")
    
    args = parser.parse_args()
    
    process_data(
        jsonl_path=args.jsonl,
        json_path=args.json,
        output_path=args.output
    )