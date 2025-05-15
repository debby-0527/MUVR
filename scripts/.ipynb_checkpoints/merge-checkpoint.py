import json
from collections import defaultdict
import random

def process_file(file_path):
    """处理文件并返回组合条目列表"""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    query_groups = defaultdict(list)
    for entry in entries:
        query = entry.get('Query', '')
        query_groups[query].append(entry)

    combined = []
    for query, group in query_groups.items():
        group_size = len(group)
        for i in range(0, group_size - 1, 2):
            combined.append((group[i], group[i+1]))
    return combined

def deduplicate_combinations(combined):
    """去重相同Query的组合条目"""
    seen = set()
    deduped = []
    for pair in combined:
        query = pair[0].get('Query', '')
        if query not in seen:
            seen.add(query)
            deduped.append(pair)
    return deduped

def select_entries(combined, max_num, seed=42):
    """随机选择条目并保持可复现"""
    if len(combined) <= max_num:
        return combined
    random.seed(seed)
    return random.sample(combined, max_num)

def main(file1_path, file2_path, output_path):
    # 处理第一个文件
    combined1 = process_file(file1_path)
    deduped1 = deduplicate_combinations(combined1)
    selected1 = select_entries(deduped1, 10)
    X = len(selected1)

    # 处理第二个文件
    combined2 = process_file(file2_path)
    selected_queries = {p[0].get('Query', '') for p in selected1}
    filtered2 = [p for p in combined2 if p[0].get('Query', '') not in selected_queries]
    deduped2 = deduplicate_combinations(filtered2)
    required = max(20 - X, 0)
    selected2 = select_entries(deduped2, required) if required > 0 else []

    # 合并并展开结果
    final_pairs = selected1 + selected2
    output_entries = []
    for pair in final_pairs:
        for entry in pair:
            entry.setdefault('Tag', '')
            output_entries.append(entry)

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in output_entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == '__main__':
    import sys
    # if len(sys.argv) != 4:
    #     print("Usage: python script.py <file1> <file2> <output>")
    #     sys.exit(1)
    # for split in ['news', 'animal', 'geng', 'region', 'dance']:
    for split in ['animal']:
        main(
            f"/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/{split}_tag_0.005_clip.jsonl", 
            f"/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/{split}_tv_0.05_clip.jsonl",
            f"/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/{split}_20.jsonl"
        )