import torch
import utils
import argparse
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import EvaluationDataset, CuVR
from datasets.generators import VideoDatasetGenerator, HDF5DatasetGenerator
import pickle
from torch.nn.functional import softmax
import torch.nn.functional as F
import os
from datetime import datetime

@torch.no_grad()
def merge_similarities(v2v_similarities, t2v_similarities):
    tv2v_similarities = {}
    for query in v2v_similarities:
        tv2v_similarities[query] = {
            k: (v + t2v_similarities[query][k]) / 2
            for k, v in v2v_similarities[query].items()
        }
    return tv2v_similarities

@torch.no_grad()
def select_key_features(features, mode='', txt_feat=None):
    if features.shape[0] <= 15:
        return features
    prev_frames = features[:-1]
    next_frames = features[1:]
    if mode != 'filter':
        v_similarities = -1 * (prev_frames * next_frames).sum(dim=1)
    if mode != '':
        t_similarities = (prev_frames @ txt_feat.T).mean(dim=1)
    similarities = v_similarities if mode == '' else t_similarities if mode == 'filter' else v_similarities + t_similarities
    _, indices = torch.topk(similarities, k=15, largest=True)
    indices = torch.clamp(indices + 1, 0, features.size(0)-1)
    return features[indices]

# @torch.no_grad()
# def calculate_batch_similarities(query_features, target_features):
#     # print(query_features.shape, target_features.shape)
#     # 计算帧间相似度矩阵 [222, 500, 15, 15]
#     sim_scores = torch.einsum('qfc,tkc->qtfk', query_features, target_features)
#     # 对每个视频对的15x15矩阵取最大值 [222, 500]
#     max_scores = torch.amax(sim_scores, dim=(2, 3))
#     # 缩放结果
#     return 100.0 * max_scores

@torch.no_grad()
def calculate_batch_similarities(query_features, target_features):
    x = (query_features @ target_features.T)
    # 展平后续所有维度
    flattened = x.view(x.size(0), -1)  # 形状变为 (B, N*M)
    
    # 沿最后一个维度取最大值，并保持维度
    max_values, _ = torch.max(flattened, dim=1, keepdim=True)
    # print(max_values[0])
    # exit()
    return max_values

def process_batch(vid_queries_tensor, text_queries_tensor, tag_queries_tensor, batch_targets, batch_video_ids, queries_ids, similarities, tag_indices):
    """批量处理特征并计算相似度"""
    # 将目标特征堆叠成张量 [batch_size, feature_dim]
    targets = torch.stack(batch_targets, dim=0)
    
    # 批量计算视频相似度
    if vid_queries_tensor is not None:
        v_sims = calculate_batch_similarities(vid_queries_tensor, targets)  # [num_queries, batch_size]
        for i in range(v_sims.shape[0]):
            query_id = queries_ids[i]
            for j, video_id in enumerate(batch_video_ids):
                similarities['v2v'][query_id][video_id] = v_sims[i, j].item()
    
    # 批量计算文本相似度
    if text_queries_tensor is not None:
        t_sims = calculate_batch_similarities(text_queries_tensor, targets)
        for i in range(t_sims.shape[0]):
            query_id = queries_ids[i]
            for j, video_id in enumerate(batch_video_ids):
                similarities['t2v'][query_id][video_id] = t_sims[i, j].item()
    
    # 批量计算标签相似度
    if tag_queries_tensor is not None:
        tag_sims = calculate_batch_similarities(tag_queries_tensor, targets)
        for idx, (q_idx, tag) in enumerate(tag_indices):
            query_id = queries_ids[q_idx]
            for j, video_id in enumerate(batch_video_ids):
                if video_id not in similarities['tag2v'][query_id]:
                    similarities['tag2v'][query_id][video_id] = {}
                similarities['tag2v'][query_id][video_id][tag] = tag_sims[idx, j].item()

@torch.no_grad()
def query_vs_target(sim_network, dataset, args, verbose=True):
    modes = set(args.mode)
    if 'tv' in modes:
        modes.update(['v', 't'])
    if 'tag' in modes:
        modes.add('tv')
    args.mode = list(modes)
    
    load_vid = 'v' in modes or 'tv' in modes
    load_txt = 't' in modes or 'tv' in modes
    load_tag = 'tag' in modes

    device = args.gpus[0] if torch.cuda.is_available() else 'cpu'

    # Load all queries and stack into tensors
    generator = HDF5DatasetGenerator(args.dataset_hdf5, dataset.get_queries(topics=args.topic))
    loader = DataLoader(generator, num_workers=args.workers)
    
    vid_queries, text_queries, tag_data, queries_ids = [], [], [], []
    tag_indices = []  # (query_idx, tag_name)

    for (video_tensor,), query_dict, (video_id,) in tqdm(loader):
        if not video_id:
            continue
            
        # Process video features
        vid_feat = video_tensor.to(device)
        if vid_feat.ndim == 3 and vid_feat.shape[1] == 32:
            vid_feat = vid_feat.mean(1)
        vid_feat = vid_feat / torch.norm(vid_feat, dim=-1, keepdim=True)
        vid_queries.append(vid_feat.squeeze(0))
        
        # Process text features
        txt_feat = query_dict['query_prompt'].to(device)
        # print(txt_feat.shape)
        if txt_feat.size(1) == 8:
            # print(txt_feat)
            # exit()
            txt_feat = torch.mean(txt_feat, dim=1).unsqueeze(1)
        # if txt_feat.ndim == 3 and txt_feat.shape[1] == 1:
        #     txt_feat = txt_feat.mean(1)
        txt_feat = txt_feat / torch.norm(txt_feat, dim=-1, keepdim=True)
        text_queries.append(txt_feat.squeeze(0))
        
        # Process tag features
        if load_tag and 'tags' in query_dict.keys():
            for tag, feat in query_dict['tags'].items():
                tag_feat = feat.to(device)
                # if tag_feat.ndim == 3 and tag_feat.shape[1] == 1:
                #     tag_feat = tag_feat.mean(1)
                tag_feat = tag_feat / torch.norm(tag_feat, dim=-1, keepdim=True)
                tag_data.append(tag_feat.squeeze(0))
                tag_indices.append((len(queries_ids), tag))
        
        queries_ids.append(video_id)

    # Stack features into tensors
    vid_queries_tensor = torch.stack(vid_queries) if vid_queries else None
    text_queries_tensor = torch.stack(text_queries) if text_queries else None
    tag_queries_tensor = torch.stack(tag_data) if tag_data else None

    # Initialize similarity dictionaries
    similarities = {
        'v2v': {qid: {} for qid in queries_ids},
        't2v': {qid: {} for qid in queries_ids},
        'tag2v': {qid: {} for qid in queries_ids}
    }

    # Process database videos
    generator = HDF5DatasetGenerator(args.dataset_hdf5, dataset.get_database(topics=args.topic))
    loader = DataLoader(generator, num_workers=args.workers)
    
    for (video_tensor,), _, (video_id,) in tqdm(loader):
        if not video_id:
            continue
        
        # Process target features
        target = video_tensor.to(device)
        if target.ndim == 3 and target.shape[1] == 32:
            target = target.mean(1)
        target = target / torch.norm(target, dim=-1, keepdim=True)
        target = target.squeeze(0)
        
        # Calculate similarities in batch
        if load_vid and vid_queries_tensor is not None:
            v_sims = calculate_batch_similarities(vid_queries_tensor, target)
            for i in range(v_sims.shape[0]):
                similarities['v2v'][queries_ids[i]][video_id] = v_sims[i].item()
        
        if load_txt and text_queries_tensor is not None:
            t_sims = calculate_batch_similarities(text_queries_tensor, target)
            for i in range(t_sims.shape[0]):
                similarities['t2v'][queries_ids[i]][video_id] = t_sims[i].item()
        
        if load_tag and tag_queries_tensor is not None:
            tag_sims = calculate_batch_similarities(tag_queries_tensor, target)
            for idx, (q_idx, tag) in enumerate(tag_indices):
                qid = queries_ids[q_idx]
                if video_id not in similarities['tag2v'][qid]:
                    similarities['tag2v'][qid][video_id] = {}
                similarities['tag2v'][qid][video_id][tag] = tag_sims[idx].item()
        # exit()

    # Merge similarities if needed
    if 'tv' in modes:
        similarities['tv2v'] = merge_similarities(similarities['v2v'], similarities['t2v'])
    
    # Evaluation
    # print(similarities['tag2v'])
    eval_results = {}
    if 'v' in modes: 
        eval_results['v'] = dataset.evaluate(similarities['v2v'], topics=args.topic)
    if 't' in modes: 
        eval_results['t'] = dataset.evaluate(similarities['t2v'], topics=args.topic)
    if 'tv' in modes: 
        eval_results['tv'] = dataset.evaluate(similarities['tv2v'], topics=args.topic)
    if 'tag' in modes: 
        eval_results['tag'] = dataset.evaluate_tag(similarities['tag2v'], similarities['tv2v'], topics=args.topic, mode=args.tag_mode)

    return eval_results

if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='Evaluate trained student on five splits.',
        formatter_class=formatter)
    parser.add_argument('--dataset', type=str, default='CUVR',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "EVVE", "VCDB", "M2VR", "CUVR"],
                        help='Evaluation dataset name.')
    parser.add_argument('--dataset_hdf5', type=str, required=True,
                        help='Base path to HDF5 dataset features.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers.')
    parser.add_argument('--load_queries', type=utils.bool_flag, default=True,
                        help='Load queries to GPU.')
    parser.add_argument('--store_similarities', type=utils.bool_flag, default=True,
                        help='Store similarity scores.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID(s).')
    parser.add_argument('--pk_path', type=str, default=None,
                        help='Precomputed similarities path.')
    parser.add_argument('--topic', type=str, default=None,
                        help='Evaluation topic.')
    parser.add_argument('--mode', type=str, default='v',
                        help='Evaluation modes: v, t, tv, tag.')

    parser.add_argument('--output_dir', type=str, default='./VLM_results_0421',
                        help='Directory to save evaluation results.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model for filename.')
    parser.add_argument('--tag_mode', type=float, default=0.0,
                        help='.')
    
    args = parser.parse_args()
    args.mode = args.mode.split(',')
    modes = set(args.mode)
    if 'tv' in modes or 'tag' in modes:
        modes.update(['v', 't'])
    if 'tag' in modes:
        modes.add('tv')
    args.mode = list(modes)
    args.gpus = list(map(int, args.gpu_id.split(',')))

    splits = ["news", "geng", "animal", "region", "dance"] if args.topic == 'all' else [args.topic]
    original_hdf5 = args.dataset_hdf5
    all_results = []

    # 生成时间戳和文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_mode = args.topic
    if 'tag' in modes:
        filename = f"{args.model_name}_tag_{args.tag_mode}_{args.topic}_{timestamp}.txt"
    elif 'tv' in modes:
        filename = f"{args.model_name}_tv_{args.topic}_{timestamp}.txt"
    elif 't' in modes:
        filename = f"{args.model_name}_t_{args.topic}_{timestamp}.txt"
    else:
        filename = f"{args.model_name}_v_{args.topic}_{timestamp}.txt"
    
    filepath = os.path.join(args.output_dir, filename)
    os.makedirs(args.output_dir, exist_ok=True)  # 确保目录存在

    for split in splits:
        args.topic = split
        args.dataset_hdf5 = f"{original_hdf5}/{split}"
        dataset = CuVR(split='all')
        if args.pk_path is None:
            eval_results = query_vs_target(None, dataset, args, verbose=True)
            if 'tag' in modes:
                split_result = eval_results.get('tag', {})
            elif 'tv' in modes:
                split_result = eval_results.get('tv', {})
            elif 't' in modes:
                split_result = eval_results.get('t', {})
            else:
                split_result = eval_results.get('v', {})
        else:
            with open(args.pk_path, 'rb') as file:
                all_similarities = pickle.load(file)
            split_result = dataset.evaluate(all_similarities['tv2v'], topics=args.topic)
        
        all_results.append(split_result)
        print(f"\nSplit: {split}")
        print(f"mAP: {split_result['mAP']:.4f}")
        print(f"uAP: {split_result['uAP']:.4f}")
        k_list = sorted([int(k.split('@')[1]) for k in split_result if k.startswith('recall@')])
        for k in k_list:
            print(f"Recall@{k}: {split_result[f'recall@{k}']:.4f}")

        exp_result = '& ' + f"{round(split_result['mAP'] * 100, 1):.1f}" + ' & ' + f"{round(split_result['uAP'] * 100, 1):.1f}"
        for k in k_list:
            exp_result = exp_result + " & " + f"{round(split_result[f'recall@{k}'] * 100, 1):.1f}"
        print(exp_result)

    avg_results = {
        'mAP': np.mean([res['mAP'] for res in all_results]),
        'uAP': np.mean([res['uAP'] for res in all_results]),
    }
    k_list = sorted([int(k.split('@')[1]) for res in all_results for k in res if k.startswith('recall@')])
    k_list = sorted(list(set(k_list)))
    for k in k_list:
        avg_results[f'recall@{k}'] = np.mean([res.get(f'recall@{k}', 0) for res in all_results])

    # 收集输出内容
    content = []
    content.append("=== Hyperparameters ===")
    for arg in vars(args):
        content.append(f"{arg}: {getattr(args, arg)}")
    
    content.append("\n=== Split Results ===")
    for split, res in zip(splits, all_results):
        content.append(f"\nSplit: {split}")
        content.append(f"mAP: {res['mAP']:.4f}")
        content.append(f"uAP: {res['uAP']:.4f}")
        k_list_split = sorted([int(k.split('@')[1]) for k in res if k.startswith('recall@')])
        for k in k_list_split:
            content.append(f"Recall@{k}: {res[f'recall@{k}']:.4f}")
        exp_result = '& ' + f"{round(res['mAP'] * 100, 1):.1f}" + ' & ' + f"{round(res['uAP'] * 100, 1):.1f} "
        for k in k_list_split:
            exp_result += f"& {round(res[f'recall@{k}'] * 100, 1):.1f} "
        content.append(f"Formatted Result: {exp_result}")

    content.append("\n=== Average Results ===")
    content.append(f"mAP: {avg_results['mAP']:.4f}")
    content.append(f"uAP: {avg_results['uAP']:.4f}")
    for k in k_list:
        content.append(f"Recall@{k}: {avg_results[f'recall@{k}']:.4f}")
    avg_exp_result = '& ' + f"{round(avg_results['mAP'] * 100, 1):.1f}" + ' & ' + f"{round(avg_results['uAP'] * 100, 1):.1f} "
    for k in k_list:
        avg_exp_result += f"& {round(avg_results[f'recall@{k}'] * 100, 1):.1f} "
    content.append(f"Average Formatted Result: {avg_exp_result}")
    split_exp_result = ''
    for split, res in zip(splits, all_results):
        split_exp_result += ' & ' + f"{round(res['mAP'] * 100, 1):.1f}"
    split_exp_result += ' & ' + f"{round(avg_results['mAP'] * 100, 1):.1f}"
    content.append(f"{split_exp_result}")

    # 写入文件
    with open(filepath, 'w') as f:
        f.write('\n'.join(content))

    # 控制台输出保持原样
    print("\nAverage Results:")
    print(f"mAP: {avg_results['mAP']:.4f}")
    print(f"uAP: {avg_results['uAP']:.4f}")
    for k in k_list:
        print(f"Recall@{k}: {avg_results[f'recall@{k}']:.4f}")
    print(avg_exp_result)
    print(split_exp_result)