import os
import random
import utils
import h5py
import glob
import torch, json
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from abc import abstractmethod
from torch.utils.data import Dataset
import pickle

# InternV2相关
import cv2
import re

class HDF5DatasetGenerator(Dataset):
    def __init__(self, feature_file, videos, min_len=1, dims=512):
        super(HDF5DatasetGenerator, self).__init__()
        self.is_folder = not feature_file.endswith('hdf5')
        if not self.is_folder:
            self.feature_file = h5py.File(feature_file, "r")
        self.path = feature_file
            
        self.videos = videos
        self.min_len = min_len
        self.dims = dims

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            video_id = self.videos[idx]
            # assert self.is_folder
            features = np.load(os.path.join(self.path, video_id+'.npy'))
            # print('video', features.shape)

            dict_path = os.path.join(self.path, video_id+'_query_dict.pkl')
            is_query = os.path.exists(dict_path)
            if is_query:
                with open(dict_path, 'rb') as f:
                    query_dict = pickle.load(f)
            # print(query_dict, query_dict.keys())
            # while features.shape[0] < self.min_len:
            #     features = np.concatenate([features, features], axis=0)
            # if features.ndim == 2:
            #     features = np.expand_dims(features, 1)
            #     if is_query:
            #         text_features = np.expand_dims(text_features, 1)
            
            features = torch.from_numpy(features.astype(np.float32))
            # print(video_id, features.shape)
            if is_query:
                query_dict['query_prompt'] = torch.from_numpy(query_dict['query_prompt'].astype(np.float32))
                for i, t in query_dict['tags'].items():
                    t = torch.from_numpy(t.astype(np.float32))
                return features, query_dict, video_id
            else:
                return features, {'query_prompt': features}, video_id
        except Exception as e:
            # print(e)
            # exit()
            return torch.tensor([]), '', ''
            # return torch.zeros((0, 1, self.dims)), {'query_prompt': torch.zeros((0, 1, self.dims))}, ''

# internv2 训练
class TrainDataset(Dataset):
    def __init__(self, query_path, target_path, topic, videos, query_text, dataset, seed, dims=512):
        super().__init__()
        self.path = os.path.join(query_path, topic)
        self.videos = videos
        self.query_text = query_text
        self.dataset = dataset
        self.dims = dims
        
        # 构建所有样本的列表 (video_id, tag_spec)
        self.samples = []
        for video_id in self.videos:
            # 获取经过筛选的有效tag及其元数据
            base_tags = self.get_filtered_tags(video_id)

            # 添加有效tag样本（保留符号和原始prompt）
            # for tag_info in base_tags:
            #     self.samples.append((video_id, tag_info))  # 存储完整tag信息
            
            # 添加all样本（标记为特殊类型）
            self.samples.append((video_id, {
                'tag': '',
                'sign': 'all',
                'prompt': ''
            }))
        
        # 使用固定seed进行全局shuffle
        self.fixed_rng = random.Random(seed)
        self.fixed_rng.shuffle(self.samples)

    def get_filtered_tags(self, video_id):
        """获取经过筛选的有效tag字典"""
        raw_tags = self.dataset['tag_lists'][video_id].keys()
        return [
            v for v in self.get_base_tag_prompt(raw_tags)
        ]

    def parse_expression(self, expr):
        # 保持原有解析逻辑不变
        operator = None
        if expr.startswith("AND"):
            operator = "AND"
            expr = expr[3:]
        elif expr.startswith("OR"):
            operator = "OR"
            expr = expr[2:]
        else:
            expr = ' ' + expr
        
        conditions = re.findall(r'( [+-])(.+?)(?=\s* [+-]|$)', expr)
        parsed_conditions = [(sign.strip(), tag.strip()) for sign, tag in conditions]
        return operator, parsed_conditions

    def get_base_tag_prompt(self, base_tag_prompt):
        final_list = []
        for query_tag in base_tag_prompt:
            operator, conditions = self.parse_expression(query_tag)
            if operator == "AND" or operator == "OR":
                continue
            assert len(conditions) == 1
            sign, tag = conditions[0]
            # print(sign, tag)
            final_list.append({'prompt': query_tag, 'sign': sign, 'tag': tag})
        return final_list

    def __len__(self):
        return len(self.samples)

    def text_preprocess(self, a, b):
        a = torch.FloatTensor(a)
        b1 = torch.FloatTensor(b['input_ids'])
        b2 = torch.FloatTensor(b['token_type_ids'])
        b3 = torch.FloatTensor(b['attention_mask'])
        return a, b1, b2, b3

    def __getitem__(self, idx):
        # try:
        video_id, tag_info = self.samples[idx]
        video_feat = np.load(os.path.join(self.path, f"{video_id}.npy"))
        video_feat = torch.FloatTensor(video_feat)
        
        q_text, q_feat = "", torch.zeros(1, 100, 1024)
        t_text, t_feat = "", torch.zeros(1, 100, 1024)
        t_sign, t_prompt = "", ""
        

        dict_path = os.path.join(self.path, f"{video_id}_query_dict.pkl")
        if os.path.exists(dict_path):
            with open(dict_path, "rb") as f:
                query_dict = pickle.load(f)

            q_text = self.query_text[video_id]
            query_prompt, q_info = query_dict["query_prompt"]
            q_feat, q_info_1, q_info_2, q_info_3 = self.text_preprocess(query_prompt, q_info)

            # assert tags := query_dict.get("tags", {})
            if tag_info['sign'] == 'all':
                # 处理all样本
                t_text = ''
                t_sign = "all"
                t_prompt = ''
                t_info = ''
                t_feat = q_feat
                t_info_1, t_info_2, t_info_3 = q_info_1, q_info_2, q_info_3
            else:
                # 从预存信息中直接获取
                t_text = tag_info['tag']
                t_sign = tag_info['sign']
                t_prompt = tag_info['prompt']
                tag_prompt, t_info = query_dict['tags'][t_text]
                t_feat, t_info_1, t_info_2, t_info_3 = self.text_preprocess(tag_prompt, t_info)

        return video_feat, (q_text, q_feat, q_info_1, q_info_2, q_info_3), (t_text, t_feat, t_info_1, t_info_2, t_info_3), video_id, t_sign, t_prompt
        # except Exception as e:
        #     print(f"Error loading {video_id}: {e}")
        #     return torch.zeros(0), ("", torch.zeros(0)), ("", torch.zeros(0)), "", "", ""

    # 保持collate_fn和next_epoch方法不变
    def collate_fn(self, batch):
        # 原有实现保持不变
        valid_batch = [b for b in batch if b[0].shape[0] > 0]
        
        video_feats, query_data, tag_data, video_ids, t_signs, t_prompts = zip(*valid_batch)
        q_texts, q_feats, q_info_1, q_info_2, q_info_3 = zip(*query_data)
        t_texts, t_feats, t_info_1, t_info_2, t_info_3 = zip(*tag_data)
        
        # 处理视频特征
        assert video_feats[0].shape[0] == 1
        padded_video = torch.cat(video_feats, dim=0)

        assert q_feats[0].shape[0] == 1
        padded_q = torch.cat(q_feats, dim=0)
        q_info_a = torch.cat(q_info_1, dim=0)
        q_info_b = torch.cat(q_info_2, dim=0)
        q_info_c = torch.cat(q_info_3, dim=0)

        assert t_feats[0].shape[0] == 1
        padded_t = torch.cat(t_feats, dim=0)
        t_info_a = torch.cat(t_info_1, dim=0)
        t_info_b = torch.cat(t_info_2, dim=0)
        t_info_c = torch.cat(t_info_3, dim=0)
        
        return padded_video, (list(q_texts), padded_q, q_info_a, q_info_b, q_info_c), (list(t_texts), padded_t, t_info_a, t_info_b, t_info_c), list(video_ids), list(t_signs), list(t_prompts)

    def next_epoch(self):
        # 由于已经全局shuffle，不需要每个epoch重新shuffle
        pass

class TestDataset(Dataset):
    def __init__(self, query_path, target_path, topic, videos, query_text, dataset, dims=512, mode='tag'):
        super().__init__()
        self.path = os.path.join(query_path, topic)
        self.videos = videos
        self.query_text = query_text
        self.dataset = dataset
        self.dims = dims
        self.mode = mode

        if self.mode not in ['tag', 'all']:
            raise ValueError(f"Invalid mode: {self.mode}, must be 'tag' or 'all'")

        # 统一使用samples数据结构
        self.samples = []
        for video_id in self.videos:
            if self.mode == 'tag':
                # 复用TrainDataset的标签处理逻辑
                base_tags = self.get_filtered_tags(video_id)
                for tag_info in base_tags:
                    self.samples.append((video_id, tag_info))
            else:
                # 添加all模式特殊标记
                self.samples.append((video_id, {
                    'tag': '',
                    'sign': 'all',
                    'prompt': ''
                }))

    def get_filtered_tags(self, video_id):
        """复用TrainDataset的标签过滤逻辑"""
        raw_tags = self.dataset['tag_lists'][video_id].keys()
        return self.get_base_tag_prompt(raw_tags)

    def parse_expression(self, expr):
        """与TrainDataset保持一致的表达式解析"""
        operator = None
        if expr.startswith("AND"):
            operator = "AND"
            expr = expr[3:]
        elif expr.startswith("OR"):
            operator = "OR"
            expr = expr[2:]
        else:
            expr = ' ' + expr
        
        conditions = re.findall(r'( [+-])(.+?)(?=\s* [+-]|$)', expr)
        return operator, [(s.strip(), t.strip()) for s, t in conditions]

    def get_base_tag_prompt(self, base_tag_prompt):
        final_list = []
        for query_tag in base_tag_prompt:
            operator, conditions = self.parse_expression(query_tag)
            if operator == "AND" or operator == "OR":
                continue
            assert len(conditions) == 1
            sign, tag = conditions[0]
            # print(sign, tag)
            final_list.append({'prompt': query_tag, 'sign': sign, 'tag': tag})
        return final_list

    def __len__(self):
        return len(self.samples)

    def text_preprocess(self, prompt, text_info):
        """增强的文本预处理方法"""
        return (
            torch.FloatTensor(prompt),
            torch.LongTensor(text_info['input_ids']),
            torch.LongTensor(text_info['token_type_ids']),
            torch.LongTensor(text_info['attention_mask'])
        )

    def __getitem__(self, idx):
        video_id, tag_info = self.samples[idx]
        video_feat = np.load(os.path.join(self.path, f"{video_id}.npy"))
        video_feat = torch.FloatTensor(video_feat)
        
        q_text, q_feat = "", torch.zeros(1, 100, 1024)
        t_text, t_feat = "", torch.zeros(1, 100, 1024)
        t_sign, t_prompt = "", ""
        

        dict_path = os.path.join(self.path, f"{video_id}_query_dict.pkl")
        if os.path.exists(dict_path):
            with open(dict_path, "rb") as f:
                query_dict = pickle.load(f)

            q_text = self.query_text[video_id]
            query_prompt, q_info = query_dict["query_prompt"]
            q_feat, q_info_1, q_info_2, q_info_3 = self.text_preprocess(query_prompt, q_info)

            # assert tags := query_dict.get("tags", {})
            if tag_info['sign'] == 'all':
                # 处理all样本
                t_text = ''
                t_sign = "all"
                t_prompt = ''
                t_info = ''
                t_feat = q_feat
                t_info_1, t_info_2, t_info_3 = q_info_1, q_info_2, q_info_3
            else:
                # 从预存信息中直接获取
                t_text = tag_info['tag']
                t_sign = tag_info['sign']
                t_prompt = tag_info['prompt']
                tag_prompt, t_info = query_dict['tags'][t_text]
                t_feat, t_info_1, t_info_2, t_info_3 = self.text_preprocess(tag_prompt, t_info)

        return video_feat, (q_text, q_feat, q_info_1, q_info_2, q_info_3), (t_text, t_feat, t_info_1, t_info_2, t_info_3), video_id, t_sign, t_prompt
        

    # 保持collate_fn和next_epoch方法不变
    def collate_fn(self, batch):
        # 原有实现保持不变
        valid_batch = [b for b in batch if b[0].shape[0] > 0]
        
        video_feats, query_data, tag_data, video_ids, t_signs, t_prompts = zip(*valid_batch)
        q_texts, q_feats, q_info_1, q_info_2, q_info_3 = zip(*query_data)
        t_texts, t_feats, t_info_1, t_info_2, t_info_3 = zip(*tag_data)
        
        # 处理视频特征
        assert video_feats[0].shape[0] == 1
        padded_video = torch.cat(video_feats, dim=0)

        assert q_feats[0].shape[0] == 1
        padded_q = torch.cat(q_feats, dim=0)
        q_info_a = torch.cat(q_info_1, dim=0)
        q_info_b = torch.cat(q_info_2, dim=0)
        q_info_c = torch.cat(q_info_3, dim=0)

        assert t_feats[0].shape[0] == 1
        padded_t = torch.cat(t_feats, dim=0)
        t_info_a = torch.cat(t_info_1, dim=0)
        t_info_b = torch.cat(t_info_2, dim=0)
        t_info_c = torch.cat(t_info_3, dim=0)
        
        return padded_video, (list(q_texts), padded_q, q_info_a, q_info_b, q_info_c), (list(t_texts), padded_t, t_info_a, t_info_b, t_info_c), list(video_ids), list(t_signs), list(t_prompts)

    def next_epoch(self):
        # 由于已经全局shuffle，不需要每个epoch重新shuffle
        pass

class TestDataset_deleted(Dataset):
    def __init__(self, query_path, target_path, topic, videos, query_text, dataset, dims=512, mode='tag'):
        super().__init__()
        self.path = os.path.join(query_path, topic)
        self.videos = videos
        self.query_text = query_text
        self.dataset = dataset
        self.dims = dims
        self.current_epoch = 0
        self.mode = mode

        # 模式验证
        if self.mode not in ['tag', 'all']:
            raise ValueError(f"Invalid mode: {self.mode}, must be 'tag' or 'all'")

        # 统一数据结构
        self.video_tag_map = []

        # 模式差异化处理
        if self.mode == 'tag':
            self._init_tag_mode()
        else:
            self._init_all_mode()

    def _init_tag_mode(self):
        """Tag模式初始化：建立视频-tag映射表"""
        for video_idx, video_id in enumerate(self.videos):
            dict_path = os.path.join(self.path, f"{video_id}_query_dict.pkl")
            if os.path.exists(dict_path):
                with open(dict_path, "rb") as f:
                    query_dict = pickle.load(f)

                if tags := query_dict.get("tags", {}):
                    # 获取基础标签提示
                    base_tag_prompt = self.get_base_tag_prompt(
                        self.dataset['tag_lists'][video_id].keys()
                    )
                    # 为每个tag创建映射项
                    for tag_info in base_tag_prompt:
                        self.video_tag_map.append((video_idx, tag_info, ))

    def _init_all_mode(self):
        """All模式初始化：仅建立视频索引"""
        for video_idx, video_id in enumerate(self.videos):
            dict_path = os.path.join(self.path, f"{video_id}_query_dict.pkl")
            if os.path.exists(dict_path):
                self.video_tag_map.append((video_idx, None))

    def __len__(self):
        return len(self.video_tag_map)

    def text_preprocess(self, a, b):
        a = torch.FloatTensor(a)
        b1 = torch.FloatTensor(b['input_ids'])
        b2 = torch.FloatTensor(b['token_type_ids'])
        b3 = torch.FloatTensor(b['attention_mask'])
        return a, b1, b2, b3

    def __getitem__(self, idx):
        try:
            video_idx, tag_info = self.video_tag_map[idx]
            video_id = self.videos[video_idx]
            video_feat = np.load(os.path.join(self.path, f"{video_id}.npy"))
            video_feat = torch.FloatTensor(video_feat)
            
            # 初始化默认值
            q_text, q_feat = "", torch.zeros(1, self.dims)
            t_text, t_feat = "", torch.zeros(1, self.dims)
            t_sign, t_prompt = "all", ""

            dict_path = os.path.join(self.path, f"{video_id}_query_dict.pkl")
            if os.path.exists(dict_path):
                with open(dict_path, "rb") as f:
                    query_dict = pickle.load(f)
    
                q_text = self.query_text[video_id]
                q_feat = torch.FloatTensor(query_dict["query_prompt"])
                if self.mode == 'tag':
                    # 从预存信息中直接获取
                    t_text = tag_info['tag']
                    t_sign = tag_info['sign']
                    t_prompt = tag_info['prompt']
                    # print(query_dict.keys())
                    t_feat = torch.FloatTensor(query_dict['tags'][t_text])  # 特征加载保持不变
                else:
                    t_text = ''
                    t_sign = "all"
                    t_prompt = ''

            return video_feat, (q_text, q_feat), (t_text, t_feat), video_id, t_sign, t_prompt
        
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
            return torch.zeros(0), ("", torch.zeros(0)), ("", torch.zeros(0)), ""

    def parse_expression(self, expr):
        operator = None
        if expr.startswith("AND"):
            operator = "AND"
            expr = expr[3:]
        elif expr.startswith("OR"):
            operator = "OR"
            expr = expr[2:]
        else:
            expr = ' ' + expr
        
        # 正则匹配所有条件（符号+tag名称）
        conditions = re.findall(r'( [+-])(.+?)(?=\s* [+-]|$)', expr)
        parsed_conditions = [(sign, tag.strip()) for sign, tag in conditions]
        return operator, parsed_conditions

    def get_base_tag_prompt(self, base_tag_prompt):
        final_list = []
        for query_tag in base_tag_prompt:
            operator, conditions = self.parse_expression(query_tag)
            if operator == "AND" or operator == "OR":
                continue
            assert len(conditions) == 1
            sign, tag = conditions[0]
            # print(sign, tag)
            final_list.append({'prompt': query_tag, 'sign': sign[1:], 'tag': tag})
        return final_list

    def collate_fn(self, batch):
        valid_batch = [b for b in batch if b[0].shape[0] > 0]
        if not valid_batch:
            return torch.Tensor(), [("", torch.Tensor())], [("", torch.Tensor())], []
        
        video_feats, query_data, tag_data, video_ids, t_signs, t_prompts = zip(*valid_batch)
        q_texts, q_feats = zip(*query_data)
        t_texts, t_feats = zip(*tag_data)
        
        # 视频特征填充
        max_video_len = max(v.size(0) for v in video_feats)
        padded_video = torch.zeros(len(video_feats), max_video_len, video_feats[0].size(1))
        for i, v in enumerate(video_feats):
            padded_video[i, :v.size(0)] = v
        
        # 查询特征填充
        max_q_len = max(q.size(0) for q in q_feats) if q_feats[0].numel() > 0 else 0
        padded_q = torch.zeros(len(q_feats), max_q_len, q_feats[0].size(1)) if max_q_len > 0 else torch.Tensor()
        for i, q in enumerate(q_feats):
            if max_q_len > 0:
                padded_q[i, :q.size(0)] = q
        
        # 标签特征填充
        max_t_len = max(t.size(0) for t in t_feats) if t_feats[0].numel() > 0 else 0
        padded_t = torch.zeros(len(t_feats), max_t_len, t_feats[0].size(1)) if max_t_len > 0 else torch.Tensor()
        for i, t in enumerate(t_feats):
            if max_t_len > 0:
                padded_t[i, :t.size(0)] = t
        
        return padded_video, (list(q_texts), padded_q), (list(t_texts), padded_t), list(video_ids), list(t_signs), list(t_prompts)

    def next_epoch(self):
        self.current_epoch += 1  # 周期计数递增



class MLLMGenerator(Dataset):

    def __init__(self, anno, dataset_path, window_sz=64, percentage=1., **kargs):
        super(MLLMGenerator, self).__init__()
        self.window_sz = window_sz
        self.dataset_path = dataset_path
        self.idx, self.q_video, self.a_video, self.hard, self.easy, self.idx2neg = anno.get_pairs()
        self.videos = []
        for i in range(len(self.q_video)):
            self.videos.append({
                                'idx': self.idx[i], \
                                'q_video': os.path.join(dataset_path, self.q_video[i]+'.mp4'), \
                                'a_video': os.path.join(dataset_path, self.a_video[i]+'.mp4')
                               })
        self.videos = self.videos[:int(len(self.videos)*percentage)]

        for i in range(len(self.hard)):
            hard = [os.path.join(dataset_path, h+'.mp4') for h in self.hard[i]]
            easy = [os.path.join(dataset_path, e+'.mp4') for e in self.easy[i]]
            self.hard[i] = hard
            self.easy[i] = easy

        # 初始化一个全零的张量
        self.label = torch.zeros(self.window_sz, self.window_sz, dtype=torch.float32)
        
        # 设置对角线元素为1
        self.label.diagonal().fill_(1.0)
        
        # 设置右上和左下的相邻元素为1
        self.label[0, 1] = 1.0  # 右上相邻
        self.label[1, 0] = 1.0  # 左下相邻

    def next_epoch(self):
        random.shuffle(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx]

    def sample_and_shuffle(self, lst, n):
        # 从列表中采样n个不重复的元素
        sampled_elements = random.sample(lst, n)
    
        # 打乱采样出的元素的顺序
        random.shuffle(sampled_elements)
    
        return sampled_elements

    def collate_fn(self, batch):
        batch = batch[0]
        idx, q_video, a_video = batch['idx'], batch['q_video'], batch['a_video']
        hard = self.hard[self.idx2neg[idx]].copy()
        easy = self.easy[self.idx2neg[idx]].copy()
        # print(len(hard), len(easy))

        cnt = self.window_sz - 2
        assert len(easy) >= cnt
        u = 1.
        hard_cnt = min(len(hard), int(cnt * u))
        easy_cnt = cnt - hard_cnt
        hard = self.sample_and_shuffle(hard, hard_cnt)
        easy = self.sample_and_shuffle(easy, easy_cnt)
        
        videos = [q_video, a_video] + hard + easy

        return videos, self.label

class VideoDatasetGenerator(Dataset):
    def __init__(self, dataset_path, videos, pattern, loader='video', fps=1, crop=None, resize=None, video_dict=None, zip_folder=None):
        super(VideoDatasetGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.pattern = pattern
        self.loader = loader
        self.fps = fps
        self.crop = crop
        self.resize = resize
        self.video_dict = video_dict
        self.zip_folder = zip_folder
        # 在类初始化时定义
        self.siglip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # self.resnet_normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )

    def __len__(self):
        return len(self.videos)

    def _center_crop_by_short_edge(self, video_tensor):
        """
        输入: video_tensor 形状为 [B, C, H, W]
        输出: 中心裁剪后的正方形 [B, C, S, S]，其中 S=min(H,W)
        """
        B, C, H, W = video_tensor.shape
        short_edge = min(H, W)
        
        # 计算裁剪区域
        if H > W:
            start_x = (H - short_edge) // 2
            return video_tensor[:, :, start_x:start_x+short_edge, :]
        else:
            start_y = (W - short_edge) // 2
            return video_tensor[:, :, :, start_y:start_y+short_edge]

    def __getitem__(self, idx):
        try:
            if self.loader == 'video':
                raw_video = utils.load_video_pyav(self.video_dict[self.videos[idx]], fps=self.fps)  # [T,H,W,C]
                video = torch.from_numpy(raw_video)  # uint8类型
                
                # 根据模型类型分支处理
                if self.zip_folder != None:
                    with open(self.zip_folder+self.videos[idx]+'.json', 'r', encoding='utf-8') as f:
                        index_list = json.load(f)
                    # 计算每个子列表的中间值
                    middle_indices = []
                    for sublist in index_list:
                        mid = len(sublist) // 2  # 对于偶数长度，取较小的中间值
                        middle_indices.append(sublist[mid])
                    
                    # 将中间值转换为张量
                    middle_indices_tensor = torch.tensor(middle_indices, dtype=torch.long)
                    
                    """SigLIP专用处理流程"""
                    # Step 1: 帧采样（取中间帧）
                    video = video[middle_indices_tensor, :, :, :]  # [10, H, W, 3]
                    
                    # Step 2: 转换为CHW并归一化
                    video = video.permute(0, 3, 1, 2).float() / 255.0  # [10,3,H,W]
                    
                    # Step 3: 短边中心裁剪
                    video = self._center_crop_by_short_edge(video)  # [10,3,S,S]
                    
                    # Step 4: 缩放到目标尺寸
                    video = F.interpolate(video, size=(364,364), mode='bilinear')
                    
                    # Step 5: SigLIP归一化
                    video = self.siglip_normalize(video)
                else:  # resnet等标准模型
                    """ResNet专用处理流程"""
                    # Step 1: 转换为CHW并归一化
                    # video = video.permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]
                    video = video.permute(0, 3, 1, 2)
                    # Step 2: 短边中心裁剪
                    video = self._center_crop_by_short_edge(video)  # [T,3,S,S]
                    
                    # Step 3: 缩放到目标尺寸
                    video = F.interpolate(video, size=(224,224), mode='bilinear')
                    
                    # Step 4: ImageNet归一化
                    # video = self.resnet_normalize(video)
                        
                
    
            elif self.loader == 'frame':
                frame_dir = os.path.join(self.dataset_path, self.pattern.replace('{id}', self.videos[idx]))
                video = utils.load_frames_opencv(frame_dir, crop=self.crop, resize=self.resize)
            return video, self.videos[idx]
        except Exception as e:
            # if len(video) != 0:
            print(e)
            return torch.tensor([]), ''

class InternV2Generator(Dataset):
    def __init__(self, dataset_path, videos, size_t=224, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(InternV2Generator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.size_t = size_t
        self.video_dict = video_dict
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters
        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data/255.0-self.v_mean)/self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube
    
    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(224, 224)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        # 模式选择：优先平均分段，否则滑窗
        if total_frames >= num_segments * min_frames_for_average:
            # --- 平均分段模式 ---
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                # 计算片段起始位置
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                # 段内采样或填充
                if len(segment_frames) > frames_per_segment:
                    # 降采样（如 20帧 -> 10帧）
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    # 填充最后一帧（如 5帧 -> 10帧）
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
    
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        else:
            # --- 均匀滑窗模式（允许重叠）---
            # 计算滑窗步长（确保窗口均匀分布）
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                # 计算窗口位置（防止越界）
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                print(total_frames, start, end)
                # 填充不足的帧
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        
        # Stack segments along a new dimension (0 for batch-like processing)
        # print(torch.cat(segments, dim=0).shape)
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        try:
            video = cv2.VideoCapture(self.video_dict[self.videos[idx]]['video_path'])
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        frames = [x for x in self._frame_from_video(video)]
        if len(frames) < self.num_segments * self.frames_per_segment:
            print('【Len】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        
        if self.num_segments == 1:
            # Original behavior - process entire video
            video_tensor = self.frames2tensor(
                frames, 
                fnum=self.frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  # [1, 8, 3, 224, 224]
        else:
            # New behavior - process segments
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=self.frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )  # [num_segments, 1, frames_per_segment, 3, 224, 224]
            
        return video_tensor, self.video_dict[self.videos[idx]]['query_text'], self.videos[idx]

class BLIP2Generator(Dataset):
    def __init__(self, dataset_path, videos, size_t=364, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(BLIP2Generator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.size_t = size_t
        self.video_dict = video_dict
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data/255.0-self.v_mean)/self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(364, 364), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube
    
    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(364, 364)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        # 模式选择：优先平均分段，否则滑窗
        if total_frames >= num_segments * min_frames_for_average:
            # --- 平均分段模式 ---
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                # 计算片段起始位置
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                # 段内采样或填充
                if len(segment_frames) > frames_per_segment:
                    # 降采样（如 20帧 -> 10帧）
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    # 填充最后一帧（如 5帧 -> 10帧）
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
    
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        else:
            # --- 均匀滑窗模式（允许重叠）---
            # 计算滑窗步长（确保窗口均匀分布）
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                # 计算窗口位置（防止越界）
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                print(total_frames, start, end)
                # 填充不足的帧
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        
        # Stack segments along a new dimension (0 for batch-like processing)
        # print(torch.cat(segments, dim=0).shape)
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        try:
            video = cv2.VideoCapture(self.video_dict[self.videos[idx]]['video_path'])
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        frames = [x for x in self._frame_from_video(video)]

        frames_per_segment = min(self.frames_per_segment, len(frames))
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        
        if self.num_segments == 1:
            # Original behavior - process entire video
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  # [1, 8, 3, 364, 364]
        else:
            # New behavior - process segments
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )  # [num_segments, 1, frames_per_segment, 3, 364, 364]
            
        return video_tensor, self.video_dict[self.videos[idx]]['query_text'], self.videos[idx]

class CoVRGenerator(Dataset):
    def __init__(self, dataset_path, videos, size_t=364, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=15):
        super(CoVRGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.size_t = size_t
        self.video_dict = video_dict
        # print(self.dataset_path, self.dataset_path)
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data/255.0-self.v_mean)/self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(364, 364), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube
    
    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(364, 364)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        # 模式选择：优先平均分段，否则滑窗
        if total_frames >= num_segments * min_frames_for_average:
            # --- 平均分段模式 ---
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                # 计算片段起始位置
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                # 段内采样或填充
                if len(segment_frames) > frames_per_segment:
                    # 降采样（如 20帧 -> 10帧）
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    # 填充最后一帧（如 5帧 -> 10帧）
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
    
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        else:
            # --- 均匀滑窗模式（允许重叠）---
            # 计算滑窗步长（确保窗口均匀分布）
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                # 计算窗口位置（防止越界）
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                print(total_frames, start, end)
                # 填充不足的帧
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        
        # Stack segments along a new dimension (0 for batch-like processing)
        # print(torch.cat(segments, dim=0).shape)
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        # print(os.path.join(self.dataset_path, self.videos[idx]))
        try:
            video = cv2.VideoCapture(os.path.join(self.dataset_path, self.videos[idx])+'.mp4')
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        frames = [x for x in self._frame_from_video(video)]

        frames_per_segment = min(self.frames_per_segment, len(frames))
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】', len(frames), self.frames_per_segment)
            print(self.videos[idx])
            return '', '', self.videos[idx]
        
        if self.num_segments == 1:
            # Original behavior - process entire video
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  # [1, 8, 3, 364, 364]
        else:
            # New behavior - process segments
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )  # [num_segments, 1, frames_per_segment, 3, 364, 364]
            
        return video_tensor[0], self.video_dict[self.videos[idx]], self.videos[idx]


class CLIPGenerator(Dataset):
    def __init__(self, dataset_path, videos, size_t=288, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(CLIPGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.size_t = size_t
        self.video_dict = video_dict
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data/255.0-self.v_mean)/self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(288, 288), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube
    
    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(288, 288)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        # 模式选择：优先平均分段，否则滑窗
        if total_frames >= num_segments * min_frames_for_average:
            # --- 平均分段模式 ---
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                # 计算片段起始位置
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                # 段内采样或填充
                if len(segment_frames) > frames_per_segment:
                    # 降采样（如 20帧 -> 10帧）
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    # 填充最后一帧（如 5帧 -> 10帧）
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
    
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        else:
            # --- 均匀滑窗模式（允许重叠）---
            # 计算滑窗步长（确保窗口均匀分布）
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                # 计算窗口位置（防止越界）
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                print(total_frames, start, end)
                # 填充不足的帧
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        
        # Stack segments along a new dimension (0 for batch-like processing)
        # print(torch.cat(segments, dim=0).shape)
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        try:
            video = cv2.VideoCapture(self.video_dict[self.videos[idx]]['video_path'])
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        frames = [x for x in self._frame_from_video(video)]

        frames_per_segment = min(self.frames_per_segment, len(frames))
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        
        if self.num_segments == 1:
            # Original behavior - process entire video
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  # [1, 8, 3, 288, 288]
        else:
            # New behavior - process segments
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )  # [num_segments, 1, frames_per_segment, 3, 288, 288]
            
        return video_tensor, self.video_dict[self.videos[idx]]['query_text'], self.videos[idx]

class FDCAGenerator(Dataset):
    def __init__(self, dataset_path, videos, size_t=288, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(CoVRGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.size_t = size_t
        self.video_dict = video_dict
        # print(self.dataset_path, self.dataset_path)
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data/255.0-self.v_mean)/self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(288, 288), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube
    
    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(288, 288)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        # 模式选择：优先平均分段，否则滑窗
        if total_frames >= num_segments * min_frames_for_average:
            # --- 平均分段模式 ---
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                # 计算片段起始位置
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                # 段内采样或填充
                if len(segment_frames) > frames_per_segment:
                    # 降采样（如 20帧 -> 10帧）
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    # 填充最后一帧（如 5帧 -> 10帧）
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
    
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        else:
            # --- 均匀滑窗模式（允许重叠）---
            # 计算滑窗步长（确保窗口均匀分布）
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                # 计算窗口位置（防止越界）
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                print(total_frames, start, end)
                # 填充不足的帧
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                
                # Process frames
                segment_frames = [cv2.resize(x[:,:,::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                
                segments.append(vid_tube)
        
        # Stack segments along a new dimension (0 for batch-like processing)
        # print(torch.cat(segments, dim=0).shape)
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        # print(os.path.join(self.dataset_path, self.videos[idx]))
        try:
            video = cv2.VideoCapture(os.path.join(self.dataset_path, self.videos[idx])+'.mp4')
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        frames = [x for x in self._frame_from_video(video)]

        frames_per_segment = min(self.frames_per_segment, len(frames))
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】', len(frames), self.frames_per_segment)
            print(self.videos[idx])
            return '', '', self.videos[idx]
        
        if self.num_segments == 1:
            # Original behavior - process entire video
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  # [1, 8, 3, 288, 288]
        else:
            # New behavior - process segments
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )  # [num_segments, 1, frames_per_segment, 3, 288, 288]
            
        return video_tensor[0], self.video_dict[self.videos[idx]], self.videos[idx]

class VLMGenerator(Dataset):
    def __init__(self, dataset_path, videos, model_type='BLIP2', size_t=364, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(VLMGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.model_type = model_type  # 'BLIP2' or 'CLIP'
        self.size_t = size_t
        self.video_dict = video_dict
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters for both models
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)
        
        # Adjust size_t based on model type
        # if model_type == 'CLIP':
        #     self.size_t = 288  # CLIP uses smaller size by default

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(364, 364), device=torch.device('cuda'), model_type=''):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        # if model_type == 'SigLip':
        #     return vid_list
        vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube

    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(364, 364)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        if total_frames >= num_segments * min_frames_for_average:
            # Average segment mode
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                if len(segment_frames) > frames_per_segment:
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                # elif len(segment_frames) < frames_per_segment:
                    # segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))
                    

                segment_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                segments.append(vid_tube)
        else:
            # Sliding window mode
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))

                segment_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                segments.append(vid_tube)
        
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        if self.videos[idx][:12] in ['BV1CC4y197Mb', 'BV1X94y1W7wN', 'BV1vW411T7HM']:
            return None, None, self.videos[idx]
            
        try:
            video = cv2.VideoCapture(self.video_dict[self.videos[idx]]['video_path'])
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx][:12])
            with open('/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/error.txt', 'a') as f:
                f.write('\n'+self.videos[idx][:12])  # 换行追加
            return None, None, self.videos[idx]
        
        frames = [x for x in self._frame_from_video(video)]
        frames_per_segment = min(self.frames_per_segment, len(frames))
        
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】')
            print(self.videos[idx][:12])
            with open('/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/error.txt', 'a') as f:
                f.write('\n'+self.videos[idx][:12])  # 换行追加
            return None, None, self.videos[idx]
        # return None, None, self.videos[idx]
        if self.num_segments == 1:
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device,
                model_type=self.model_type
            )  
        else:
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )
            
        return video_tensor, self.video_dict[self.videos[idx]]['query_text'], self.videos[idx]

