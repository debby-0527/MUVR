import os
import json
from collections import defaultdict
import numpy as np

class M2VR:
    def __init__(self, anno_root="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations", broken_list=None):
        """
        Args:
            anno_root: 标注文件根目录，包含多个主题子文件夹
            broken_list: 损坏视频列表（可选）
        """
        self.anno_root = anno_root
        self.topics = self._discover_topics()
        self.broken = set(broken_list) if broken_list else set()
        
        # 初始化数据结构
        self._init_structure()
        self._load_all_topics()
        
        # 合并跨主题数据
        self.queries = [vid for t in self.topics for vid in self.topic_data[t]['queries']]
        self.database = [vid for t in self.topics for vid in self.topic_data[t]['database']]
        
        # 建立全局视图
        self._build_global_index()

    def _discover_topics(self):
        """发现所有主题文件夹"""
        return [d for d in os.listdir(self.anno_root) 
                if os.path.isdir(os.path.join(self.anno_root, d)) and d[0]!='.' ]

    def _init_structure(self):
        """初始化存储结构"""
        self.topic_data = {
            t: {
                'videos': {},       # 视频ID到元数据映射
                'queries': [],      # 查询视频ID列表
                'database': [],     # 数据库视频ID列表
                'positives': defaultdict(set),  # 正样本关系
                'independent': defaultdict(set) # 负样本关系
            } for t in self.topics
        }
        self.topic_text = {
            t: {} for t in self.topics
        }
        
        # 全局索引
        self.video_metadata = {}          # 视频ID到完整元数据
        self.video_to_topic = {}          # 视频所属主题
        self.relationship_stats = defaultdict(set) # 关系类型统计

    def _parse_video_id(self, frames_path):
        """从frames_path解析视频唯一标识"""
        return frames_path.split('/')[-1]

    def _load_videos(self, topic):
        """加载单个主题的视频数据"""
        path = os.path.join(self.anno_root, topic, 'videos.json')
        with open(path, 'r', encoding='utf-8') as f:
            videos = json.load(f)

        path = os.path.join(self.anno_root, topic, 'queries_en.json')
        with open(path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        for q in queries:
            vid = self._parse_video_id(q['frames_path'])
            text = q['prompt_en']
            self.topic_text[topic][vid] = text
            
        valid_videos = []
        self.vid_name = {}
        self.name_vid = {}
        for v in videos:
            vid = self._parse_video_id(v['frames_path'])
            # if vid in self.broken:
            #     continue
                
            # 记录元数据
            v['video_id'] = vid
            self.video_metadata[vid] = v
            self.video_to_topic[vid] = topic
            self.vid_name[v['id']] = vid
            self.name_vid[vid] = v['id']
            
            # 分类查询/数据库
            if v.get('is_query', 0) == 1:
                self.topic_data[topic]['queries'].append(vid)
            else:
                self.topic_data[topic]['database'].append(vid)
                
            valid_videos.append(vid)
        return valid_videos

    def _load_relationships(self, topic):
        """加载单个主题的关系数据"""
        path = os.path.join(self.anno_root, topic, 'relationships_zh_en_modified.json')
        with open(path, 'r', encoding='utf-8') as f:
            rels = json.load(f)
            
        for rel in rels:
            vid1 = self.vid_name[rel['video1_id']]
            vid2 = self.vid_name[rel['video2_id']]
            relation = rel['relationship']
            
            # 记录关系类型
            self.relationship_stats[relation].add((vid1, vid2))
            
            # 构建正负样本关系
            if relation == 'independent' or relation == 'hard_independent':
                self._add_independent(topic, vid1, vid2)
            else:
                self._add_positive(topic, vid1, vid2, relation)

    def _add_positive(self, topic, vid1, vid2, relation):
        """添加正样本关系"""
        # 双向关系
        for src, dst in [(vid1, vid2), (vid2, vid1)]:
            if src in self.topic_data[topic]['queries']:
                self.topic_data[topic]['positives'][src].add(dst)
                
    def _add_independent(self, topic, vid1, vid2):
        """添加负样本关系"""
        for src, dst in [(vid1, vid2), (vid2, vid1)]:
            if src in self.topic_data[topic]['queries']:
                self.topic_data[topic]['independent'][src].add(dst)
                
    def _load_all_topics(self):
        """加载所有主题的数据"""
        for topic in self.topics:
            self._load_videos(topic)
            self._load_relationships(topic)

    def _build_global_index(self):
        """构建全局索引"""
        # 合并正样本关系
        self.global_positives = defaultdict(set)
        for topic in self.topics:
            for q, vs in self.topic_data[topic]['positives'].items():
                self.global_positives[q].update(vs)
                
        # 合并负样本关系
        # self.global_independent = defaultdict(set)
        # for topic in self.topics:
        #     for q, vs in self.topic_data[topic]['independent'].items():
        #         self.global_independent[q].update(vs)

    def get_queries(self, topics=None):
        """获取指定主题的查询视频"""
        if not topics:
            return self.queries
        if type(topics) == list:
            return [vid for t in topics for vid in self.topic_data[t]['queries']]
        else:
            return [vid for vid in self.topic_data[topics]['queries']]

    def get_database(self, topics=None):
        """获取指定主题的数据库视频"""
        if not topics:
            return self.database
        if type(topics) == list:
            return [vid for t in topics for vid in self.topic_data[t]['database']]
        else:
            return [vid for vid in self.topic_data[topics]['database']] 
            
    def get_query_text(self, topics=None):
        assert topics and type(topics) != list
        return self.topic_text[topics]

    def calculate_metric(self, y_true, y_score, gt_len):
        if gt_len == 0:
            return 0.0
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)

    def compute_metrics(self, y_true, y_score, gt_len, k_list=[1,5,10,50,100]):
        """计算多个评估指标，包括AP, Recall@k, Precision@k等"""
        if gt_len == 0:
            return {
                'AP': 0.0,
                **{f'recall@{k}': 0.0 for k in k_list},
                **{f'precision@{k}': 0.0 for k in k_list},
                **{f'hit@{k}': 0.0 for k in k_list},
                'MRR': 0.0,
                'R-Precision': 0.0
            }
        
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_y_true = y_true[sorted_indices]
        
        # 计算AP
        precisions = np.cumsum(sorted_y_true) / (np.arange(len(sorted_y_true)) + 1)
        recall_deltas = sorted_y_true / gt_len
        ap = np.sum(precisions * recall_deltas)
        
        metrics = {'AP': ap}
        
        # 各k值指标
        for k in k_list:
            topk = sorted_y_true[:k]
            tp = np.sum(topk)
            metrics[f'recall@{k}'] = tp / gt_len
            metrics[f'precision@{k}'] = tp / k if k !=0 else 0.0
            metrics[f'hit@{k}'] = 1.0 if tp > 0 else 0.0
        
        # MRR
        for idx, val in enumerate(sorted_y_true):
            if val == 1:
                metrics['MRR'] = 1.0 / (idx + 1)
                break
        else:
            metrics['MRR'] = 0.0
        
        # R-Precision
        r = gt_len
        metrics['R-Precision'] = np.sum(sorted_y_true[:r]) / r if r !=0 else 0.0
        
        return metrics
    
    def calculate_mAP(self, query, targets, all_db, positives, k_list):
        """计算单个查询的评估指标"""
        query_gt = set(positives[query]).intersection(all_db) if query in positives else set()
        
        y_true, y_score = [], []
        for target, sim in targets.items():
            if target != query and target in all_db:
                y_true.append(int(target in query_gt))
                y_score.append(float(sim))
        
        if not y_true:  # 无有效目标
            return {k: 0.0 for k in ['AP'] + [f'recall@{k}' for k in k_list] + [f'precision@{k}' for k in k_list] + ['MRR', 'R-Precision']}, [], []
        
        return self.compute_metrics(y_true, y_score, len(query_gt), k_list), y_true, y_score
    
    def calculate_uAP(self, similarities, all_db, positives, queries):
        y_true, y_score, gt_len = [], [], 0
        for query in queries:
            if query not in similarities:
                continue
            if query in positives:
                query_gt = set(positives[query]).intersection(all_db)
            else:
                query_gt = set()
            gt_len += len(query_gt)
            for target, sim in similarities[query].items():
                if target != query and target in all_db:
                    y_true.append(int(target in query_gt))
                    y_score.append(float(sim))
        
        return self.calculate_metric(y_true, y_score, gt_len)
    
    def evaluate(self, similarities, relation_type=None, topics=None, verbose=True, k_list=[200,500,1000,2000]):
        # 初始化数据结构
        database = self.get_database(topics) if topics else self.database
        queries = self.get_queries(topics) if topics else self.queries
        
        # 确定关系类型对应的数据集
        if relation_type is None:
            positives = self.global_positives
            exclude = set()
        else:
            if relation_type == 'event':
                positives = self.global_positives_event
                exclude = self.videos_copy | self.videos_copy_and_event
            elif relation_type == 'copy':
                positives = self.global_positives_copy
                exclude = self.videos_event | self.videos_copy_and_event
            elif relation_type == 'copy_and_event':
                positives = self.global_positives_copy_and_event
                exclude = self.videos_event | self.videos_copy
            else:
                raise ValueError(f"Unsupported relation type: {relation_type}")
            database = list(set(database) - exclude)
        
        all_db = set(database)
        mAP, mAP_dict = [], {}
        not_found = 0
        # 指标收集器
        metrics_data = {
            'mAP': [],
            'recall': defaultdict(list),
            'precision': defaultdict(list),
            'hit': defaultdict(list),
            'MRR': [],
            'R-Precision': []
        }
    
        for query in queries:
            if query not in similarities:
                not_found += 1
                continue
            if query not in positives:
                continue
                
            metrics, _, _ = self.calculate_mAP(query, similarities[query], all_db, positives, k_list)
            
            # 收集指标
            metrics_data['mAP'].append(metrics['AP'])
            for k in k_list:
                metrics_data['recall'][k].append(metrics[f'recall@{k}'])
                metrics_data['precision'][k].append(metrics[f'precision@{k}'])
                metrics_data['hit'][k].append(metrics[f'hit@{k}'])
            metrics_data['MRR'].append(metrics['MRR'])
            metrics_data['R-Precision'].append(metrics['R-Precision'])

        # 计算平均值
        results = {
            'mAP': np.mean(metrics_data['mAP']) if metrics_data['mAP'] else 0.0,
            'uAP': self.calculate_uAP(similarities, all_db, positives, queries),
            **{f'recall@{k}': np.mean(v) for k, v in metrics_data['recall'].items()},
            **{f'precision@{k}': np.mean(v) for k, v in metrics_data['precision'].items()},
            **{f'hit@{k}': np.mean(v) for k, v in metrics_data['hit'].items()},
            'MRR': np.mean(metrics_data['MRR']) if metrics_data['MRR'] else 0.0,
            'R-Precision': np.mean(metrics_data['R-Precision']) if metrics_data['R-Precision'] else 0.0
        }
        
        if verbose:
            print("\n评估指标汇总:")
            print(f"mAP: {results['mAP']:.4f}")
            print(f"uAP: {results['uAP']:.4f}")
            for k in k_list:
                print(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
            # for k in k_list:
            #     print(f"Precision@{k}: {results[f'precision@{k}']:.4f}")
                # print(f"Hit@{k}: {results[f'hit@{k}']:.4f}")
            # print(f"MRR: {results['MRR']:.4f}")
            print(f"R-Precision: {results['R-Precision']:.4f}")
        
        return results

# 使用示例
if __name__ == "__main__":
    dataset = M2VR(
        anno_root="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations",
        broken_list=["BV13b421n7TU", "BV11a4y1A7gK"] # 损坏视频列表
    )
    
    # 模拟相似度计算结果
    sims = {
        "BV13b421n7TU": {"BV13b421n7TU": 0.9, "BV11a4y1A7gK": 0.8},
        "BV11a4y1A7gK": {"BV13b421n7TU": 0.85}
    }
    
    # 全量评估
    print("Global mAP:", dataset.evaluate(sims))
    
    # 评估特定关系和主题
    print("Event mAP in animal:", 
          dataset.evaluate(sims, 
                          relation_type="instance", 
                          topics=["animal"]))