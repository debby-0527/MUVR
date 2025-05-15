import os
import json
from collections import defaultdict
import numpy as np
import re

class CuVR:
    def __init__(self, split='all', anno_root="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations", broken_list=None):
        """
        Args:
            split: 数据集划分，可选 'all', 'train', 'test'
            anno_root: 标注文件根目录，包含多个主题子文件夹
            broken_list: 损坏视频列表（可选）
        """
        self.split = split
        self.anno_root = anno_root
        self.topics = self._discover_topics()
        self.broken = set(broken_list) if broken_list else set()
        
        # 初始化数据结构
        self._init_structure()
        self.related_vids = set()  # 存储所有相关视频ID
        self._load_all_topics()
        
        # 合并跨主题数据
        self.queries = [vid for t in self.topics for vid in self.topic_data[t]['queries']]
        # 构建database，包含所有相关视频且排除查询视频
        self.database = list(self.related_vids - set(self.queries))

        # 建立全局视图
        self._build_global_index()

    def _discover_topics(self):
        """发现所有主题文件夹"""
        return [d for d in os.listdir(self.anno_root) 
                if os.path.isdir(os.path.join(self.anno_root, d)) and d[0]!='.']

    def _init_structure(self):
        """初始化存储结构"""
        self.topic_data = {
            t: {
                'database': set(),
                'queries': [],      # 查询视频ID列表
                'tag_lists': {}
            } for t in self.topics
        }
        self.topic_text = {
            t: {} for t in self.topics
        }
        
        # 全局索引
        self.video_to_topic = {}          # 视频所属主题

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

        # 根据split筛选查询
        filtered_queries = []
        for q in queries:
            if self.split == 'all' or q.get('split') == self.split:
                filtered_queries.append(q)
        queries = filtered_queries

        path = os.path.join(self.anno_root, topic, 'query_tags.json')
        with open(path, 'r', encoding='utf-8') as f:
            tags = json.load(f)

        for q in queries:
            vid = self._parse_video_id(q['frames_path'])
            text = q['prompt_en']
            if str(q['id']) in tags.keys():
                self.topic_text[topic][vid] = {'query_prompt': text, 'tags': tags[str(q['id'])]}
            else:
                self.topic_text[topic][vid] = {'query_prompt': text, 'tags': []}
            
        valid_videos = []
        self.vid_name = {}
        self.name_vid = {}
        for v in videos:
            vid = self._parse_video_id(v['frames_path'])
                
            # 记录元数据
            v['video_id'] = vid
            self.video_to_topic[vid] = topic
            self.vid_name[v['id']] = vid
            self.name_vid[vid] = v['id']
            
            # 分类查询
            if v.get('is_query', 0) == 1 and vid in [self._parse_video_id(q['frames_path']) for q in queries]:
                self.topic_data[topic]['queries'].append(vid)
                
            valid_videos.append(vid)
        return valid_videos

    def _load_relationships(self, topic):
        """加载单个主题的关系数据"""
        path = os.path.join(self.anno_root, topic, 'query_rel_lists.json')
        with open(path, 'r', encoding='utf-8') as f:
            query_rel_lists = json.load(f)
        
        for item in query_rel_lists:
            query_id = item['query_id']
            query_vid = self.vid_name.get(query_id)
            if query_vid not in self.topic_data[topic]['queries']:
                continue  # 跳过未保留的查询
            
            rel_lists = item['rel_lists']
            for relation, rel_list in rel_lists.items():
                related_vids = [self.vid_name[v] for v in rel_list['id_list']]
                self.related_vids.update(related_vids)
                self.topic_data[topic]['database'].update(related_vids)
                if relation not in self.topic_data[topic]:
                    self.topic_data[topic][relation] = {}
                self.topic_data[topic][relation][query_vid] = related_vids

        path = os.path.join(self.anno_root, topic, 'query_tag_lists.json')
        with open(path, 'r', encoding='utf-8') as f:
            query_tag_lists = json.load(f)
        
        for item in query_tag_lists:
            query_id = item['query_id']
            query_vid = self.vid_name.get(query_id)
            if query_vid not in self.topic_data[topic]['queries']:
                continue  # 跳过未保留的查询
            
            tag_lists = item['tag_lists']
            self.topic_data[topic]['tag_lists'][query_vid] = {}
            for tag_prompt, tag_list in tag_lists.items():
                related_vids = [self.vid_name[v] for v in tag_list['id_list']]
                self.related_vids.update(related_vids)
                self.topic_data[topic]['tag_lists'][query_vid][tag_prompt] = related_vids

    def _load_all_topics(self):
        """加载所有主题的数据"""
        for topic in self.topics:
            self._load_videos(topic)
            self._load_relationships(topic)

    def _build_global_index(self):
        """构建全局索引"""
        self.global_positives = defaultdict(set)
        for topic in self.topics:
            if 'pos_all' in self.topic_data[topic]:
                for q, vs in self.topic_data[topic]['pos_all'].items():
                    self.global_positives[q].update(vs)

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

    def get_tags(self, topics=None):
        """获取指定主题的数据库视频"""
        # if not topics:
        #     return self.database
        # if type(topics) == list:
        #     return [vid for t in topics for vid in self.topic_data[t]['database']]
        # else:
        return self.topic_data[topics]['tag_lists']
            
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
            # print(target, sim)
            if target != query and target in all_db:
                y_true.append(int(target in query_gt))
                y_score.append(float(sim))
        # print(y_true)
        
        if not y_true:  # 无有效目标
            return {k: 0.0 for k in ['AP'] + [f'recall@{k}' for k in k_list] + [f'precision@{k}' for k in k_list] + [f'hit@{k}' for k in k_list] + ['MRR', 'R-Precision']}, [], []
        
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

    def get_kth_largest_score(self, base_similarities, K):
        # 提取所有的base_score值
        scores = list(base_similarities.values())
        
        # 如果K在0-1之间，返回最小值
        if 0 < K < 1:
            return min(scores)
        
        # 确保K是整数
        K = int(K)
        
        # 对分数进行排序（降序）
        sorted_scores = sorted(scores, reverse=True)
        
        # 处理K超出范围的情况
        if K > len(sorted_scores):
            return sorted_scores[-1]  # 返回最小的
        else:
            return sorted_scores[K-1]  # 返回第K大的（索引从0开始）

    def get_final_similarities(self, base_similarities, tag_similarities, query_tag, mode=0.0):   
        assert mode >= 0
        operator, conditions = self.parse_expression(query_tag)
        K_score = self.get_kth_largest_score(base_similarities, mode)
        p = 2 if mode>1 else mode

        if operator == "AND" or operator == "OR":
            return None
        final_similarities = {}
        for v, base_score in base_similarities.items():
            adjustments = []
            for sign, tag in conditions:
                sim = tag_similarities[v][tag]  # 假设基础分数存在
                adj = sim if sign == ' +' else -sim
                # print(sign, sim, adj)
                adjustments.append(adj)

            # final_similarities[v] = base_score
            if operator == "AND" or operator is None:
                if base_score >= K_score: 
                    if p == 2:
                        final_similarities[v] = sum(adjustments)
                    else:
                        final_similarities[v] = base_score + p * sum(adjustments)
                else:
                    if p == 2:
                        final_similarities[v] = base_score-2
                    else:
                        final_similarities[v] = base_score + p * sum(adjustments)
                # final_similarities[v] = base_score
            elif operator == "OR":
                final_similarities[v] = base_score + (max(adjustments) if adjustments else 0.0)
        return final_similarities

    def evaluate_tag(self, tag_similarities, base_similarities=None, topics=None, verbose=False, k_list=[200,500,1000,2000], mode=0.0):
        # 初始化数据结构
        database = self.get_database(topics) if topics else self.database
        queries = self.get_queries(topics) if topics else self.queries
        # print(queries)
        
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
            if base_similarities is not None and query not in base_similarities:
                not_found += 1
                continue

            if query not in tag_similarities:
                not_found += 1
                continue
            # if query not in positives:
            #     continue
            # print(query)
            if query not in self.topic_data[topics]['tag_lists'].keys():
                # print(query)
                continue
            query_tag_pos = self.topic_data[topics]['tag_lists'][query]
            
            for query_tag, pos_list in query_tag_pos.items():
                operator, conditions = self.parse_expression(query_tag)
                if operator == "AND" or operator == "OR":
                    continue
                if base_similarities is not None:
                    final_similarities = self.get_final_similarities(base_similarities[query], tag_similarities[query], query_tag, mode)
                else:
                    query_similarities = tag_similarities[query][query_tag]
                    # print(tag_similarities[query])
                    final_similarities = {}
                    for v, s in query_similarities.items():
                        final_similarities[v] = s
                    # print(tag_similarities[query][query_tag])
                    # print(tag_similarities[query].keys(), query_tag)
                    
                if final_similarities == None:
                    continue

                # print(final_similarities)
                # print(pos_list)
                # print(query_tag)
                # exit()
                positives = {query: set(pos_list)}
                metrics, _, _ = self.calculate_mAP(query, final_similarities, all_db, positives, k_list)
                
                if False and metrics['AP'] < 0.02:
                
                    query_gt = set(positives[query]).intersection(all_db) if query in positives else set()
                    y_true, y_score = [], []
                    max_neg, neg_target = 0.0, None
                    max_pos, pos_target = 0.0, None
                    for target, sim in final_similarities.items():
                        # print(target, sim)
                        if target != query and target in all_db:
                            if (not (target in query_gt)) and (float(sim) > max_neg):
                                max_neg = float(sim)
                                neg_target = target
                            elif (target in query_gt) and (float(sim) > max_pos):
                                max_pos = float(sim)
                                pos_target = target
                    print('='*50)
                    print(query, query_tag, metrics['AP'])
                    print(pos_target, max_pos)
                    print(neg_target, max_neg)


                if False and metrics['AP'] < 0.006:
                
                    query_gt = set(positives[query]).intersection(all_db) if query in positives else set()
                    y_true, y_score = [], []
                    max_neg, neg_target = 0.0, None
                    max_pos, pos_target = 0.0, None
                    for target, sim in final_similarities.items():
                        # print(target, sim)
                        if target != query and target in all_db:
                            if (not (target in query_gt)) and (float(sim) > max_neg):
                                max_neg = float(sim)
                                neg_target = target
                            elif (target in query_gt) and (float(sim) > max_pos):
                                max_pos = float(sim)
                                pos_target = target
                    print('='*50)
                    print(query, metrics['AP'])
                    print(pos_target, max_pos)
                    print(neg_target, max_neg)
                    # {'Query': query, 'Target': pos_target, 'Label': 1}
                    # {'Query': query, 'Target': neg_target, 'Label': 0}
                    # 以追加模式打开文件
                    if max_pos == 0 or max_neg == 0 or metrics['AP'] == 0 or pos_target is None or neg_target is None:
                        pass
                    else:
                        with open('/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/animal_tag_0.005.jsonl', 'a', encoding='utf-8') as f:
                            # 写入正样本
                            f.write(json.dumps({'Query': query, 'Target': pos_target, 'Tag': query_tag, 'Label': 1}, ensure_ascii=False) + '\n')
                            # 写入负样本
                            f.write(json.dumps({'Query': query, 'Target': neg_target, 'Tag': query_tag, 'Label': 0}, ensure_ascii=False) + '\n')
                
                # print(query, query_tag, len(pos_list), metrics)
                # print()
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
            # 'uAP': self.calculate_uAP(similarities, all_db, positives, queries),
            'uAP': 0.0,
            **{f'recall@{k}': np.mean(v) for k, v in metrics_data['recall'].items()},
            **{f'precision@{k}': np.mean(v) for k, v in metrics_data['precision'].items()},
            **{f'hit@{k}': np.mean(v) for k, v in metrics_data['hit'].items()},
            'MRR': np.mean(metrics_data['MRR']) if metrics_data['MRR'] else 0.0,
            'R-Precision': np.mean(metrics_data['R-Precision']) if metrics_data['R-Precision'] else 0.0
        }
        # 计算最高和最低 AP 的查询
        # max_AP_query = max(metrics_data, key=lambda k: metrics_data[k]['mAP'])
        # min_AP_query = min(metrics_data, key=lambda k: metrics_data[k]['mAP'])
        # print(f"AP 最高的查询是 '{max_AP_query}'，其 AP 值为 {metrics_data[max_AP_query]['mAP']:.4f}")
        # print(f"AP 最低的查询是 '{min_AP_query}'，其 AP 值为 {metrics_data[min_AP_query]['mAP']:.4f}")
        
        if verbose:
            print("\n评估指标汇总:")
            print(f"mAP: {results['mAP']:.4f}")
            # print(f"uAP: {results['uAP']:.4f}")
            for k in k_list:
                print(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
            # for k in k_list:
            #     print(f"Precision@{k}: {results[f'precision@{k}']:.4f}")
                # print(f"Hit@{k}: {results[f'hit@{k}']:.4f}")
            # print(f"MRR: {results['MRR']:.4f}")
            print(f"R-Precision: {results['R-Precision']:.4f}")
        
        return results
    
    def evaluate(self, similarities, relation_type=None, topics=None, verbose=False, k_list=[200,500,1000,2000]):
        # print(similarities)
        # 初始化数据结构
        database = self.get_database(topics) if topics else self.database
        queries = self.get_queries(topics) if topics else self.queries
        # print(queries)
        
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
            # print(query, similarities.keys())
            # exit()
            if query not in similarities:
                # print(similarities)
                not_found += 1
                continue
            if query not in positives:
                continue
            # print(similarities[query])
            # print(positives)
            # exit()
            metrics, _, _ = self.calculate_mAP(query, similarities[query], all_db, positives, k_list)
            
            # if metrics['AP'] < 0.05:
                
            #     query_gt = set(positives[query]).intersection(all_db) if query in positives else set()
            #     y_true, y_score = [], []
            #     max_neg, neg_target = 0.0, None
            #     max_pos, pos_target = 0.0, None
            #     for target, sim in similarities[query].items():
            #         # print(target, sim)
            #         if target != query and target in all_db:
            #             if (not (target in query_gt)) and (float(sim) > max_neg):
            #                 max_neg = float(sim)
            #                 neg_target = target
            #             elif (target in query_gt) and (float(sim) > max_pos):
            #                 max_pos = float(sim)
            #                 pos_target = target
            #     print('='*50)
            #     print(query, metrics['AP'])
            #     print(pos_target, max_pos)
            #     print(neg_target, max_neg)
            #     # {'Query': query, 'Target': pos_target, 'Label': 1}
            #     # {'Query': query, 'Target': neg_target, 'Label': 0}
            #     # 以追加模式打开文件
            #     with open('/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/s2vs/WebVR_Rerank/dance_tv_0.05.jsonl', 'a', encoding='utf-8') as f:
            #         # 写入正样本
            #         f.write(json.dumps({'Query': query, 'Target': pos_target, 'Label': 1}, ensure_ascii=False) + '\n')
            #         # 写入负样本
            #         f.write(json.dumps({'Query': query, 'Target': neg_target, 'Label': 0}, ensure_ascii=False) + '\n')
                
                
            
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

        # 计算最高和最低 AP 的查询
        # max_AP_query = max(metrics_data['mAP'], key=lambda k: metrics_data['mAP'][k].float())
        # min_AP_query = min(metrics_data['mAP'], key=lambda k: metrics_data['mAP'][k].float())
        # print(f"AP 最高的查询是 '{max_AP_query}'，其 AP 值为 {metrics_data['mAP'][max_AP_query].float():.4f}")
        # print(f"AP 最低的查询是 '{min_AP_query}'，其 AP 值为 {metrics_data['mAP'][min_AP_query].float():.4f}")
        
        if verbose:
            # print("\n评估指标汇总:")
            print(f"mAP: {results['mAP']:.4f}")
            print(f"uAP: {results['uAP']:.4f}")
            for k in k_list:
                print(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
            # print(f"R-Precision: {results['R-Precision']:.4f}")

            
        
        return results

# 使用示例
if __name__ == "__main__":
    dataset = CuVR(
        anno_root="/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations",
        # broken_list=["BV13b421n7TU", "BV11a4y1A7gK"] # 损坏视频列表
    )
    exit()
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