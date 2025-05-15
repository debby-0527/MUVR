import numpy as np
import pickle as pk

from collections import defaultdict
import json, os
from decord import VideoReader

class M2VR(object):
    
    def __init__(self, anno_folder='/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/annotations'):
        self.topics = ['news']  # 根据实际情况修改主题名称
        self._load_anno(anno_folder)
        broken = ['BV13b421n7TU','BV11a4y1A7gK','BV1DZ42117EC_00.00.00-00.02.00','BV1DZ42117EC_00.02.00-00.04.00','BV1DZ42117EC_00.04.00-00.05.02','BV1DZ42117EC_00.05.02-00.06.04','BV1dv421k7yj_00.00.00-00.02.00','BV1dv421k7yj_00.02.00-00.03.03','BV1dv421k7yj_00.03.03-00.04.06','BV1HH4y1B7HH','BV1PQ4y1F7wP','BV1ay411e7uj','BV1oC4y1o7rk_00.00.00-00.02.00','BV1oC4y1o7rk_00.02.00-00.03.01','BV1oC4y1o7rk_00.03.01-00.04.03','BV1wA4m157Ud','BV1CC4y197Mb_00.00.00-00.01.27','BV1CC4y197Mb_00.01.27-00.02.55']
        broken = set(broken)
        self.queries = [i for i in dataset['queries'] if i not in broken]
        self.database = [i for i in dataset['database'] if i not in broken]

        # 分别获取每种关系类型的正样本
        self.copy = dataset['positives']['copy']
        self.event = dataset['positives']['event']
        self.copy_and_event = dataset['positives']['copy and event']
        self.independent = dataset['positives']['independent']

        # 合并所有关系类型的正样本，保留原有功能
        merged_dict = defaultdict(list)
        for d in (self.copy, self.event, self.copy_and_event):
            for key, value in d.items():
                merged_dict[key].extend(value)
        self.positives = dict(merged_dict)

        merged_dict = defaultdict(list)
        for key, value in self.copy.items():
            merged_dict[key].extend(value)
        self.positives_copy = dict(merged_dict)

        merged_dict = defaultdict(list)
        for key, value in self.event.items():
            merged_dict[key].extend(value)
        self.positives_event = dict(merged_dict)

        merged_dict = defaultdict(list)
        for key, value in self.copy_and_event.items():
            merged_dict[key].extend(value)
        self.positives_copy_and_event = dict(merged_dict)


        # 获取每种关系类型涉及的视频集合
        self.videos_copy = set()
        for vids in self.positives_copy.values():
            self.videos_copy.update(vids)
        self.videos_event = set()
        for vids in self.positives_event.values():
            self.videos_event.update(vids)
        self.videos_copy_and_event = set()
        for vids in self.positives_copy_and_event.values():
            self.videos_copy_and_event.update(vids)

        
        # from tqdm import tqdm
        # for i in tqdm(self.database):
        #     try:
        #         a = VideoReader('/mnt/wfs/mmchongqingssdwfssz/project_wx-search-alg-gs/sleepfeng/datasets/VR/M2VR/news_all/'+i+'.mp4')[0].asnumpy()
        #     except Exception as e:
        #         print(i)
        # print(len(broken))
        # exit()
        self.idx, self.q_video, self.a_video, self.hard, self.easy = [], [], [], [], []
        self.idx2neg = {}
        temp = set()
        idx = 0
        for query in self.positives.keys():
            cur_list = [query] + self.positives[query]
            cur_list = [i for i in cur_list if i not in broken]
            hard = list(self.independent[query])
            hard = [i for i in hard if i not in broken]
            pos = hard + cur_list
            pos = set(pos)
            easy = [i for i in self.database if i not in pos]
            self.hard.append(hard)
            self.easy.append(easy)
            for i in range(len(cur_list)-1):
                j = i + 1
                while j < len(cur_list):
                    a = cur_list[i]
                    b = cur_list[j]
                    # print(a, b)
                    if (a+b not in temp) and (b+a not in temp):
                        temp.add(a+b)
                        self.idx.append(idx)
                        self.idx2neg[idx] = len(self.hard) - 1
                        idx = idx + 1
                        self.q_video.append(a)
                        self.a_video.append(b)
                        
                    j = j + 1
                if i == 0:
                    break
            if idx > 20:
                break

    def _load_anno(self, anno_folder):
        # 加载视频元数据
        with open(os.path.join(anno_folder, 'news/videos.json'), 'r', encoding='utf-8') as f:
            videos_data = json.load(f)
        
        # 构建基础数据结构
        self.id_to_video = {v['id']: v for v in videos_data}
        self.all_videos = [v['id'] for v in videos_data]
        self.queries = [v['id'] for v in videos_data if v.get('is_query')]

        # 加载关系数据并按主题分类
        with open(os.path.join(anno_folder, 'relationships.json'), 'r', encoding='utf-8') as f:
            relationships = json.load(f)

        # 初始化数据结构
        self.positives = {topic: defaultdict(set) for topic in self.topics}
        self.independent = {topic: defaultdict(set) for topic in self.topics}

        # 解析关系数据
        for rel in relationships:
            topic = rel['topic']  # 假设关系数据中包含topic字段
            vid1, vid2 = rel['video1_id'], rel['video2_id']
            relationship = rel['relationship']

            # 处理正样本
            if relationship != 'independent':
                if vid1 in self.queries:
                    self.positives[topic][vid1].add(vid2)
                if vid2 in self.queries:
                    self.positives[topic][vid2].add(vid1)
            # 处理独立样本（负样本）
            else:
                if vid1 in self.queries:
                    self.independent[topic][vid1].add(vid2)
                if vid2 in self.queries:
                    self.independent[topic][vid2].add(vid1)

        exit()

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def get_pairs(self):
        return self.idx, self.q_video, self.a_video, self.hard, self.easy, self.idx2neg        

    def calculate_metric(self, y_true, y_score, gt_len):
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)

    def calculate_mAP(self, query, targets, all_db, positives):
        if query in positives:
            query_gt = set(positives[query]).intersection(all_db)
        else:
            query_gt = set()

        y_true, y_score = [], []
        for target, sim in targets.items():
            if target != query and target in all_db:
                y_true.append(int(target in query_gt))
                y_score.append(float(sim))

        return self.calculate_metric(y_true, y_score, len(query_gt)), y_true, y_score

    def calculate_uAP(self, similarities, all_db, positives, queries):
        y_true, y_score, gt_len = [], [], 0
        for query in queries:
            if query in similarities:
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

    def evaluate(self, similarities, all_db=None, relationship=None, verbose=True):
        mAP, not_found = [], 0
        mAP_dict = {}

        # 如果未指定关系类型，使用默认的合并关系
        if all_db is None:
            all_db = set(self.get_database())

        if relationship is None:
            positives = self.positives
            queries = self.queries
            # 不排除任何视频
        elif relationship == 'event':
            positives = self.positives_event
            queries = list(positives.keys())
            # 排除其他关系类型的视频
            exclude_videos = self.videos_copy.union(self.videos_copy_and_event)
            all_db = all_db - exclude_videos
        elif relationship == 'copy':
            positives = self.positives_copy
            queries = list(positives.keys())
            # 排除其他关系类型的视频
            exclude_videos = self.videos_event.union(self.videos_copy_and_event)
            all_db = all_db - exclude_videos
        elif relationship == 'copy and event':
            positives = self.positives_copy_and_event
            queries = list(positives.keys())
            # 排除其他关系类型的视频
            exclude_videos = self.videos_copy.union(self.videos_event)
            all_db = all_db - exclude_videos
        else:
            raise ValueError("Invalid relationship type: {}".format(relationship))
        # print(type(positives))
        for query in queries:
            if (query not in similarities) or (query not in positives.keys()) or (len(positives[query]) == 1 and positives[query][0] == query) or (len(positives[query]) == 0):
                not_found += 1
            else:
                AP, y_true, y_score = self.calculate_mAP(query, similarities[query], all_db, positives)
                if AP >= 0:
                    mAP += [AP]
                    mAP_dict[query] = {'AP': AP, 'y_true': y_true, 'y_score': y_score}

        uAP = self.calculate_uAP(similarities, all_db, positives, queries)

        if verbose:
            print('=' * 5, 'M2VR Dataset', '=' * 5)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(mAP)))
            print('uAP: {:.4f}'.format(uAP))

            # 计算最高和最低 AP 的查询
            max_AP_query = max(mAP_dict, key=lambda k: mAP_dict[k]['AP'])
            min_AP_query = min(mAP_dict, key=lambda k: mAP_dict[k]['AP'])
            print(f"AP 最高的查询是 '{max_AP_query}'，其 AP 值为 {mAP_dict[max_AP_query]['AP']:.4f}")
            print(f"AP 最低的查询是 '{min_AP_query}'，其 AP 值为 {mAP_dict[min_AP_query]['AP']:.4f}")

            # 1. 处理字典，仅保留 'AP' 字段
            processed_dict = {query: {'AP': entry['AP']} for query, entry in mAP_dict.items()}

            # 2. 将处理后的字典保存为 JSON 文件
            json_file_path = '/data/workspace/M2VR/mAP.json'
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_dict, f, ensure_ascii=False, indent=4)

        return {'M2VR_mAP': np.mean(mAP), 'M2VR_uAP': uAP}

