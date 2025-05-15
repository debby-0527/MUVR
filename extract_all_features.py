import torch
import argparse
import os
import glob
import multiprocessing
import numpy as np
import torch.multiprocessing as mp
import pickle
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EvaluationDataset
from datasets.generators import VLMGenerator
from InternV2_config import Config, eval_dict_leaf
from InternV2_utils import setup_internvideo2
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

import warnings; warnings.filterwarnings('ignore')

import clip
from lavis.models import load_model_and_preprocess
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import timm, open_clip

class BaseFeatureExtractor:   
    SIZE_MAP = {
        "OAI-CLIP": 288,     
        "BLIP": 224,        
        "BLIP2": 364,        
        "SigLIP": 384,       
        "EVA-CLIP": 224,     
        "OpenCLIP": 224,     
        "Meta-CLIP": 336     
    }
    
    def __init__(self, args):
        self.args = args
        self.model_name = None
        self.device = None

    @classmethod
    def create(cls, model_name, args):
        if model_name == "OAI-CLIP":
            return OAI_CLIPFeatureExtractor(args)
        elif model_name == "BLIP":
            return BLIPFeatureExtractor(args)
        elif model_name == "BLIP2":
            return BLIP2FeatureExtractor(args) 
        elif model_name == "SigLIP":
            return SigLIPFeatureExtractor(args)
        elif model_name == "EVA-CLIP":
            return EVA_CLIPFeatureExtractor(args)
        elif model_name == "OpenCLIP":
            return OpenCLIPFeatureExtractor(args)
        elif model_name == "Meta-CLIP":
            return Meta_CLIPFeatureExtractor(args)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def initialize_generator(self, video_list, video_dict):
        return VLMGenerator(
            self.args.input_folder,
            video_list,
            model_type=self.model_name,
            size_t=self.SIZE_MAP[self.model_name],
            video_dict=video_dict,
            device=self.device,
            num_segments=self.args.num_segments,
            frames_per_segment=self.args.frames_per_segment,
        )

    def base_setup(self, gpu_id):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        # print(f"Process {os.getpid()} using GPU {gpu_id}")

    def load_model(self):
        raise NotImplementedError

    def extract_image_features(self, model, frames):
        raise NotImplementedError

    def extract_text_features(self, model, text):
        raise NotImplementedError

    def process_image_features(self, features):
        return features

    @torch.no_grad()
    def check(self, features):
        # 类型检查
        if not isinstance(features, np.ndarray):
            raise TypeError(f"输入必须是numpy数组，当前类型：{type(features)}")
            
        ndim = features.ndim
        
        # 处理一维情况
        if ndim == 1:
            return np.expand_dims(features, axis=0)
            
        # 处理有效维度情况
        elif 2 <= ndim <= 3:
            batch_dim = features.shape[0]
            # if batch_dim not in (1, 15):
            #     raise ValueError(
            #         f"非法批量维度{batch_dim}（形状{features.shape}），"
            #         f"要求：第一维度必须为1或15"
            #     )
            return features
            
        # 处理非法维度情况
        else:
            raise ValueError(
                f"不支持的维度数量{ndim}（形状{features.shape}），"
                f"仅接受1-3维特征"
            )
        
    @torch.no_grad()
    def extract_features_worker(self, gpu_id, video_list, video_dict):
        self.base_setup(gpu_id)
        model = self.load_model()
        generator = self.initialize_generator(video_list, video_dict)
        loader = DataLoader(generator, num_workers=self.args.workers, batch_size=None)

        # print(f'\n> GPU {gpu_id} starting feature extraction')
        last_t_shape = (0,0)
        pbar = tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id)
        for video_tensor, query_dict, video_id in pbar:
            if video_tensor is None:
                continue

            # Process video frames
            bs, nf, c, h, w = video_tensor.shape
            frames = video_tensor.view(bs * nf, c, h, w)
            features = self.check(self.extract_image_features(model, frames))
            processed_features = self.process_image_features(features)
            
            # Save features
            v_shape, t_shape = self.save_features(model, video_id, processed_features, query_dict)
            if t_shape is not None:
                pbar.set_postfix(video=video_id[:12], V=v_shape, T=t_shape)
                last_t_shape = t_shape
                assert last_t_shape[-1] == v_shape[-1]
                # exit()
            else:
                pbar.set_postfix(video=video_id[:12], V=v_shape, T=last_t_shape)

    def save_features(self, model, video_id, features, query_dict):
        os.makedirs(self.args.output_folder, exist_ok=True)
        npy_path = os.path.join(self.args.output_folder, f"{video_id}.npy")
        np.save(npy_path, features)
        v_shape, t_shape = features.shape, None
        if query_dict and query_dict.get('query_prompt', ''):
            t_shape = self.save_text_features(model, video_id, query_dict)

        return v_shape, t_shape

    def save_text_features(self, model, video_id, query_dict):
        npy_text_path = os.path.join(self.args.output_folder, f"{video_id}_query_dict.pkl")
        t_feat = self.check(self.extract_text_features(model, query_dict['query_prompt']))
        output_dict = {
            'query_prompt': t_feat,
            'tags': {tag: self.check(self.extract_text_features(model, tag)) for tag in query_dict['tags']}
        }
        with open(npy_text_path, 'wb') as f:
            pickle.dump(output_dict, f)
        return t_feat.shape

    def clean_text(self, text):
        cleaned = re.sub(r'\s*<mask[^>]*>\s*', ' ', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

class OAI_CLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "OAI-CLIP"

    def load_model(self):
        model, _ = clip.load("RN50x4", device=self.device) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        return model.eval()

    def extract_image_features(self, model, frames):
        return model.encode_image(frames).cpu().numpy()

    def extract_text_features(self, model, text):
        with torch.no_grad():
            tokenized = clip.tokenize([self.clean_text(text)], truncate=True).to(self.device)
            return model.encode_text(tokenized).cpu().numpy()[0]

class BLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "BLIP"
        
    def load_model(self):
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_feature_extractor",
            model_type="base",
            is_eval=True,
            device=self.device
        )
        return model

    def extract_image_features(self, model, frames):
        with torch.no_grad():
            features = model.extract_features({"image": frames}, mode="image")
            return features.image_embeds_proj.cpu().numpy()

    def process_image_features(self, features):
        return features.reshape(-1, 197, 256)[:, 0, :]  # 适配BLIP的(帧数, 16, 256)输出

    def extract_text_features(self, model, text):
        with torch.no_grad():
            features = model.extract_features(
                {"text_input": [self.clean_text(text)]}, 
                mode="text"
            )
            return features.text_embeds_proj[:, 0, :].cpu().numpy()[0]

# BLIP2输出维度 （帧数，32，256）
class BLIP2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "BLIP2"

    def load_model(self):
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="coco",
            is_eval=True,
            device=self.device
        )
        return model

    def extract_image_features(self, model, frames):
        with torch.no_grad():
            features = model.extract_features({"image": frames}, mode="image")
            return features.image_embeds_proj.cpu().numpy()

    def process_image_features(self, features):
        return features.reshape(1, -1, 32, 256)[0]  # Adjust for BLIP2's output shape

    def extract_text_features(self, model, text):
        with torch.no_grad():
            features = model.extract_features(
                {"text_input": [self.clean_text(text)]}, 
                mode="text"
            )
            return features.text_embeds_proj[:, 0, :].cpu().numpy()[0]

class SigLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "SigLIP"
        # 分离加载图像处理器和文本分词器
        self.image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")

    def load_model(self):
        # 明确指定加载视觉模型
        model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            # 不再需要trust_remote_code（自v4.37起官方支持）
        )
        return model.to(self.device).eval()

    def extract_image_features(self, model, frames):
        """
        处理动态分辨率图像输入（保持原始宽高比）
        - 自动分割为384x384的patch网格
        """
        with torch.no_grad():
            # 使用图像专用处理器
            # inputs = self.image_processor(
            #     images=frames,
            #     return_tensors="pt",
            #     padding=True,  # 动态填充保持宽高比
            # ).to(self.device)
            
            # 直接通过模型前向传播获取特征
            outputs = model.vision_model(pixel_values=frames)

            # 获取池化后的输出作为图像特征
            # patch_features = visual_output.pooler_output  # 形状为 [batch_size, hidden_size]
            return outputs.pooler_output.cpu().numpy()

    def extract_text_features(self, model, text):
        """
        多语言文本编码（修正后的实现）
        - 使用正确的64 tokens长度限制
        - 移除非必要的文本清理
        """
        with torch.no_grad():
            # 使用独立的文本分词器
            inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=64,          # 官方要求的固定长度
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 通过文本编码器获取特征
            outputs = model.text_model(**inputs)
            return outputs.pooler_output[0].cpu().numpy()

class EVA_CLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "EVA-CLIP"
        self.tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")

    def load_model(self):
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-E-14-plus", 
            pretrained="laion2b_s9b_b144k"
        )
        return model.to(self.device).eval()

    def extract_image_features(self, model, frames):
        with torch.no_grad():
            return model.encode_image(frames).cpu().numpy()

    def extract_text_features(self, model, text):
        
        with torch.no_grad():
            text = self.tokenizer([self.clean_text(text)]).to(self.device)
            return model.encode_text(text).cpu().numpy()[0]

class OpenCLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "OpenCLIP"

    def load_model(self):
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", 
            pretrained="laion2b_s32b_b79k"
        )
        return model.to(self.device).eval()

    def extract_image_features(self, model, frames):
        with torch.no_grad():
            return model.encode_image(frames).cpu().numpy()

    def extract_text_features(self, model, text):
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        with torch.no_grad():
            text = tokenizer([self.clean_text(text)]).to(self.device)
            return model.encode_text(text).cpu().numpy()[0]

class Meta_CLIPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "Meta-CLIP"
        
    def load_model(self):
        model, _ = clip.load("ViT-L/14@336px", device=self.device) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        return model.eval()

    def extract_image_features(self, model, frames):
        return model.encode_image(frames).cpu().numpy()

    def extract_text_features(self, model, text):
        with torch.no_grad():
            tokenized = clip.tokenize([self.clean_text(text)], truncate=True).to(self.device)
            return model.encode_text(tokenized).cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='Unified Feature Extraction')
    parser.add_argument('--model', type=str, required=True,
                       choices=["OAI-CLIP", "BLIP", "BLIP2", "SigLIP", "EVA-CLIP", "OpenCLIP", "Meta-CLIP"],
                       help='Feature extraction model to use')
    parser.add_argument('--dataset', type=str, default='CUVR',
                       choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "EVVE", "VCDB", "M2VR", "CUVR"],
                       help='Evaluation dataset name')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Root directory containing video files')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output directory for feature files')
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of data loading workers per GPU')
    parser.add_argument('--gpu_ids', type=str, default='0',
                       help='Comma-separated list of GPU IDs (e.g. "0,1,2")')
    parser.add_argument('--topic', type=str, default=None,
                       help='Specific topic for dataset processing')
    parser.add_argument('--num_segments', type=int, default=1,
                       help='Number of temporal segments')
    parser.add_argument('--frames_per_segment', type=int, default=8,
                       help='Frames per temporal segment')

    args = parser.parse_args()
    args.gpus = [int(gid) for gid in args.gpu_ids.split(',')]

    # Validate input
    if not os.path.isdir(args.input_folder):
        raise ValueError(f"Input directory not found: {args.input_folder}")

    # Get video files
    video_paths = glob.glob(os.path.join(args.input_folder, "*.mp4"))
    if not video_paths:
        raise ValueError("No MP4 files found in input directory")

    # Create video dictionary
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
    video_dict = {vid: p for vid, p in zip(video_ids, video_paths)}

    # Filter already processed videos
    pending_videos = [vid for vid in video_ids 
                     if not os.path.exists(os.path.join(args.output_folder, f"{vid}.npy"))]

    # pending_videos = [vid for vid in video_ids]

    # Load dataset metadata
    dataset_class = EvaluationDataset[args.dataset.upper().replace('-', '_')].get_dataset()
    text_dict = dataset_class.get_query_text(topics=args.topic)

    # Enhance video dict with text metadata
    for vid in pending_videos:
        video_dict[vid] = {
            'video_path': video_dict[vid],
            'query_text': text_dict.get(vid, '')
        }

    # Split work across GPUs
    num_gpus = len(args.gpus)
    video_lists = [[] for _ in range(num_gpus)]
    for idx, vid in enumerate(pending_videos):
        video_lists[idx % num_gpus].append(vid)

    # Initialize feature extractor
    extractor = BaseFeatureExtractor.create(args.model, args)

    # Start processing
    processes = []
    for gpu_idx, gpu_id in enumerate(args.gpus):
        if not video_lists[gpu_idx]:
            continue

        p = multiprocessing.Process(
            target=extractor.extract_features_worker,
            args=(gpu_id, video_lists[gpu_idx], video_dict)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    print("Feature extraction completed successfully.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()