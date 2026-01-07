# [NeurIPS 2025] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence

MUVR is a benchmark codebase for Multi-Modal Untrimmed Video Retrieval. It aims to evaluate the performance of various Video-Language Models (VLMs) on video retrieval tasks, supporting multiple retrieval modes (video, text, video+text) and tag-based filtering mechanisms.

## Features

- **Multi-model Support**: Supports mainstream VLMs, including BLIP, BLIP2, SigLIP, OpenCLIP, OAI-CLIP, Meta-CLIP, EVA-CLIP, etc.
- **Flexible Evaluation Modes**:
  - **Video-to-Video (v)**: Retrieval using video queries.
  - **Text-to-Video (t)**: Retrieval using text queries.
  - **Text+Video-to-Video (tv)**: Retrieval combining text and video queries.
  - **Tag-based Filtering**: Supports filtering retrieval results based on tags.
- **Feature Extraction**: Provides a unified interface to extract features from videos.

## Environment Preparation

Please ensure the following dependencies are installed in your environment:

```bash
torch torchvision numpy tqdm h5py opencv-python
clip transformers open_clip_torch salesforce-lavis timm
```

## Project Structure

```
MUVR/
├── datasets/               # Dataset loading and processing logic
│   ├── transforms/         # Data augmentation and transformation
│   ├── cuvr.py             # Our MUVR dataset
│   └── ...
├── scripts/                # Running scripts
│   ├── 1_extract_features.sh
│   ├── 2_muvr_base.sh
│   └── 3_muvr_filter.sh
├── evaluate_all_vlm_mul.py # Main evaluation program
├── extract_all_features.py # Main feature extraction program
├── utils.py                # Utility functions
└── README.md               # Project documentation
```

## Quick Start

### 1. Data Preparation

**Download Data**: You can download the original videos and pre-extracted features from Hugging Face: [https://huggingface.co/datasets/debby0527/MUVR](https://huggingface.co/datasets/debby0527/MUVR).

Please place your video data in the specified directory. By default, the script reads video data from `/MUVR/videos/all_videos`. You can modify the `base_input` variable in the script to specify your data path.

### 2. Feature Extraction

First, use `extract_all_features.py` to extract features from videos. You can run the provided script directly:

```bash
bash scripts/1_extract_features.sh
```

**Note**: Please check and modify the `base_input` and `base_output` paths, as well as the list of models `models` to be used in `scripts/1_extract_features.sh` before running.

The script will iterate through the specified models and topics, extracting video features and saving them to `/MUVR/features/VLMs_1_15` (default path).

### 3. Base Evaluation

After feature extraction is complete, run the evaluation script to test model performance:

```bash
bash scripts/2_muvr_base.sh
```

The script will evaluate retrieval performance under different modes (t, v, tv) and output the results to `./VLM_results`.

### 4. Tag-based Filtering Evaluation

If you need to evaluate performance based on tag filtering, you can run:

```bash
bash scripts/3_muvr_filter.sh
```

## Script Parameter Description

### `extract_all_features.py`

- `--model`: Specify the model name to use (e.g., EVA-CLIP, OpenCLIP).
- `--input_folder`: Video input directory.
- `--output_folder`: Feature output directory.
- `--num_segments`: Number of video segments.
- `--frames_per_segment`: Number of frames extracted per segment.
- `--topic`: Dataset topic.
- `--gpu_ids`: Specify the GPU IDs to use.

### `evaluate_all_vlm_mul.py`

- `--dataset_hdf5`: Feature file path.
- `--topic`: Evaluation topic.
- `--mode`: Evaluation mode (`t`, `v`, `tv`, `tag`).
- `--model_name`: Model name.
- `--tag_mode`: Tag mode parameter.
- `--output_dir`: Result output directory.

## Citation

```bash
@article{feng2025muvr,
  title={MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence},
  author={Feng, Yue and Hu, Jinwei and Lu, Qijia and Niu, Jiawei and Tan, Li and Yuan, Shuo and Yan, Ziyi and Jia, Yizhen and He, Qingzhi and Ge, Shiping and others},
  journal={arXiv preprint arXiv:2510.21406},
  year={2025}
}
```

