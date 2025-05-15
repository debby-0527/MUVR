import os
import argparse
import h5py
from tqdm import tqdm
import ..utils  # 确保 utils 模块在 Python 路径中
import logging

def collect_mp4_files(folder_path):
    """
    遍历指定文件夹及其子文件夹，收集所有 MP4 文件的路径。

    Args:
        folder_path (str): 需要遍历的根文件夹路径。

    Returns:
        list: 所有 MP4 文件的绝对路径列表。
    """
    mp4_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                full_path = os.path.join(root, file)
                mp4_files.append(full_path)
    return mp4_files

def extract_and_save_features(mp4_files, hdf5_path, fps=30, crop=True, resize=(224, 224)):
    """
    提取每个 MP4 文件的视频特征并保存到 HDF5 文件中。

    Args:
        mp4_files (list): MP4 文件路径列表。
        hdf5_path (str): 要保存的 HDF5 文件路径。
        fps (int, optional): 帧率。默认值为 30。
        crop (bool, optional): 是否裁剪视频。默认值为 True。
        resize (tuple, optional): 调整视频大小。默认值为 (224, 224)。
    """
    # 使用 h5py 打开 HDF5 文件
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for video_path in tqdm(mp4_files, desc="Processing videos"):
            try:
                # 提取视频名称（不包含扩展名）
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                
                # 调用特征提取函数
                feature = utils.load_video_ffmpeg(video_path, fps=fps, crop=crop, resize=resize)
                
                # 检查 feature 是否为 numpy 数组
                if not isinstance(feature, (list, tuple)):
                    import numpy as np
                    feature = np.array(feature, dtype=np.float32)
                else:
                    feature = np.array(feature, dtype=np.float32)
                
                # 保存到 HDF5 文件中
                hdf5_file.create_dataset(video_name, data=feature, dtype='float32', compression="gzip")
            
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {e}")
                continue

def parse_arguments():
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description="提取视频特征并保存到 HDF5 文件。")
    parser.add_argument('--input_folder', type=str, required=True, help='包含 MP4 文件的根文件夹路径。')
    parser.add_argument('--output_file', type=str, default='m2vr.hdf5', help='输出 HDF5 文件的路径。默认值为 "m2vr.hdf5"。')
    parser.add_argument('--fps', type=int, default=1,
                        help='Fps value for video loading.')
    # --fps 参数，指定视频加载时的帧率
    parser.add_argument('--crop', type=int, default=224,
                        help='Crop value for video loading.')
    # --crop 参数，指定视频加载时的裁剪尺寸
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize value for video loading.')
    # --resize 参数，指定视频加载时的重置尺寸
    return parser.parse_args()

def setup_logging():
    """
    设置日志配置。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    """
    主函数，执行特征提取和保存操作。
    """
    setup_logging()
    args = parse_arguments()

    input_folder = args.input_folder
    output_file = args.output_file
    fps = args.fps
    crop = args.crop
    resize = tuple(args.resize)

    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        logging.error(f"输入文件夹不存在: {input_folder}")
        return

    logging.info(f"开始遍历文件夹: {input_folder}")
    mp4_files = collect_mp4_files(input_folder)
    logging.info(f"找到 {len(mp4_files)} 个 MP4 文件。")

    if len(mp4_files) == 0:
        logging.warning("未找到任何 MP4 文件，程序即将退出。")
        return

    logging.info(f"开始提取视频特征并保存到 {output_file}...")
    extract_and_save_features(mp4_files, output_file, fps=fps, crop=crop, resize=resize)
    logging.info("特征提取和保存完成。")

if __name__ == "__main__":
    main()