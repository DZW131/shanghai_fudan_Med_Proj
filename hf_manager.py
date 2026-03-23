#!/usr/bin/env python3
"""
Hugging Face 资源管理工具
用于上传/下载模型权重和数据集
支持两个独立仓库：
- 权重仓库: DZW666/shanghai_fudan_weights
- 数据集仓库: DZW666/shanghai-fudan-cancer-dataset
"""

import os
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, upload_folder, upload_file, snapshot_download


# 默认仓库配置
WEIGHTS_REPO = "DZW666/shanghai_fudan_weights"
DATASET_REPO = "DZW666/shanghai-fudan-cancer-dataset"


class HuggingFaceManager:
    """管理 Hugging Face 上的资源"""

    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        初始化

        Args:
            repo_id: Hugging Face 仓库 ID (格式: username/repo-name)
            token: Hugging Face API token (可选，默认从环境变量读取)
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def upload_checkpoints(self, checkpoints_dir: str = "checkpoints"):
        """上传模型权重到 Hugging Face"""
        checkpoints_path = Path(checkpoints_dir)
        if not checkpoints_path.exists():
            print(f"目录不存在: {checkpoints_dir}")
            return

        # 检查是否有权重文件
        weight_files = list(checkpoints_path.glob("*.pt")) + list(checkpoints_path.glob("*.pth")) + \
                      list(checkpoints_path.glob("*.safetensors")) + list(checkpoints_path.glob("*.bin"))

        if not weight_files:
            print(f"警告: {checkpoints_dir} 目录中没有找到权重文件 (.pt, .pth, .safetensors, .bin)")
            print("跳过上传。如需上传空目录，请使用 --force 参数")
            return

        print(f"正在上传 {len(weight_files)} 个权重文件到 {self.repo_id} ...")
        upload_folder(
            folder_path=str(checkpoints_path),
            repo_id=self.repo_id,
            repo_type="model",
            path_in_repo=".",
            token=self.token,
        )
        print(f"✓ 权重上传完成: {self.repo_id}")

    def upload_datasets(self, datasets_dir: str = "datasets"):
        """上传数据集到 Hugging Face"""
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            print(f"目录不存在: {datasets_dir}")
            return

        print(f"正在上传数据集到 {self.repo_id} ...")
        upload_folder(
            folder_path=str(datasets_path),
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo=".",
            token=self.token,
        )
        print(f"✓ 数据集上传完成: {self.repo_id}")

    def download_checkpoints(self, local_dir: str = "checkpoints"):
        """从 Hugging Face 下载模型权重"""
        print(f"正在从 {self.repo_id} 下载权重到 {local_dir} ...")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            local_dir=local_dir,
            token=self.token,
        )
        print(f"✓ 权重下载完成: {local_dir}")

    def download_datasets(self, local_dir: str = "datasets"):
        """从 Hugging Face 下载数据集"""
        print(f"正在从 {self.repo_id} 下载数据集到 {local_dir} ...")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            token=self.token,
        )
        print(f"✓ 数据集下载完成: {local_dir}")


class ProjectResourceManager:
    """
    项目资源管理器
    自动使用配置的默认仓库
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self.weights_manager = HuggingFaceManager(WEIGHTS_REPO, self.token)
        self.dataset_manager = HuggingFaceManager(DATASET_REPO, self.token)

    def upload_weights(self, checkpoints_dir: str = "checkpoints"):
        """上传权重到权重仓库"""
        self.weights_manager.upload_checkpoints(checkpoints_dir)

    def upload_datasets(self, datasets_dir: str = "datasets"):
        """上传数据集到数据集仓库"""
        self.dataset_manager.upload_datasets(datasets_dir)

    def download_weights(self, local_dir: str = "checkpoints"):
        """从权重仓库下载权重"""
        self.weights_manager.download_checkpoints(local_dir)

    def download_datasets(self, local_dir: str = "datasets"):
        """从数据集仓库下载数据集"""
        self.dataset_manager.download_datasets(local_dir)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hugging Face 资源管理工具 - 复旦医学项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
默认仓库配置:
  权重仓库: DZW666/shanghai_fudan_weights
  数据集仓库: DZW666/shanghai-fudan-cancer-dataset

使用示例:
  # 上传权重
  python hf_manager.py upload-weights

  # 上传数据集
  python hf_manager.py upload-datasets

  # 下载权重
  python hf_manager.py download-weights

  # 下载数据集
  python hf_manager.py download-datasets

  # 使用自定义 Token
  python hf_manager.py upload-weights --token hf_xxxxxx
        """
    )

    parser.add_argument(
        "action",
        choices=["upload-weights", "upload-datasets", "download-weights", "download-datasets",
                 "upload-ckpt", "upload-data", "download-ckpt", "download-data"],
        help="要执行的操作"
    )
    parser.add_argument("--dir", default=None, help="本地目录路径 (默认: checkpoints/ 或 datasets/)")
    parser.add_argument("--token", default=None, help="Hugging Face API Token (默认从 HF_TOKEN 环境变量读取)")
    parser.add_argument("--weights-repo", default=WEIGHTS_REPO, help=f"权重仓库 ID (默认: {WEIGHTS_REPO})")
    parser.add_argument("--dataset-repo", default=DATASET_REPO, help=f"数据集仓库 ID (默认: {DATASET_REPO})")

    args = parser.parse_args()

    # 使用命令行指定的仓库或默认仓库
    token = args.token or os.environ.get("HF_TOKEN")

    # 创建管理器
    weights_manager = HuggingFaceManager(args.weights_repo, token)
    dataset_manager = HuggingFaceManager(args.dataset_repo, token)

    # 执行操作
    if args.action in ["upload-weights", "upload-ckpt"]:
        weights_manager.upload_checkpoints(args.dir or "checkpoints")
    elif args.action in ["upload-datasets", "upload-data"]:
        dataset_manager.upload_datasets(args.dir or "datasets")
    elif args.action in ["download-weights", "download-ckpt"]:
        weights_manager.download_checkpoints(args.dir or "checkpoints")
    elif args.action in ["download-datasets", "download-data"]:
        dataset_manager.download_datasets(args.dir or "datasets")


if __name__ == "__main__":
    main()
