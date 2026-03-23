#!/usr/bin/env python3
"""
Hugging Face 资源管理工具
用于上传/下载模型权重和数据集
"""

import os
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, upload_folder, upload_file, snapshot_download


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

        print(f"正在上传权重到 {self.repo_id}/checkpoints ...")
        upload_folder(
            folder_path=str(checkpoints_path),
            repo_id=self.repo_id,
            repo_type="model",
            path_in_repo="checkpoints",
            token=self.token,
        )
        print("✓ 权重上传完成")

    def upload_datasets(self, datasets_dir: str = "datasets"):
        """上传数据集到 Hugging Face"""
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            print(f"目录不存在: {datasets_dir}")
            return

        print(f"正在上传数据集到 {self.repo_id}/datasets ...")
        upload_folder(
            folder_path=str(datasets_path),
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo="datasets",
            token=self.token,
        )
        print("✓ 数据集上传完成")

    def download_checkpoints(self, local_dir: str = "checkpoints"):
        """从 Hugging Face 下载模型权重"""
        print(f"正在从 {self.repo_id} 下载权重到 {local_dir} ...")
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=["checkpoints/*"],
            token=self.token,
        )
        print("✓ 权重下载完成")

    def download_datasets(self, local_dir: str = "datasets"):
        """从 Hugging Face 下载数据集"""
        print(f"正在从 {self.repo_id} 下载数据集到 {local_dir} ...")
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["datasets/*"],
            token=self.token,
        )
        print("✓ 数据集下载完成")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Hugging Face 资源管理工具")
    parser.add_argument("--repo-id", required=True, help="Hugging Face 仓库 ID (格式: username/repo-name)")
    parser.add_argument("--token", default=None, help="Hugging Face API Token")
    parser.add_argument("--action", choices=["upload-ckpt", "upload-data", "download-ckpt", "download-data"],
                        required=True, help="要执行的操作")
    parser.add_argument("--dir", default=None, help="本地目录路径")

    args = parser.parse_args()

    manager = HuggingFaceManager(repo_id=args.repo_id, token=args.token)

    if args.action == "upload-ckpt":
        manager.upload_checkpoints(args.dir or "checkpoints")
    elif args.action == "upload-data":
        manager.upload_datasets(args.dir or "datasets")
    elif args.action == "download-ckpt":
        manager.download_checkpoints(args.dir or "checkpoints")
    elif args.action == "download-data":
        manager.download_datasets(args.dir or "datasets")


if __name__ == "__main__":
    main()
