# 上海复旦肿瘤TROP2科研项目 - 细胞分割模型

## 一、项目概述

### 1.1 课题背景

本项目基于 Meta AI 的 SAM 2.1 (Segment Anything Model 2.1) 基础模型，针对**肿瘤细胞膜 (Cell Membrane, ME)** 和**细胞核 (Cell Nucleus, NU)** 进行分割任务的微调训练。

### 1.2 任务目标

| 任务 | 描述 | 标签 |
|------|------|------|
| 细胞膜分割 | 分割肿瘤细胞的外层细胞膜 | 肿瘤细胞膜 |
| 细胞核分割 | 分割肿瘤细胞的细胞核 | 肿瘤细胞核 |

---

## 二、技术方案

### 2.1 基础模型

- **模型**: SAM 2.1 Base+ (Hiera-B+)
- **参数量**: 80.8M
- **预训练权重**: `sam2.1_hiera_base_plus.pt`
- **来源**: https://github.com/facebookresearch/sam2

### 2.2 技术路线

```
Image Encoder (Hiera)
       ↓
  [Feature Maps + FPN Neck]
       ↓
┌──────────────────┐
│  Memory Attention │ (4层 RoPEAttention)
└──────────────────┘
       ↓
┌─────────────────────────────────────────┐
│    Mask Decoders (多解码器架构)          │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ Decoder 0    │  │ Decoder 1    │      │
│  │ (细胞膜 ME)  │  │ (细胞核 NU)  │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
       ↓
  二进制掩码输出 (sigmoid激活)
```

### 2.3 训练策略

| 策略 | 单任务训练 | 联合训练 |
|------|-----------|----------|
| 配置 | `trop2_me_only.yaml` / `trop2_nu_only.yaml` | `trop2_me_nu.yaml` |
| 解码器数量 | 1 (multitask_num=1) | 2 (multitask_num=2) |
| 训练难度 | 较低 | 较高 |
| 最终效果 | **较好** | 待优化 |

**最终采用方案**: 单任务分离训练

---

## 三、数据集配置

### 3.1 数据来源

```
datasets/trop2/
├── train/
│   ├── JPEGImages/              # 原始图像
│   │   └── <video_id>/
│   │       └── 0000.png
│   ├── Annotations_mask_me/     # 细胞膜标注
│   │   └── <video_id>.png
│   └── Annotations_mask_nu/     # 细胞核标注
│       └── <video_id>.png
├── val/
└── test/
```

### 3.2 数据规模

| 数据集 | 图像数量 | 分辨率 |
|--------|----------|--------|
| 训练集 | 多帧序列 | 1024×1024 |
| 验证集 | - | 1024×1024 |
| 测试集 | 多帧序列 | 1024×1024 |

### 3.3 数据增强

- RandomHorizontalFlip (consistent_transform: True)
- RandomAffine (degrees: 25, shear: 20)
- ColorJitter (brightness: 0.1, contrast: 0.03-0.05)
- RandomGrayscale (p: 0.05)

---

## 四、训练配置

### 4.1 单任务训练 - 细胞膜 (ME)

**配置文件**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_trop2_me_only.yaml`

```yaml
scratch:
  resolution: 1024
  train_batch_size: 1
  num_train_workers: 4
  num_frames: 1
  max_num_objects: 3
  base_lr: 1.0e-5
  vision_lr: 6.0e-06
  num_epochs: 200

dataset:
  img_folder: datasets/trop2/train/JPEGImages
  gt_folder: datasets/trop2/train/Annotations_mask_me

launcher:
  experiment_log_dir: checkpoints/trop2_me_only_train
  gpus_per_node: 1
```

### 4.2 单任务训练 - 细胞核 (NU)

**配置文件**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_trop2_nu_only.yaml`

```yaml
scratch:
  resolution: 1024
  train_batch_size: 1
  num_train_workers: 4
  num_frames: 1
  max_num_objects: 3
  base_lr: 1.0e-5
  vision_lr: 6.0e-06
  num_epochs: 200

dataset:
  img_folder: datasets/trop2/train/JPEGImages
  gt_folder: datasets/trop2/train/Annotations_mask_nu

launcher:
  experiment_log_dir: checkpoints/trop2_nu_only_train
  gpus_per_node: 1
```

### 4.3 训练命令

```bash
# 训练细胞膜模型
python train.py --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_trop2_me_only.yaml --devices "0"

# 训练细胞核模型
python train.py --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_trop2_nu_only.yaml --devices "0"
```

---

## 五、评估指标

### 5.1 指标说明

| 指标 | 全称 | 说明 |
|------|------|------|
| BDQ | Boundary Quality | 边界质量评分 |
| BSQ | Segmentation Quality | 分割质量评分 |
| BPQ | Prompt Quality | 提示点质量评分 |
| AJI | Aggregated Jaccard Index | 聚合Jaccard指数 |

### 5.2 测试结果

| 模型 | BDQ | BSQ | BPQ | AJI |
|------|-----|-----|-----|-----|
| **细胞膜 (ME)** | 0.9007 | 0.7500 | 0.6762 | 0.6908 |
| **细胞核 (NU)** | 0.9216 | 0.7329 | 0.6761 | 0.7144 |

**结论**: 单任务训练方案在细胞膜和细胞核分割任务上均取得了较好的效果。

---

## 六、推理使用

### 6.1 推理配置

| 模型 | 配置名 | checkpoint路径 |
|------|--------|----------------|
| 细胞膜 | `trop2_me_only` | `checkpoints/trop2_me_only_train/checkpoints/checkpoint.pt` |
| 细胞核 | `trop2_nu_only` | `checkpoints/trop2_nu_only_train/checkpoints/checkpoint.pt` |

### 6.2 推理命令

```bash
# 推理细胞膜
python infer.py --img_path <image_path> --model trop2_me_only --save_res

# 推理细胞核
python infer.py --img_path <image_path> --model trop2_nu_only --save_res

# 批量评估
python infer.py --mode test --eval --model trop2_me_only --save_res
python infer.py --mode test --eval --model trop2_nu_only --save_res
```

### 6.3 推理结果示例

**细胞膜分割**: 轮廓清晰，边界质量良好
**细胞核分割**: 核心区域分割准确，细节保留较好

---

## 七、仓库信息

### 7.1 GitHub 仓库

**地址**: https://github.com/DZW131/shanghai_fudan_Med_Proj

**主要文件**:
- `train.py` - 训练入口
- `infer.py` - 推理入口
- `sam2/` - SAM 2.1 核心代码
- `training/` - 训练相关代码
- `sam2/configs/sam2.1_training/` - 训练配置文件

**提交历史**:
- `36a6777` - fix: Update trop2_me_only to use multitask_num=2
- `b7e64a0` - fix: Correct checkpoint save path based on experiment_log_dir
- `63cfeb6` - feat: Add single-task training configs and fix multi-task training

### 7.2 HuggingFace 仓库

**地址**: https://huggingface.co/DZW666/shanghai_fudan_weights

**模型文件**:

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `sam2.1_hiera_base_plus.pt` | SAM 2.1 Base+ 官方预训练权重 | 324MB |
| `sam2_trop2_me_only_finetuned.pth` | **细胞膜微调权重** | 324MB |
| `sam2_trop2_nu_only_finetuned.pth` | **细胞核微调权重** | 324MB |
| `sam2_trop2_me_nu_finetuned.pth` | ME+NU联合训练权重 | 324MB |
| `3_23_sam2_trop2_me_nu_finetuned.pth` | 历史权重 | 324MB |

**使用示例**:
```python
from huggingface_hub import hf_hub_download

# 下载模型权重
me_ckpt = hf_hub_download(
    repo_id='DZW666/shanghai_fudan_weights',
    filename='sam2_trop2_me_only_finetuned.pth'
)
```

---

## 八、技术要点总结

### 8.1 解决的问题

1. **多任务训练中 NU 分割效果差**: 采用单任务分离训练策略
2. **Hydra 配置初始化问题**: 修复了 train.py 和 build_sam.py 中的 Hydra 初始化
3. **checkpoint 保存路径问题**: 修复了硬编码路径，改为基于 experiment_log_dir 动态生成
4. **推理结果污染训练集**: 清理推理结果文件避免影响训练

### 8.2 关键代码修改

**sam2/build_sam.py**: 解码器权重复制
```python
# Copy decoder 0 weights to decoder 1 for multitask training
sd_filtered_decoder0_to_1 = {
    k.replace("sam_mask_decoders.0.", "sam_mask_decoders.1."): v
    for k, v in sd_filtered.items()
    if "sam_mask_decoders.0." in k
}
sd_filtered.update(sd_filtered_decoder0_to_1)
```

**train.py**: checkpoint 保存路径
```python
exp_dir = cfg.launcher.experiment_log_dir
ckpt_path = f"{exp_dir}/checkpoints/checkpoint.pt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
torch.save(checkpoint, ckpt_path)
```

---

## 九、后续优化方向

1. **联合训练优化**: 尝试用单任务模型作为初始化进行联合训练
2. **数据增强**: 针对细胞核较小的问题，增加针对性增强
3. **损失函数**: 尝试 Focal Loss 或 Dice Loss 优化小目标分割
4. **测试时增强 (TTA)**: 水平翻转等测试时增强策略

---

## 十、参考文献

1. Ravi, N., et al. "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714, 2024.
2. Meta AI SAM 2.1: https://github.com/facebookresearch/sam2
3. HuggingFace: https://huggingface.co/DZW666/shanghai_fudan_weights

---

**汇报人**: [姓名]
**汇报日期**: 2026-03-24
