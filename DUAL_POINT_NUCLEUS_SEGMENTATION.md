# Dual-Point Training for Cell Nucleus Segmentation

## Overview

This document describes the dual-point training strategy implemented for cell nucleus (NU) segmentation using SAM2.

## Training Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_pt_to_sample` | 2 | Number of points sampled per prompt during training |
| `multimask_max_pt_num` | 2 | Maximum points for multimask output |
| `prob_to_use_pt_input_for_train` | 0.5 | Probability of using point input during training |
| `prob_to_use_box_input_for_train` | 0.5 | Probability of using box input during training |
| `prob_to_sample_from_gt_for_train` | 0.1 | Probability of sampling from GT mask instead of error regions |

### Point Sampling Mechanism

During training, two points are independently sampled from error regions:
- **False Negative (FN) regions** → Positive points (label=1)
- **False Positive (FP) regions** → Negative points (label=0)

The two points are independent, meaning they could be:
- 2 positive points (both inside the target)
- 2 negative points (both in background)
- 1 positive + 1 negative (mixed)

## Code Changes

### 1. `sam2/modeling/sam2_utils.py`
Added `num_pt` parameter to `get_next_point()`:
```python
def get_next_point(gt_masks, pred_masks, method, num_pt=1):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks, num_pt=num_pt)
```

### 2. `training/model/sam2.py`
- Added `num_pt_to_sample` parameter
- Passed `num_pt` to `get_next_point()`

### 3. `sam2/configs/sam2.1_training/sam2.1_hiera_b+_trop2_nu_only.yaml`
```yaml
num_pt_to_sample: 2  # Dual-point training
multimask_max_pt_num: 2  # Enable multimask for dual-point
```

### 4. `training/utils/checkpoint_utils.py`
Fixed checkpoint loading to skip `maskmem_tpos_enc`:
```python
state_dict = {k:v for k, v in state_dict.items() if "maskmem_tpos_enc" not in k}
```

## Training Results

### Test Set Metrics
| Metric | Value |
|--------|-------|
| Dice | 0.9317 |
| BQ (Boundary Quality) | 0.7340 |
| BPQ (Boundary Precision Quality) | 0.6843 |
| AJI (Aggregated Jaccard Index) | 0.7178 |

### Training Details
- **Epochs**: 200
- **Batch Size**: 1
- **Resolution**: 1024x1024
- **Training Time**: ~2 hours
- **Checkpoint Size**: ~309 MB (model only)

## Files

### Checkpoints (HuggingFace - `trop2_nu_dual_point` branch)
- `trop2_nu_only_train_dual_point/checkpoints/checkpoint.pt` - Full checkpoint (~869 MB)
- `trop2_nu_only_train_dual_point/checkpoints/checkpoint_model_only.pt` - Model only (~309 MB)

### Config
- `trop2_nu_only_train_dual_point/sam2.1_hiera_b+_trop2_nu_only.yaml`

## Usage

### Inference
```bash
python infer.py --img_path <image_path> --model trop2_nu_only_dual_point --save_res
```

### Evaluation
```bash
python infer.py --mode test --model trop2_nu_only_dual_point --eval
```

## Comparison: Single-Point vs Dual-Point

| Aspect | Single-Point | Dual-Point |
|--------|--------------|------------|
| Points per prompt | 1 | 2 |
| Spatial constraints | Limited | More comprehensive |
| Error correction signals | Single location | Multiple locations |
| Training time | Similar | Similar |
| Model size | Same | Same |

## Notes

- The dual-point training provides richer spatial information for the model
- Two points are sampled independently from error regions
- Each point can be positive (inside target) or negative (in background)
- The model learns to utilize multiple spatial cues for segmentation
