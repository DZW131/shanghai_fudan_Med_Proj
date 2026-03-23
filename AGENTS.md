# Qoder Agent 记忆文件 - 复旦医学项目

## 项目概览
- 项目名称: 上海复旦医学图像分割项目 (SAM2)
- 服务器路径: /root/sam2
- 创建时间: 2026-03-23

## GitHub 配置
- 仓库地址: https://github.com/DZW131/shanghai_fudan_Med_Proj
- 远程分支: origin/main
- 认证方式: SSH Key
- SSH Key 位置: ~/.ssh/id_ed25519
- 已配置用户: ziwen du <3104833059@qq.com>

## Hugging Face 配置

### 权重仓库
- 仓库ID: DZW666/shanghai_fudan_weights
- 类型: Model
- 用途: 存储模型权重文件 (.pt, .pth, .safetensors, .bin)
- 当前状态: 空（预留未来使用）

### 数据集仓库
- 仓库ID: DZW666/shanghai-fudan-cancer-dataset
- 类型: Dataset
- 用途: 存储训练数据集
- 当前内容:
  - PanNuke/ (~944MB)
  - trop2/ (~510MB)
- 总大小: ~1.5GB, 1589个文件
- 上传时间: 2026-03-23

### Token 配置
- Token: [从环境变量 HF_TOKEN 读取]
- 权限: Write (classic token)
- 环境变量: HF_TOKEN
- 本地设置: 见 setup_hf_env.sh

## 常用命令速查

### Git 操作
```bash
cd /root/sam2
git add .
git commit -m "描述"
git push origin main
```

### Hugging Face 操作
```bash
cd /root/sam2
source setup_hf_env.sh  # 加载环境变量

# 上传
python hf_manager.py upload-weights
python hf_manager.py upload-datasets

# 下载
python hf_manager.py download-weights
python hf_manager.py download-datasets
```

### 环境设置
```bash
# Token 已保存在 setup_hf_env.sh 中
source /root/sam2/setup_hf_env.sh
```

## 项目结构
```
sam2/
├── .github/workflows/     # GitHub Actions
├── checkpoints/           # 模型权重 (Git忽略, HF管理)
├── datasets/              # 数据集 (Git忽略, HF管理)
├── sam2/                  # 源代码
├── hf_manager.py          # HF资源管理脚本
├── setup_hf_env.sh        # 环境配置脚本 (含Token)
├── PROJECT_SETUP.md       # 项目设置文档
└── AGENTS.md              # 本文件
```

## 注意事项
- 大文件不要提交到 GitHub（已被 .gitignore 排除）
- 权重和数据集分开管理在两个 HF 仓库
- Token 保存在 setup_hf_env.sh，请妥善保管
- SSH Key 已配置，可直接推送代码

## 相关链接
- GitHub: https://github.com/DZW131/shanghai_fudan_Med_Proj
- HF 权重: https://huggingface.co/DZW666/shanghai_fudan_weights
- HF 数据集: https://huggingface.co/DZW666/shanghai-fudan-cancer-dataset
