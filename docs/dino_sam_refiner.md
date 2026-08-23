# DINO + SAM-style 多次点击细化

仓库里的这组实验不是把通用 SAM2 直接当作最终分割器，而是采用更适合 FA 渗漏的两阶段方案：

1. DINOv3 ConvNeXt-Tiny 先给出两通道病灶粗分割；
2. 根据粗分割的漏检/误检生成 SAM 风格的正、负点提示，并把累计点击交给一个残差细化器。

这一区分很重要：项目已有的直接 SAM2 微调结果（见 `EXPERIMENT_LOG.md`）受到 FA 域差异影响，低于 DINO 基线；交互式方案只使用 SAM 的 prompt 思路，最终语义分割由 DINO/refiner 完成。

## 从 `split_dataorigin.zip` 开始

压缩包内已经包含 `dataset/dataset/split_dataorigin/...` 目录结构。在项目根目录解压即可；`dataset/` 只作为只读输入：

```bash
cd /root/autodl-tmp/Uveitis-Segmentation-bs
unzip -q /root/autodl-tmp/split_dataorigin.zip -d .
```

把 DINOv3 ConvNeXt-Tiny 预训练权重放到配置指定的位置（权重不会提交到 git）：

```bash
mkdir -p weights
cp /root/autodl-tmp/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
```

先确认数据配对和环境：

```bash
python scripts/check_environment.py
python scripts/index_dataset.py --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml
```

## 复现实验

下面的命令按折训练 DINO 粗分割。每一折的验证集只使用该折，另外四折用于训练；这和 `cache_dino_predictions.py` 的缓存逻辑一致。

```bash
for fold in f1 f2 f3 f4 f5; do
  python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
    --run-name vsubr_vw05_${fold} --fold ${fold}
done
```

缓存 DINO logits、图像和标签，供细化器重复使用。默认 `uint8` 图像缓存会在加载时恢复 ImageNet 归一化，并且每个样本都强制复制为独立连续存储，避免把整个 batch backing storage 重复写入：

```bash
python scripts/cache_dino_predictions.py \
  --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
  --checkpoint-template 'runs/vsubr_vw05_{fold}/{fold}/checkpoints/best.pt' \
  --output-root outputs/dino_refiner_cache/vsubr_vw05_compact \
  --folds f1,f2,f3,f4,f5 --splits train,val --image-storage uint8
```

按仓库原始 MVP 口径训练交互细化器（默认 `0/1/3/5` 次点击）：

```bash
python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner.yaml
```

原始配置使用 `strategy: random`、`cumulative: false`，用于和 `EXPERIMENT_LOG.md` 中已经记录的 MVP 数字对齐。严格的 SAM 风格闭环配置使用 `configs/dino_sam_refiner_iterative.yaml`：它用 `strategy: farthest` 选择空间分散的误差点，并保留历史点击；每一轮先运行 refiner，再从最新残差选择下一点：

```bash
python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner_iterative.yaml
```

已有 checkpoint 可以单独评估：

```bash
python scripts/evaluate_interactive_refiner.py \
  --config configs/dino_sam_refiner.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_mvp/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,1,3,5 --output outputs/interactive_eval/f1.json
```

## 训练/部署边界

离线 Dice 曲线为了客观比较，使用标签生成 oracle 误差点；这部分只用于评估，不能在真实部署时使用。部署时调用 `bs.interactive_refiner.refine_with_clicks`，将医生实际点击的 `[B,C,K,2]` 坐标传入即可。该函数不读取标签，并返回每轮 logits 历史，适合导出 0/1/3/5-click 可视化。

`train_interactive_refiner.py` 会把输出写入 `outputs/interactive_refiner_runs/`，包括每折的 `metrics.csv`、`train.log` 和 `checkpoints/{best,latest}.pt`。缓存和权重也都位于 `outputs/`/`weights/`，不会改写数据集。
