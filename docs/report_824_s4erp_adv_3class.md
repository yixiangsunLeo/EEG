---
title: "S4ERP-Adv 对抗训练与 3-class 跨被试实验报告"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-08-24"
updated: "2026-08-24"
status: "active"
---

# S4ERP-Adv 对抗训练与 3-class 跨被试实验报告

> [!WARNING]
> **本报告有三处已确认的错误（2026-08-25 核对），数字请以
> [`docs/RESULTS_TRUTH.md`](RESULTS_TRUTH.md) 为准。**
>
> 1. §4.1 / §5.2 把 `paper_baseline_3c_821_v1` 称为 "2-class baseline"。**它是 3 分类**，
>    本仓库不存在任何 2 分类实验。两批实验的真实差别是**训练配方**（AdamW/64/100ep/patience25
>    对 Adam 1e-4/128/500ep/patience100）。因此 §0 结论 1 和 §5.2 的整个论证**不成立**。
> 2. §5.1 的 DualCD-S4-Flatten `67.07 ± 8.40%` 是**前 6 折**的均值，脚注却写 7/11 折。
>    fold 6 完成后 7 折均值是 **69.87 ± 11.20%**。
> 3. 全表标准差用的是总体标准差（÷n），与仓库其他地方口径不一致。
>
> 另：§6 和 §8 的待办"需要换 `val_mode=sessions` 复跑确认"**已于 2026-08-07~10 完成**
> （`foundation_3c_loso_paper_v1`），结论是 val_mode 只值约 0.5pp，不是 baseline 高于论文的原因。
>
> 仍然成立的部分：S4ERP-Adv 相对 S4ERP 的 **+2.92pp** 是同 run 同折的干净比较；
> λ 扫描无差异；ATCNet 在我们的配方下最强。

> 一句话定位：为 S4ERP 增加 GRL 对抗训练（被试不变特征学习），并在 3-class LOSO 协议下与全部 baseline 对比，对标 DSGNet 论文（IEEE TNSRE 2026）。

## 0. 核心结论 (honest)

1. **对抗训练在 3-class 下有效**：S4ERP-Adv (67.60%) > S4ERP (64.68%)，**+2.92%**。2-class 下无差异，3-class 下提升明显。
2. **ATCNet 是当前最强 baseline**：70.85%，高于论文报告的 68.34%（+2.5%）。
3. **我们复现的所有 baseline 均高于 DSGNet 论文**（EEGNeX +3.9%, EEGNet +2.3%），可能原因是验证协议不同（见 §6）。
4. **lambda_adv 扫描无差异**：λ=0.1/0.5/1.0/2.0 全部 67.2%~67.8%，对抗强度不是关键因素。
5. **过拟合仍是主要问题**：所有模型 train-val gap 巨大，S4ERP-Adv 的 CE 在 epoch 30 降到 0.006 而 val 停滞。

## 1. 实验背景

### 1.1 P10 项目主线

跨 session MI-EEG 解码泛化研究。核心问题：同一被试不同天 EEG 分布漂移导致性能下降 ~10pp（80.7% → 71.1%）。

研究路线：诊断 → 基线 → 对齐验证 → prototype drift 分析 → adaptation → memory → agent。

### 1.2 本阶段定位

跨被试（cross-subject）泛化是 P10 的平行主线。目标是：
- 建立可复现的跨被试 baseline（对标 DSGNet 论文）
- 验证对抗训练（Domain Adversarial Training）能否通过学习**被试不变特征**提升泛化
- 为后续跨 session 场景的对抗适配（domain=session）积累经验

### 1.3 参考项目

- **FD-ARL**（ICASSP 2026, 同实验室）：Feature Disentanglement with Adversarial-Reconstruction Learning，GRL + 多分类 domain discriminator + 特征解耦
- **DSGNet**（IEEE TNSRE 2026, 用户提供）：Dual-branch Subject-aligned Generalization Network，Class Alignment Loss（类内紧凑 + 类间可分 + 跨被试中心对齐）

## 2. 数据集与协议

### 2.1 数据

| 项目 | 值 |
|:---|:---|
| 数据集 | WBCIC-SHU 2025（3-class 子集） |
| Manifest | `outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv` |
| 被试 | 11（sub-001 ~ sub-011） |
| Session | 33（每人 3 session，间隔多日） |
| 总 trials | 9899 |
| 任务 | 3-class：左手 / 右手 / 双脚 |
| 通道 | 58ch (10-20) |
| 采样 | 250 Hz（从 1000 Hz 降采样），4s epoch |
| 归一化 | per_sample_zscore（fit-free，无泄露） |

### 2.2 协议

| 项目 | 值 |
|:---|:---|
| 协议 | LOSO（Leave-One-Subject-Out） |
| Folds | 11（每个被试做一次测试） |
| 每 fold | 8 训练 / 2 验证 / 1 测试 被试 |
| val_mode | subjects（留出不同被试验证，严格无泄露） |
| 模型选择 | macro_f1, patience=25, early stop |
| 优化器 | AdamW, lr=1e-3 (s4erp 族) / 1e-4 (dualcd), wd=1e-4 |
| 调度 | CosineAnnealingLR(T_max=100) |
| 梯度裁剪 | max_norm=4.0 |
| seed | 0 |

## 3. 模型设计：S4ERP-Adv

### 3.1 动机

S4ERP 在 2-class 实验中过拟合最严重（train-val gap +32.7%）。假设：encoder 学到了被试特异性特征（而非通用的 MI 表征），导致跨被试泛化差。对抗训练强制 encoder 输出被试不变的特征。

### 3.2 架构

```
输入: [B, 58, 1000]
  │
  ├─ ShallowNetEmbedding(58→128)  ┐
  ├─ S4EncoderCore(128, 4层, D=8) ├─ 与 S4ERP 完全相同的 backbone
  └─ z [B, 487, 128]              ┘
        │
        ├─ flatten → features [B, 62336]
        │     └─ Classifier: GELU→Dropout→Linear(62336→3) → logits [B,3]
        │
        └─ mean_pool(dim=1) → [B, 128]    ← 池化降维避免判别器过大
              └─ GRL(λ) → MLP(128→128→11) → subject_logits [B,11]
```

### 3.3 关键设计

| 组件 | 设计 | 理由 |
|:---|:---|:---|
| GRL | 前向恒等，反向取反 × λ | 标准 DANN（Ganin 2016） |
| 判别器 | 11 类（每被试一类）多分类 | 比二分类（源/目标）更强约束 |
| 池化 | 对 patch 维 mean-pool（62336→128） | 判别器参数从 16M 降到 17K，梯度稳定 |
| λ warmup | 前 5 epoch λ=0，线性升至 λ_max | 先学任务，再学不变性 |
| 损失 | CE + CE(subject)（GRL 后） | 单优化器联合训练 |

### 3.4 参数量

| 组件 | 参数 |
|:---|---:|
| Backbone | 1,246,208 |
| Task head | 189,000 |
| Adv head | 17,000 |
| **Total** | **1,451,150** |

## 4. 实验链

### 4.1 实验 1：2-class baseline（报告 821，8/14）

- 7 模型 × 11 被试 LOSO 2-class
- 结果：EEGNeX 69.91% 最优；S4ERP 68.26%；过拟合普遍
- 结论：跨被试天花板 ~70%，被试差异 ≫ 模型差异

### 4.2 实验 2：S4ERP-Adv 开发（8/22）

- 新增 `S4ERPAdversarial` 模型 + GRL + subject discriminator
- 踩坑记录：
  - SLURM conda 环境错误 → 改用 `shu_gpu.sbatch` 模板（mi_torch_cu118）
  - CUDA device-side assert → 根因：3-class label {0,1,2} 传给 2-class 模型
  - `uses_custom_loss=False` 导致对抗路径未触发 → 修复为 variant 名判断

### 4.3 实验 3：lambda_adv 扫描（8/22-23）

- 4 配置并行：λ=0.1/0.5/1.0/2.0 × warmup=5
- 结果：67.79 / 67.37 / 67.60 / 67.23%，无显著差异
- 结论：对抗强度在 0.1~2.0 范围内不敏感

### 4.4 实验 4：3-class 全模型批量对比（8/23-24，当前）

- 动机：DSGNet 论文用 3-class LOSO，2-class 结果不可直接对比
- 预实验：8 模型 × 1 fold × 3 epochs → `dualcd_transformer` CUDA OOM（68M 参数）→ 移除
- 正式：7 模型 × 11 folds（4 GPU 并行）
- 状态：6/7 完成，`dualcd_s4_flatten` 进行中（7/11 folds）

## 5. 结果

### 5.1 3-class LOSO 结果（6/7 模型）

| 模型 | Accuracy | Macro-F1 | AUC | 参数量 | 论文报告 |
|:---|:---:|:---:|:---:|---:|:---:|
| **ATCNet** | **70.85 ± 14.50%** | 70.37% | 86.12% | 115K | 68.34% |
| EEGNeX | 68.78 ± 13.08% | 68.55% | 85.00% | 59K | 64.88% |
| EEG-Deformer | 67.81 ± 13.32% | 67.18% | 84.10% | 1.6M | -- |
| **S4ERP-Adv** | **67.60 ± 13.56%** | 67.45% | 83.80% | 1.45M | -- |
| EEGNet | 67.22 ± 12.19% | 66.99% | 83.30% | 3.5K | 64.92% |
| DualCD-S4-Flatten* | 67.07 ± 8.40% | 67.03% | 83.99% | 67M | -- |
| S4ERP | 64.68 ± 12.93% | 64.32% | 81.84% | 1.4M | -- |

*dualcd_s4_flatten 仅 7/11 folds，最终结果可能微调

### 5.2 对抗训练效果

| 对比 | 2-class (报告821) | 3-class (当前) |
|:---|:---:|:---:|
| S4ERP | 68.26% | 64.68% |
| S4ERP-Adv | 67.79% (λ=0.1) | **67.60%** |
| 差值 | -0.47% | **+2.92%** |

3-class 下对抗训练带来显著提升。可能的解释：3-class 任务（含双脚类）更难，被试特异性特征干扰更大，对抗正则化的价值更明显。

### 5.3 过拟合分析

- S4ERP-Adv epoch 10 达到 best macro_f1=69.4%，CE 已降到 0.111
- epoch 30+ 时 CE≈0.006，val 不再提升（65-68% 波动）
- 说明对抗正则化不足以完全抑制过拟合，但仍比无对抗的 S4ERP 泛化更好

## 6. 与 DSGNet 论文对比

### 6.1 协议差异

| 项目 | DSGNet 论文 | 我们 |
|:---|:---|:---|
| 分类数 | 3-class | 3-class |
| 被试 | 11 | 11 |
| 协议 | LOSO | LOSO |
| **验证方式** | **val_mode=sessions**（train ses1+2, val ses3，同被试） | **val_mode=subjects**（留出不同被试） |
| 预处理 | 0.5-40Hz bandpass | per_sample_zscore |

### 6.2 结果对比

| 模型 | 论文 | 我们 | 差异 |
|:---|:---:|:---:|:---:|
| ATCNet | 68.34% | 70.85% | +2.51% |
| EEGNeX | 64.88% | 68.78% | +3.90% |
| EEGNet | 64.92% | 67.22% | +2.30% |
| DSGNet (proposed) | 68.56% | -- | -- |

**所有 baseline 均高于论文**。最可能原因：验证协议不同——`val_mode=subjects` 的验证被试来自训练集外（模型选择更保守），而论文的 `val_mode=sessions` 允许训练/验证共享被试（模型选择更宽松）。**需要换 val_mode=sessions 复跑确认**。

## 7. 代码与结果位置

### 7.1 服务器路径

```
/share/home/yuan/SYX/eeg-mi-online/
├── code/
│   ├── models/eeg_foundation/
│   │   ├── losses.py              # GRL（GradientReversalLayer）
│   │   ├── models.py              # S4ERPAdversarial 类
│   │   └── adapter.py             # s4erp_adv 注册 + training_step
│   ├── training/e2e_trainer.py    # batch 支持 (X, y, subject_id)
│   ├── experiments/cross_subject_protocols.py  # LOSO + 对抗数据加载
│   └── configs/
│       ├── models/s4erp_adv.yaml
│       └── experiments/
│           ├── foundation_cross_subject_3class.yaml      # 正式 3-class
│           └── foundation_cross_subject_adv_{l01,l05,l10,l20}_w5.yaml  # λ 扫描
├── outputs/experiments/wbci_shu/
│   ├── foundation_cross_subject_3class_v1/cells/         # 3-class 结果
│   └── foundation_cross_subject_adv_*_w5/cells/          # λ 扫描结果
└── logs/slurm/                                           # 作业日志
```

### 7.2 Git commits

| commit | 内容 |
|:---|:---|
| `2a09cf3` | feat: add S4ERP-Adv adversarial domain adaptation |
| `ae95cd1` | fix: use wbci_3c manifest (11 subjects) |
| `c0cc95e` | fix: uses_custom_loss=True for s4erp_adv |
| `1e0eb9c` | fix: n_classes=3 for 3-class MI |
| `bf49e24` | feat: lambda warmup schedule + lambda_adv=1.0 |

### 7.3 本地路径

- 报告 821 表格：`0_docs/report_821_cross_subject_20260814/tables/`
- DSGNet 论文解读：`02_Literature/2026_Subject_Independent_DL_Framework_MI_Decoding.md`

## 8. 下一步建议

1. **【优先】换 val_mode=sessions 复跑**（与论文协议对齐），确认 baseline 差异来源。仅需改 config 一行 + 重跑 7 模型
2. **换 seed 验证稳定性**：seeds [0] → [0,1,2]，排除单 seed 运气
3. **对抗训练继续调优**：λ 扫描无差异说明瓶颈不在对抗强度，可在以下方向尝试：
   - 判别器容量增加（128→256 隐藏层）
   - 特征解耦（FD-ARL 风格：z_task / z_subject 分离 + 重建损失）
   - 跨 session 对抗（domain=session 而非 subject，回归 P10 主线）
4. **dualcd_transformer 修复**：减 batch_size（64→32）+ 梯度累积，或跳过（67M 参数性价比差）
5. **过拟合治理**：dropout 0.1→0.3、weight decay 增大，或早停更激进

## 9. 文件清单

| 文件 | 内容 |
|:---|:---|
| `outputs/experiments/wbci_shu/foundation_cross_subject_3class_v1/cells/*/result.json` | 每 fold 完整指标 |
| `outputs/experiments/wbci_shu/foundation_cross_subject_adv_*_w5/cells/*/result.json` | λ 扫描结果 |
| `logs/slurm/3c_*-*.out` | 训练日志 |
| `code/configs/experiments/foundation_cross_subject_3class.yaml` | 正式实验配置 |
