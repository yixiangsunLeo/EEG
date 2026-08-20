---
title: "Experiment Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-07"
status: "active"
---

# Experiment Layer

## 1. Purpose

实验协议层，组合 dataset、model、method、trainer、metrics。

## 2. What Belongs Here

| 文件 | 泛化轴 | 划分单位 |
|:---|:---|:---|
| `session_protocols.py` | within-session CV + 单源跨 session | trial（同 session 内）/ session |
| `session_multisource_protocols.py` | Phase 2a 多源跨 session | session |
| `session_alignment_protocols.py` | Phase 2b 无学习对齐 | session |
| `prototype_drift.py` / `prototype_drift_summarize.py` | Phase 2c 原型漂移诊断 | session |
| `session_tta.py` | Phase 3 TTA（paused，未废弃） | session |
| `cross_subject_protocols.py` | **跨被试（当前主线）** | **subject**（LOSO / subject k-fold / holdout）；可选 session 级 train/val |
| `cross_session_e2e_protocols.py` | **被试内跨 session（SHU 2022，2026-08-16）** | **session**（配置 train/test session 列表，每被试一个模型） |
| `metrics.py` / `data_quality.py` | 共用指标与质检 | — |

## 3. Update Rules

新增协议必须说明数据划分、防泄漏规则、输出 CSV 字段。

跨被试协议的三条铁律（`cross_subject_protocols.py` 里有断言 + `tests/foundation/` 有对应测试）：
测试被试绝不出现在 train/val；验证集来自留出训练被试（`val_mode=subjects`）、训练被试的 trial
切片（`val_mode=trials`），或**同批非测试被试按 session 切**（`val_mode=sessions`：train
ses-01+02 / val ses-03，对齐 SHUv5 / WBCIC-SHU 3C 论文）；归一化必须是 per-trial fit-free 的。

**三曲线（2026-08-10）**：`train.curves: true` 时，协议层额外构建两个**只做监控**的 loader
（训练集固定子集在 eval 模式 + 测试被试），交给 `e2e_trainer` 每 epoch 记录进 `history` 的
`train_eval` / `test`。**test 绝不参与模型选择**——`best.pt`、早停、`best_score` 只读 val，
守卫在 `code/training/e2e_trainer.py`。绘图见 `scripts/plot_three_curves.py`。

**被试内跨 session（`cross_session_e2e_protocols.py`，2026-08-16）**：SHU 2022 用。每被试一个
模型，train/test 的 session 列表来自 config（本轮 `ses-01+02+03` → `ses-04+05`），两者必须
不相交且被试必须齐备（缺 session 的被试跳过并告警，不会悄悄用更少数据训练）。验证集只能从
**训练 session** 的 trial 里分层切（`val_fraction`，本轮 1/9 = 论文 8:1:1 的 train:val）。
每个 cell 是 `(model, subject, seed)`，走同一个 `e2e_trainer`，所以断点续跑、best/last 双
checkpoint、三曲线与跨被试完全一致；测试指标**逐 session 分别算**（论文 Fig.7 的 s2–s5 口径）
再加一个池化值，都存进 `result.json`。

**cell 目录命名**（`scripts/plot_three_curves.py` 三种都能解析）：
`<model>__fold<k>__seed<s>`（跨被试单划分）、`<model>__split<ss>__fold<k>__seed<s>`
（跨被试扫多个 `split_seeds`）、`<model>__sub-<id>__seed<s>`（跨 session）。
注意 `split_seeds` 改的是**被试划分**，`train.seeds` 改的是**训练随机种子**，两者不要混。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
