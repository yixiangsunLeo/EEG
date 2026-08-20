---
title: "交接文档 — paper_baseline_3c_821_v1（8:2:1 跨被试统一对比 run）"
tags:
  - "#modality/eeg"
created: "2026-08-12"
updated: "2026-08-12"
status: "active"
---

# 交接文档 — `paper_baseline_3c_821_v1`

> 面向「同一集群、换到另一个账号继续跑」。只讲这一个 run，其他阶段（Phase 0–2c / Phase 3 TTA）
> 与本文无关，照旧看 `AGENTS.md`。
>
> **对应 Git commit**：`ad0c5d2`（已 push 到 `origin/main`）。新账号 `git pull` 到这个提交即代码一致。
> **旧账号项目根**（拷贝源）：`/share/home/Zihang/MI`
> **当前状态**：**交接已完成，已在新账号 `yuan` 恢复训练**（2026-08-13 00:48，job `38023–38030`），
> 进度 **50/77 cell**。交接的实际执行记录见 §4.5。

---

## 1. 这个实验在跑什么

一句话：**在 WBCIC-SHU 三分类（11 被试）上，用同一套 8:2:1 跨被试划分 + 同一套论文 recipe，把 4 个已发表 baseline 和我们 3 个模型放在同一个 run 里跑完，得到一张可直接对比的表，并输出 train/val/test 三条曲线。**

| 维度 | 设定 |
|:---|:---|
| 数据 | WBCIC-SHU 3C，11 被试 × 3 session，`n_channels=58`、`n_times=1000`、`n_classes=3`、`sfreq=250` |
| 泛化轴 | **跨被试**（subject-independent），LOSO 11 折 |
| 划分 | **8:2:1（按被试）**：每折留 1 被试测试；其余 10 人里 2 人当验证、8 人当训练，每人 3 session 全用。trial 数正好 **7199 : 1800 : 900** |
| recipe | DSGNet 论文 §IV-A：Adam、lr `1e-4`、batch `128`、max `500` epoch；无 scheduler / 无 weight decay / 无 grad clip |
| 早停 | `patience=100`（**我们的偏离**，论文是固定 500 epoch；学长要求加早停） |
| 模型选择 | 只看**验证集** `macro_f1` 存 `best.pt` |
| 归一化 | `per_sample_zscore`（per-trial、fit-free，统计量不跨划分） |
| 三曲线 | 每 epoch 记 train / val / test 三条；**test 只做可视化，绝不参与 best.pt 或早停**（守卫在 `code/training/e2e_trainer.py`） |
| seed | 只 `[0]` |

**7 个模型**（`models:` 顺序）：

| 模型 | 来源 | 实测参数量 | 论文 Table II SHUv5 Acc |
|:---|:---|---:|---:|
| `eegnet_official` | [18] `vlawhern/arl-eegmodels`（Keras→Torch 1:1） | 3,523 | 0.6492 |
| `eegnex` | [20] `chenxiachan/EEGNeX`（Keras→Torch 1:1） | 59,275 | 0.6488 |
| `eeg_deformer` | [23] `yi-ding-cs/EEG-Deformer`（官方 PyTorch 原样） | 1,612,307 | 0.6529 |
| `atcnet` | [24] `Altaheri/EEG-ATCNet`（Keras→Torch 1:1） | 114,719 | 0.6834（论文里最强 baseline） |
| `dualcd_s4_flatten` | 我们（学长包） | 66,923,523 | — |
| `s4erp` | 我们（学长包） | 1,433,219 | — |
| `dualcd_transformer` | 我们（学长包） | 67,962,371 | — |

**baseline 硬规矩**：只认原作者官方仓库。没有完整官方代码的一律排除并写明理由 —— EEG-Inception [27]、MDGEEG [35]、EEG-DG [38]、DSGNet 本身（只有预览）都被排除；DSGNet 的 Acc 0.6856 / F1 0.6833 只作为**论文引用数字**，不是我们跑出来的。理由记在 `code/models/paper_baselines/README.md`。

**⚠ 这个 run 不是 DSGNet 论文划分。** 论文是「同被试的 session 3 当验证」，那一版在 `foundation_cross_subject_wbci_3c.yaml`（`run_id=foundation_3c_loso_paper_v1`，已跑完）。本 run 是学长 2026-08-09 要求的 8:2:1 按被试划分。两者都测留出被试的 3 个 session，**不要混着报**。

---

## 2. 当前进度（截至 2026-08-12 23:50）

**50 / 77 cell 完成**（77 = 7 模型 × 11 折）。

| 模型 | 完成 | 缺的 fold |
|:---|:---|:---|
| `eegnet_official` | 11/11 | — |
| `eegnex` | 11/11 | — |
| `eeg_deformer` | 11/11 | — |
| `atcnet` | 8/11 | 5, 8, 10（有 `last.pt`，可断点续） |
| `dualcd_s4_flatten` | 3/11 | 3,4,5,6,7,8,9,10 |
| `s4erp` | 3/11 | 3,4,5,6,7,8,9,10 |
| `dualcd_transformer` | 3/11 | 3,4,5,6,7,8,9,10 |

### 阶段性数字（**只含已完成折，不能当最终结果**）

```
eegnet_official      Acc 0.6636±0.1245  F1 0.6606  n=11
eegnex               Acc 0.6991±0.1293  F1 0.6968  n=11
eeg_deformer         Acc 0.6849±0.1414  F1 0.6825  n=11
atcnet               Acc 0.7273±0.0961  F1 0.7218  n=8   ← 折数不同，勿直接比
dualcd_s4_flatten    Acc 0.7137±0.0947  F1 0.7130  n=3
s4erp                Acc 0.7541±0.0901  F1 0.7538  n=3
dualcd_transformer   Acc 0.7156±0.0957  F1 0.7155  n=3
```

三个大模型只有 fold 0–2，而被试难度差异极大（sub-011 两条 arm 都只有 ~0.41）。**n 不同的模型之间不可比**，必须等 11/11 齐了再下任何结论。

### 为什么停了

原先 4 卡按 fold 分组（3+3+3+2、每组 7 模型）提交 `36928–36931`：只有 `36928` COMPLETED，`36929/30/31` 全部 **TIMEOUT@48h**。想加长墙时到 96h/72h/60h **全被 `QOSMaxWallDurationPerJobLimit` 拒**——这个集群 QOS 上限就是 **48h**。于是改成「每个未完成 fold 一个 job + `--models` 只跑缺的模型」，重新提交了 `37966–37973`；随后用户决定换账号，已全部 `scancel`。

单折耗时实测（用于排 job）：

| 模型 | h/fold（均值） | 最长 |
|:---|---:|---:|
| `dualcd_transformer` | 6.98 | 7.13 |
| `dualcd_s4_flatten` | 4.84 | 5.09 |
| `atcnet` | 2.06 | 2.90 |
| `s4erp` | 0.74 | 0.91 |
| `eeg_deformer` | 0.55 | 0.76 |
| `eegnex` | 0.46 | 0.82 |
| `eegnet_official` | 0.22 | 0.29 |

剩余总量 ≈ **107 GPU·小时**。按「一折一 job」，每个 job ≈ 12.6h（flatten+s4erp+transformer），远在 48h 内；若能 4 卡并发，约 26h 墙钟跑完。

---

## 3. 关键文件在哪（新账号 pull 后即有）

| 内容 | 路径 |
|:---|:---|
| **本实验全部参数** | `code/configs/experiments/paper_baseline_3c_821.yaml` |
| 模型结构默认超参 | `code/configs/models/{eegnet_official,eegnex,eeg_deformer,atcnet,dualcd_s4_flatten,s4erp,dualcd_transformer}.yaml` |
| 本 run 实际用的模型超参 | 同一实验 YAML 的 `model_params:` 段（**优先于** models/*.yaml） |
| 已发表 baseline 代码 + 出处 | `code/models/paper_baselines/`（含 README 说明每个模型的官方仓库与移植方式） |
| ATCNet | `code/models/atcnet/`（`_official_keras/` 是 vendored 官方 Keras，`atcnet_torch.py` 是 1:1 移植） |
| 我们的模型 | `code/models/eeg_foundation/` |
| 模型注册表 | `code/models/registry.py`（按名字构建） |
| 训练器（含三曲线 + 续跑） | `code/training/e2e_trainer.py` |
| 跨被试协议（划分/评测/跳过逻辑） | `code/experiments/cross_subject_protocols.py` |
| runner 接线 | `code/runners.py` → `run_foundation_cross_subject` |
| Slurm 提交/续跑脚本 | `scripts/slurm/submit_paper_baseline_821.sh` |
| 汇总脚本 | `scripts/summarize_cross_subject.py` |
| 三曲线绘图 | `scripts/plot_three_curves.py` |
| 论文对标锚点 | `4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md` |

**不在 Git 里的**（必须拷或重建）：`outputs/`、`checkpoints/`、`logs/`、`*.local.yaml`、预处理数据。

---

## 4. 怎么把已有成果拷到新账号

新账号项目根记作 `$NEW`（例：`/share/home/<新号>/MI`）。旧账号根 = `/share/home/Zihang/MI`。

### 4.0 ⚠ 先解决「新账号读不到旧账号目录」

旧账号的**家目录**原本是 `drwx------`（700，仅本人），而它下面的 `MI/` 及各级子目录**本来就是**
`drwxrwxr-x`。也就是说别的账号是被家目录这一层挡住的，不是被项目目录挡住的。

**✅ 2026-08-13 曾短暂 `chmod o+x`（`drwx-----x`）方便新账号 rsync；同日已 `chmod o-x` 收回，现为 `drwx------`。**
若还需再搬文件，在旧账号临时重新开放：

```bash
chmod o+x /share/home/Zihang     # 临时开放穿过
# …新账号 rsync 完…
chmod o-x /share/home/Zihang     # 立刻收回
ls -ld /share/home/Zihang        # 应是 drwx------
```

不想动家目录权限的话，备选路线：`/share/workspace2` 是 `drwxrwxrwt`（全局可写 + sticky，
约 8.4 T 空余），可以在那里开一个临时中转目录放打包文件，拷完删掉。

**不要为此往 GitHub 传数据**：单个 npz 最大 63 MB，超 50 MiB Git 就会警告，2 G 进仓库是永久
膨胀（要清得重写历史）；而且完全没必要 —— 数据可按 §4.3.1 在新账号重跑生成，真正需要搬的只有
~20 M 的 run 状态。

### 4.1 先对齐代码

```bash
cd "$NEW" && git pull        # 应到 ad0c5d2 或更新
git log --oneline -1
```

### 4.2 本机路径配置（**不进 Git**）

```bash
cd "$NEW"
ls code/configs/paths.local.yaml || cp code/configs/paths.example.yaml code/configs/paths.local.yaml
```

本 run 只关心两个键：

```yaml
raw_data:
  shu_2c_root: "/share/workspace2/moto_imagination/WBCIC_SHU"
manifests:
  wbci_3c_processed_manifest: "outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv"
```

- `shu_2c_root` 指向共享盘上的官方数据（`/share/workspace2/moto_imagination/WBCIC_SHU`，全局可读、
  只读使用）。3C 源目录由脚本自动拼成 `<root>/derivatives/3C dataset_processeddata`，**不依赖旧账号**。
- `wbci_3c_processed_manifest` 是**项目内相对路径**，所以数据放在 `$NEW/outputs/processed/wbci_shu_3c_mat_clean/`
  就不用改配置。

旧账号的 `paths.local.yaml` 直接 `cp` 过去也可以（其余键都指向 `/share/workspace2/...` 共享盘）。

### 4.3 三份要拷的东西

| # | 内容 | 大小 | 必要性 |
|:--|:---|---:|:---|
| A | 3C 预处理数据 `outputs/processed/wbci_shu_3c_mat_clean/` | 2.0 G | **必须**，但**建议新账号自己重跑生成，而不是拷**，见 §4.3.1 |
| B | run 输出 `outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/` | 14 M | **必须**，这是「哪些 cell 已完成」的唯一凭据，也存着已完成折的指标 + 三曲线 history |
| C | 权重 `checkpoints/wbci_shu/paper_baseline_3c_821_v1/` | 8.8 G | **可选**，见下 |

#### 4.3.1 ⚠ A 走「重跑生成」，不要拷贝（2026-08-13 已定）

`processed_manifest.csv` 的 `npz_path` 列是**绝对路径**，而 `code/datasets/session_splits.py`
直接拿这一列喂 `load_session_npz()`，**不会**按当前项目根重新拼路径。所以把 manifest 原样拷到
新账号，它会去读 `/share/home/Zihang/MI/outputs/...` —— 要么报文件不存在，要么悄悄依赖旧账号的盘。
（同类问题 `AGENTS.md` 里已记过一次：WBCIC `embed_index__*.csv` 的失效 `npz_path`。）

官方 3C 源数据在共享盘上是**全局可读**的（`/share/workspace2/moto_imagination/WBCIC_SHU/derivatives`
= `drwxrwxr-x`），预处理只是「读官方 `.mat` → 标签归一化 → 转存 `.npz`」，确定性、无随机性。
所以新账号直接自己生成，manifest 里写的就是它自己的正确路径：

```bash
cd "$NEW"
python scripts/preprocess_wbci_3c.py --dry-run   # 先看计划
python scripts/preprocess_wbci_3c.py             # 生成 33 session npz + manifest
# 自检：应 33 行 ok，且 npz_path 指向 $NEW
wc -l outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
rg -c "$NEW" outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
```

**若坚持拷贝 A**，拷完必须重写路径列，否则会读到旧账号：

```bash
sed -i "s#/share/home/Zihang/MI#$NEW#g" \
  "$NEW/outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv"
```

#### 4.3.2 rsync 命令

```bash
NEW=/share/home/<新号>/MI          # ← 改成真实路径
OLD=/share/home/Zihang/MI

mkdir -p "$NEW/outputs/processed" \
         "$NEW/outputs/experiments/wbci_shu" \
         "$NEW/checkpoints/wbci_shu"

# A. 数据（2.0G）— 已定为在新账号重跑生成（§4.3.1），不走这条。
#    万一非要拷，拷完必须 sed 改 manifest 的 npz_path：
# rsync -ah --info=progress2 \
#   "$OLD/outputs/processed/wbci_shu_3c_mat_clean/" \
#   "$NEW/outputs/processed/wbci_shu_3c_mat_clean/"

# B. run 输出（14M，必须）
rsync -ah --info=progress2 \
  "$OLD/outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/"

# C-精简（推荐，6M）：只拷可续跑的 ATCNet 断点，省 8.8G
rsync -ah --info=progress2 \
  --include='atcnet__fold5__seed0/***' \
  --include='atcnet__fold8__seed0/***' \
  --include='atcnet__fold10__seed0/***' \
  --include='*/' --exclude='*' \
  "$OLD/checkpoints/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/checkpoints/wbci_shu/paper_baseline_3c_821_v1/"
```

**为什么 C 可以精简**：三曲线 `history` 和所有指标都写在 `outputs/.../cells/*/result.json` 里（`scripts/summarize_cross_subject.py` 和 `scripts/plot_three_curves.py` **只读 `cells/`，不读权重**）。所以已完成 cell 的 `best.pt` / `last.pt` 对「跑完 + 出表 + 出图」毫无必要，8.8G 里绝大部分是两个 67M 模型 fold0–2 的 541 MB 级权重。

**什么时候要 C-完整**：以后要拿训练好的权重做别的事（可视化特征、接 TTA、重新评测、放论文补充材料）。那就整目录 rsync：

```bash
rsync -ah --info=progress2 \
  "$OLD/checkpoints/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/checkpoints/wbci_shu/paper_baseline_3c_821_v1/"
```

**可以不拷的**：`dualcd_s4_flatten` fold 3/4/6 的 `last.pt`（各 541 MB）。那是被 `scancel` 打断前只训了约 25 分钟的残局，拷 1.6 G 只省 25 分钟，不如让它从头训 —— 不拷不影响正确性。

### 4.4 拷完自检

```bash
cd "$NEW"
# 应为 50
find outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells -name result.json | wc -l
# 应存在
ls outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
# manifest 的 npz_path 必须指向 $NEW；这里应该没有任何输出
rg -c '/share/home/Zihang/MI' outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
# 不该报错，且计划里已完成的 cell 会被识别
python code/run.py --dry-run --config code/configs/experiments/paper_baseline_3c_821.yaml
```

### 4.5 实际执行记录（2026-08-13，新账号 `yuan`）

新账号项目根是 **`/share/home/yuan/SYX/eeg-mi-online`**（不是 `~/MI`）。代码已在 `26bb950`，
比 `ad0c5d2` 新一个提交（就是本交接文档），无需 pull。

做了什么：

| 步骤 | 结果 |
|:---|:---|
| `paths.local.yaml` | 新建，`shu_2c_root` → `/share/workspace2/moto_imagination/WBCIC_SHU`（yuan 自己的共享盘目录） |
| A 预处理数据 | 按 §4.3.1 **本地重跑生成**，33/33 session `status=ok`，9899 trial，2.0 G |
| B run 输出 | rsync 过来，50 个 `result.json`，14 M |
| C-精简 权重 | rsync 过来，`atcnet__fold{5,8,10}__seed0` 的 `best.pt`+`last.pt`，6.5 M |
| 提交 | `bash scripts/slurm/submit_paper_baseline_821.sh` → job `38023–38030`（fold 3–10） |

`FOLD_MODELS` 表**不用改**：重新数过，缺的 cell 与脚本里写死的完全一致（27 个）。

**验证脚本 `scripts/verify_821_handoff.py`**（本次新增）。换账号 / 换机后跑它，比 `--dry-run` 强得多，
四件事：

1. **数据是否和来源账号逐位一致** —— 比对解码后 `X`/`y` 数组的 hash，而不是 `.npz` 容器字节
   （不同 numpy/zlib 版本压出来的字节可以不同，数组却一样）。**33/33 session MATCH**。
2. **重建的折是否和已完成 cell 里记录的折一致** —— 若被试名单漂移，「fold 3」就换了人，
   而 `result.json` 还在，会被静默跳过、把两种划分混进一张表。**50/50 一致**。
3. **每折是否仍是 8/2/1 被试、互不重叠、9899 trial 全用上**。
4. **本地重算的 `cell_signature` 是否等于旧账号写进 `result.json` / `last.pt` 的那个** ——
   这条最硬：同一个指纹也是续跑守卫用的，对上就说明数据维度、划分、`model_params`、
   optimizer/lr/batch/归一化/梯度累积全都一致。**50/50 cell + 3 个 ATCNet 断点全 MATCH**。
   不查这条的话，不一致要等 job 跑到那个 cell 才炸（几小时后）。

```bash
python scripts/verify_821_handoff.py                  # 抽查 3 个 session，四项应全 PASS
python scripts/verify_821_handoff.py --hash-sessions 33   # 全量比数据（~3 min）
```

> ⚠ **trial 数不是每折都 7199:1800:900。** `sub-010/ses-01` 官方只有 **299** trial（不是 300，
> 见 `code/preprocessing/wbci_3c_mat.py` 的 `min_trials`）。所以 sub-010 当验证的 fold 8 是
> `7200:1799:900`，当测试的 fold 9 是 `7200:1800:899`。**这是数据本身的性质，不是 bug**，
> 旧账号也是这个数（数据逐位一致）。8:2:1 是**被试**比例，不是 trial 数的硬等式。

**交接正确性的最强证据**：fold 5 的 ATCNet 从拷来的 `last.pt` **续在 epoch 225** 继续训练。
`run_signature` 只要有任何差异（数据维度、划分、model_params、optimizer/lr/batch/归一化…）
`e2e_trainer.py` 就会直接抛 `RuntimeError`。它没抛 —— 说明本地重跑生成的数据 + 当前 YAML
和旧账号算出来的指纹**完全相同**。

**收权限**：搬完记得在旧账号 `chmod o-x /share/home/Zihang`（§4.0）。

---

## 5. 新账号上怎么继续跑

```bash
cd "$NEW"
bash scripts/slurm/submit_paper_baseline_821.sh
```

脚本现在的形态：**fold 3–10 各一个 job**，每个 job 只跑该 fold 还缺的模型，`-t 48:00:00`。fold/模型清单写死在脚本顶部的 `FOLD_MODELS`，**若中途又完成了一部分，重提交前照 §6 重新数一遍并改这个表**（不改也行，只是会白排已完成的 cell —— 它们会被跳过，不会重训）。

续跑语义（`code/experiments/cross_subject_protocols.py`）：

- 有 `result.json` → **跳过**，数字不变
- 无 `result.json` 但有 `last.pt` → 从断点续（optimizer / scheduler / RNG / history / patience 全部复原）
- 两者都无 → 从头训

### 5.1 QOS 与排队（2026-08-13 在新账号实测）

两个账号的 QOS **完全一样**，都是 `32cpu4gpu64g`，所以「换号拿新配额」并不成立：

```
sacctmgr show qos where name=32cpu4gpu64g format=Name,MaxWall,MaxTRESPU
32cpu4gpu64g   2-00:00:00   cpu=32,gres/gpu=4,mem=64G
```

| 限制 | 是否真的生效 | 后果 |
|:---|:---|:---|
| `MaxWall` 48h | **生效** | 单 job 最长 48h，60/72/96h 一律被拒 |
| `cpu=32` | **生效** | `shu_gpu.sbatch` 每 job `-c 8` → **同时只能 4 个 job**，其余 `QOSMaxCpuPerUserLimit` PENDING |
| `gres/gpu=4` | **生效** | 同时 4 张卡 |
| `mem=64G` | **不生效** | 旧号 4 个 job 各 `--mem=40G`（共 160 G）在 `2026-08-10T22:38:40` 同时起跑。**所以不用为了并发去砍 `--mem`** |

`mem` 那一条很重要：按 64 G 除以 4 会算出「每 job 只能 16 G」，而 `pb821_g1` 实测 **MaxRSS 15.6 G**
（`dualcd_transformer` 单独跑也有 13.3 G），16 G 会贴着上限随时 OOM。既然 mem 不计入配额，
**保持 `--mem=40G` 不要动**。查历史用量：

```bash
sacct -j <jobid> --format=JobID%14,JobName%16,State%13,Elapsed%11,ReqMem%7,MaxRSS%11
```

### 5.2 4 张卡凑不齐时怎么办

提交后可能只起来 3 个，第 4 个 PENDING 且 reason 是 `(Resources)`（不是 QOS）—— 说明
`gpu2node` 两个节点 16 张卡被别人占满了。集群是**同构**的（`gpu01–gpu05` 全是 RTX 4090 D 24 G，
旧账号日志里 5 个节点跑出来的 `nvidia-smi` 型号一致），所以直接把某个 PENDING job 挪到 `gpu3node`：

```bash
scontrol update jobid=<id> partition=gpu3node     # 立刻能起
scontrol show node gpu01 gpu02 gpu03 gpu04 gpu05 | grep -E 'NodeName|AllocTRES'   # 先看哪个节点有空卡
```

**不能写多分区**：`partition=gpu2node,gpu3node` 会被拒（`Multiple partition job request not
supported when a partition is set in the association`），association 里已经把分区钉死了，只能一次填一个。

环境：conda env **`mi_torch_cu118`**（`_common.sh` 会 fallback 到 `/share/software/anaconda3/2024.10`），
`--gres=gpu:1`，`--mem=40G`。**禁止在登录节点跑训练**（预处理是纯 I/O + 转置，可以在登录节点跑）。

---

## 6. 随时查进度

```bash
cd "$NEW"
find outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells -name result.json | wc -l   # 目标 77
squeue -u "$USER" -o '%.10i %.9P %.20j %.2t %.10M %.12l %R'
tail -5 logs/slurm/pb821_*-*.out
```

逐模型看缺哪些折：

```bash
python - <<'PY'
from pathlib import Path
root = Path('outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells')
models = ['eegnet_official','eegnex','eeg_deformer','atcnet',
          'dualcd_s4_flatten','s4erp','dualcd_transformer']
for m in models:
    miss = [f for f in range(11)
            if not (root / f'{m}__fold{f}__seed0' / 'result.json').exists()]
    print(f'{m:22s} done {11-len(miss):2d}/11  missing {miss}')
PY
```

---

## 7. 77/77 之后做什么

```bash
python scripts/summarize_cross_subject.py --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
python scripts/plot_three_curves.py      --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
```

然后把最终 Acc / F1 对着论文 Table II 写进 `progress.md` + `0_docs/STATUS.md`，锚点数字更新到 `DSGNET_SHUv5_3C_ANCHOR.md`。

---

## 8. 坑（务必看完）

1. **别改 `paper_baseline_3c_821.yaml`。** 改了划分或超参又复用同一输出目录，`cell_signature` 守卫会直接报错（这是故意的，防止悄悄把两种设定的结果混进一张表）。真要改设定就换 `run_id`。
2. **`--no-resume` 会重训覆盖**，不要顺手加。
3. **test 曲线只能画图。** 任何形式的「看 test 挑 epoch / 挑模型」都作废这个 run 的可信度。
4. **两个 DualCD 用梯度累积**（`micro_batch_per_model`: flatten 32、transformer 16）保持等效 batch 128；24 G 卡直接 batch 128 会 OOM。**不要改成直接降 batch**，那会偷偷改掉 recipe。已知偏离：这两个模型的 BatchNorm 统计量按 micro-batch 算。
5. `n=3` 和 `n=11` 的模型不能比。等齐 11 折。
6. `outputs/.../runs/*.csv` 是带 `--tag-suffix` 的分片汇总，**不是**权威结果；权威只认 `cells/*/result.json`（汇总脚本也只读它）。
7. 别把 `eegnet_official`（本 run 的官方 EEGNet）和 `eegnet`（Phase 0–2c 用的项目内变体）搞混。
8. `*.local.yaml` **永远不要提交**。
9. **`processed_manifest.csv` 的 `npz_path` 是绝对路径且被直接使用**（`code/datasets/session_splits.py`
   不做重定根）。换机 / 换账号后必须确认它指向当前项目根，否则会读到别人的盘或直接报错。详见 §4.3.1。

---

## 9. 相关文档

| 文件 | 作用 |
|:---|:---|
| `AGENTS.md` | 项目唯一权威记忆 |
| `FOUNDATION_E2E_ROUTE_PLAN.md` | 端到端主线路线 |
| `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` | 跨被试协议讨论备忘 |
| `4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md` | 论文 Table II 对标锚点 |
| `code/models/paper_baselines/README.md` | 每个 baseline 的官方出处 / 移植方式 / 排除理由 |
| `SETUP.md` | 换机环境与 `*.local.yaml` 配置 |
| `progress.md` | 进度日记 |
