# NASA5 到 Basilisk V1.0 电池 RUL 迁移学习

本仓库提供短寿命锂电池剩余寿命预测的可复现交付版本，包括冻结模型、NASA5 源域数据、Basilisk V1.0 目标域训练数据、特征契约，以及从动态预训练到目标域重训的完整程序。

## 模型

最终部署模型由 3 个随机种子成员等权集成：

```text
NASA5 多时间跨度退化动力学预训练
  -> 16维 BHUMP 因果特征
  -> TCN24 局部时序编码
  -> B_stats38 全历史统计分支
  -> NASA 退化进度跨电池参考
  -> seed 52 / 53 / 54 RUL 集成
```

严格设备级 OOF 结果：RUL MAE 为 **4.3456 EFC**，SOH MAE 为 **0.01151**。该值来自 320 台 V1.0 训练设备的严格 OOF 评估，不是 full320 训练集回放误差。

## 数据范围

| 域 | 内容 | 是否包含 |
| --- | --- | --- |
| 源域 | NASA B0005、B0018、B0033、B0043、B0044 原始 MAT、特征与标签 | 是 |
| 目标域 | Basilisk V1.0 的 320 台公开训练设备、24,425 个周期 | 是 |
| 目标域原始曲线 | 2,752,050 行 V-I-T 诊断曲线 | 是，Git LFS |
| 目标域 rich 特征 | 259 列周期级特征表 | 是，Git LFS |
| 验证与封存集 | NASA B0030/B0042、B0038/B0039；Basilisk validation、sealed、OOD | 否 |

模型输入固定为 16 维 BHUMP 特征。容量、真实 SOH、RUL、EOL、真实内阻以及未来曲线均不作为推理输入。

## 目录

```text
datasets/                    NASA5 源域与 Basilisk V1.0 训练数据
  contracts/                 冻结特征契约和泄漏审计
  source_nasa5/              五块 NASA 源电池
  target_basilisk_v10/       320 台目标训练设备
model/                       冻结三种子集成模型与推理程序
training/                    预训练、迁移微调、集成训练与冻结 OOF 配置
experiments/                 长寿命与多机理研究分支，不参与最终提交模型
DATASET_MANIFEST.json        数据范围与统计信息
SHA256SUMS.csv               文件完整性哈希
```

## 安装

```powershell
pip install -r model\requirements.txt
```

克隆包含 LFS 文件的仓库后，执行：

```powershell
git lfs pull
```

## 推理

模型可以读取周期级 16 维特征表，或原始 V-I-T 曲线长表。

```powershell
python model\predict_battery_rul.py `
  --input datasets\target_basilisk_v10\raw_curves\battery_train_curves.csv.gz `
  --output predictions.csv `
  --device cuda
```

没有 CUDA 时将 `--device cuda` 改为 `--device cpu`。详细输入字段和输出字段见 [model/README.md](model/README.md)。

## 从头重训

训练顺序为 NASA5 动态预训练、Basilisk V1.0 目标域适配、B_stats 与跨电池参考分支训练，最后导出 seed 52/53/54 集成。完整命令见 [training/README.md](training/README.md)。最终 full320 重训只使用仓库内的 320 台 V1.0 训练设备，且不读取 validation、sealed 或 OOD 数据。

## 研究分支

[experiments/](experiments/README.md) 进一步收录两个独立方向：未训练的 ALT26 -> Basilisk V1.4.1 长寿命模型定义，以及 V1.0 + V1.5 的 pooled、multi-expert、MLDG、GroupDRO、SWAD 对照实验。它们均保留真实筛选状态和负结果，不会被主模型推理程序加载。

## GitHub 上传

两个大文件已经在 `.gitattributes` 中配置为 Git LFS。创建 GitHub 空仓库后，按 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md) 的命令推送即可。

## 使用边界

NASA 数据保留其原始数据使用条款；Basilisk V1.0 是面向算法验证的灰箱仿真数据，不能等同于真实航天器电池在轨寿命数据。独立 validation 上，NASA 迁移模型未稳定超过强目标域参考控制，因此仓库将其作为可复现研究模型，而不宣称已获得对所有未知电池稳定有效的迁移优势。
