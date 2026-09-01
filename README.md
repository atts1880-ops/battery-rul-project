# 航天器电池退化建模与剩余寿命迁移预测

本项目面向航天器关键部件智能运维场景，研究在目标域标注有限、源域与目标域试验条件不同的情况下，如何根据电池历史诊断放电曲线预测健康状态（SOH）和剩余寿命（RUL）。

项目完成了从数据构建、因果特征提取、NASA 源域预训练、Basilisk 目标域迁移微调、严格设备级验证到最终模型封装的完整链路。当前正式提交模型适用于约 42-122 EFC 的短寿命仿真电池；长寿命和多机理泛化模型作为研究分支单独保存，不参与正式模型集成。

## 项目解决的问题

赛题要求不仅是拟合一个寿命数值，还需要说明退化如何产生、哪些量可以在线观测、源域知识如何迁移，以及评价过程是否存在数据泄漏。本项目围绕这些要求完成了以下工作：

| 赛题关注点 | 本项目实现 |
| --- | --- |
| 退化机理建模 | 在 Basilisk V1.0 中设置容量衰减、内阻增长、温度与负载加速以及可选 knee 加速 |
| 可观测遥测生成 | 根据诊断工况生成电压、电流、温度曲线，模型不读取仿真内部退化真值 |
| 源域与目标域构建 | NASA5 真实试验电池作为源域，Basilisk V1.0 灰箱仿真电池作为目标域 |
| 跨域迁移 | 在 NASA5 上学习 SOH、多时间跨度健康变化、退化速度和加速度，再迁移 TCN 时序编码参数 |
| SOH/RUL 联合预测 | 使用因果 TCN、全历史统计特征和跨电池退化进度参考同时输出 SOH 与 RUL |
| 泛化评价 | 按完整设备进行严格 OOF 划分，不让同一设备的相邻周期跨越训练集和测试集 |
| 泄漏控制 | 容量、真实 SOH、RUL、EOL、真实内阻、knee 真值和未来曲线均不作为模型输入 |
| 工程交付 | 提供冻结权重、推理接口、完整训练程序、数据合同、哈希清单和研究分支 |

## 数据来源

### 源域：NASA5

源域来自 NASA Battery Aging Data Set，使用以下五块电池：

```text
B0005 / B0018 / B0033 / B0043 / B0044
```

仓库包含五块电池的原始 MATLAB 数据、周期级标签和特征表，共整理得到 202 个有效源域周期。NASA 的容量、SOH 和未来轨迹只用于源域监督标签，不会在目标域推理时作为输入。B0030、B0042 验证电池和 B0038、B0039 sealed 电池不包含在仓库中，也没有参与最终训练。

### 目标域：Basilisk V1.0

目标域为 NASA 诊断规律校准的 Basilisk V1.0 灰箱仿真数据。仓库包含 320 台公开训练设备、24,425 个诊断周期和 2,752,050 行原始 V-I-T 曲线。

每台设备具有独立的初始容量、初始内阻、基础容量衰减率、内阻增长率、温度/负载加速程度和 knee 参数。诊断观测采用约 -2 A 放电工况，根据电池状态生成电压和温度曲线。寿命终点定义为有效容量首次不高于 1.6 Ah，即：

```text
EOL threshold: SOH <= 0.80
RUL = max(EOL EFC - current EFC, 0)
```

V1.0 是用于算法验证的工程灰箱数据，不是经过真实航天器电池硬件标定的在轨数据。80 台 validation、50 台 sealed ID、25 台温度 OOD 和 25 台负载 OOD 设备不在仓库内，也没有进入 full320 训练。

## 输入与因果约束

原始诊断曲线字段为：

```text
unit_id, time, sample_index, elapsed_s, voltage_v, current_a, temperature_c
```

项目从每个诊断周期的 V-I-T 曲线中提取候选特征，再冻结为 16 维 BHUMP 特征合同。主要信息包括电压分段斜率、单位时间电压变化率、多个电压区间的通过时间，以及相对首周期的曲线形态变化。完整特征顺序见 [model/model_manifest.json](model/model_manifest.json)。

模型在时刻 `t` 只能使用该设备从首次观测到 `t` 的历史信息。修改 `t` 之后的未来曲线不能改变当前特征、SOH 或 RUL 输出。以下字段被明确禁止作为模型输入：

```text
capacity / true_soh / rul / eol / soc / true_resistance /
knee truth / mechanism label / future measurements
```

## 技术路线

最终模型名称为 `NASA5 dynamics-adaptive TCN24+B_stats38 ensemble`，由三个独立随机种子成员等权集成：

```text
NASA5 原始退化序列
  -> 16维 BHUMP 因果曲线特征
  -> NASA 多时间跨度退化动力学预训练
  -> TCN24 局部时序表示
  -> Basilisk V1.0 目标域 SSL 与监督微调
  -> 38维 B_stats 全历史统计表示
  -> NASA 退化进度跨电池参考
  -> seed 52 / 53 / 54 等权集成
  -> SOH 与 RUL 输出
```

### 1. NASA 多任务动态预训练

TCN 使用长度为 24 个周期的因果窗口。源域预训练同时学习：

- 当前 SOH；
- `Delta SOH @ 1/4/8/16 EFC`；
- 对数退化速度；
- 退化加速度；
- 多时间跨度变化与退化速度之间的一致性。

未来 SOH 只用于构造 NASA 训练标签，模型输入仍然截止于当前周期。预训练结束后丢弃训练专用动态头，只迁移共享 TCN 编码参数，不迁移 NASA 的绝对 EOL 或 RUL 回归头。

### 2. 目标域自监督与迁移微调

Basilisk 目标适配首先使用当前训练设备进行目标域自监督，再训练 SOH/RUL 预测头。训练采用分阶段解冻：先适配新增输出头，再解冻最后一个 TCN 块，最后以更小的编码器学习率联合微调全部网络，从而减小源域参数被快速覆盖的风险。

### 3. 局部时序与全历史融合

TCN24 擅长表示最近 24 个周期内的局部变化，但早期健康状态、累计变化量和长期退化趋势可能超出局部窗口。因此项目增加 38 维因果 `B_stats`，将局部 TCN 表示与截至当前时刻的全历史统计信息融合，用于预测当前 SOH 和基础 RUL。

### 4. 跨电池退化进度参考

模型在相同退化进度节点比较目标电池与五块 NASA 电池的动态表示，形成跨电池参考预测。该分支只迁移归一化退化形状，不直接套用 NASA 的绝对寿命。当源电池支持不足或参考分歧较大时，模型降低 NASA 修正强度并回退到目标域预测。

### 5. 三种子集成

最终模型分别使用 seed 52、53、54 在全部 320 台目标训练设备上重训。推理时对三个成员的最终 RUL 和 SOH 预测等权平均，并输出成员间标准差作为训练稳定性参考。该标准差不等同于经过校准的概率置信区间。

## 严格评价方法

模型选择使用设备级严格 OOF，而不是随机拆分周期：

```text
320台目标训练设备
  -> 5个外层fold，每折64台只用于OOF测试
  -> 其余256台中再划分230台训练、26台内层选择
  -> 冻结轮数和策略后，在256台上重拟合
  -> 只对该折64台未见设备预测
```

归一化器、目标域 SSL、监督训练、B_stats、NASA 参考策略和轮数选择都不能读取外层测试设备。正式 OOF 使用 seed 52、53、54；全部结构冻结后，才执行 full320 最终重训。

## 模型效果

### 严格设备级 OOF

| 方法 | RUL MAE/EFC | RMSE/EFC | R2 | SOH MAE |
| --- | ---: | ---: | ---: | ---: |
| `target_ssl_bstats` | 4.9590 | 6.6733 | 0.9187 | 0.01153 |
| `target_reference_control` | 4.4093 | 5.8409 | 0.9377 | 0.01153 |
| `nasa_all5_uniform` | 4.4142 | 5.8999 | 0.9364 | 0.01153 |
| `nasa_adaptive` | 4.3498 | 5.7095 | 0.9405 | 0.01153 |
| **`nasa_dynamics_adaptive`** | **4.3456** | **5.7367** | **0.9399** | **0.01151** |

最终模型相对 `target_ssl_bstats` 的 RUL MAE 降低约 12.4%，说明 NASA 动态预训练、全历史表示和跨电池参考相对于较弱目标域基线具有明显作用；但相对强目标域参考控制只降低约 1.4%，不能据此宣称已经取得稳定且显著的大幅正迁移。

### 独立 validation 边界

在一次性评价的 80 台独立 validation 设备上，`target_reference_control` 的 RUL MAE 为 5.1793 EFC，NASA 动态模型为 5.4706 EFC；NASA 模型的 SOH MAE 和最差设备 RUL MAE 略好，但总体 RUL MAE 没有超过强目标域控制。因此本仓库保留完整迁移路线和真实结果，同时明确其结论是“严格 OOF 下有限正迁移，独立验证上尚未稳定超过强控制”。

## 主要创新与工程特点

- 不直接迁移 NASA 的绝对寿命，而是迁移多时间跨度退化动力学和归一化退化进度；
- 将 24 周期局部 TCN 表示与 38 维全历史因果统计量融合；
- 通过跨电池动态参考实现样本级源域信息利用，并在支持不足时回退到目标域模型；
- 使用完整设备隔离的嵌套 OOF，避免相邻周期窗口泄漏；
- 同时保留强目标域控制、目标域基础模型和 NASA 消融模型，使迁移收益可核查；
- 提供模型权重、训练入口、推理接口、数据合同、实验清单和 SHA-256 文件校验。

## 仓库结构

```text
datasets/
  contracts/                 冻结16维特征合同与泄漏审计
  source_nasa5/              NASA5原始数据、标签和特征
  target_basilisk_v10/       V1.0的320台目标训练设备
model/                       冻结三种子模型、推理程序和模型清单
training/                    预训练、迁移微调、B_stats和集成训练程序
experiments/
  long_life_alt26_v141/      ALT26->V1.4.1长寿命模型定义，尚未训练
  v15_multimechanism/        pooled、multi-expert、MLDG、GroupDRO、SWAD研究
DATASET_MANIFEST.json        数据范围与排除集合
SHA256SUMS.csv               仓库文件完整性哈希
GITHUB_UPLOAD.md             GitHub与Git LFS上传说明
```

`experiments/` 中的长寿命和 V1.5 多机理模型均保留真实实验状态。其中 ALT26->V1.4.1 尚未训练；V1.5 multi-expert 和连续迁移实验未通过晋升条件，因此不会被正式模型推理程序加载。

## 安装

仓库使用 Git LFS 保存两个大型目标域数据文件。克隆后执行：

```powershell
git lfs pull
pip install -r model\requirements.txt
```

主要依赖为 Python、NumPy、Pandas、SciPy、PyTorch 和 Matplotlib。

## 推理

推理程序可以读取原始 V-I-T 曲线长表，也可以读取已经提取好的周期级特征表：

```powershell
python model\predict_battery_rul.py `
  --input datasets\target_basilisk_v10\raw_curves\battery_train_curves.csv.gz `
  --output predictions.csv `
  --device cuda
```

无 CUDA 时使用 `--device cpu`。输出包括：

- `predicted_soh`：三成员平均 SOH；
- `predicted_rul_cycles`：三成员平均 RUL，单位 EFC；
- `rul_ensemble_std`：三个成员之间的预测标准差；
- `model_count`：集成成员数量，固定为 3。

详细字段和诊断输出见 [model/README.md](model/README.md)。

## 重新训练

完整 full320 重训命令见 [training/README.md](training/README.md)。训练流程只读取仓库中的 NASA5 和 320 台 Basilisk V1.0 训练设备，不读取 validation、sealed 或 OOD 数据。

```powershell
cd training
python train_bhump_v10_nasa_dynamics_full320.py `
  --data-root ..\datasets\target_basilisk_v10\features `
  --source-file ..\datasets\source_nasa5\features\nasa_source_rich.csv `
  --selection-run .\frozen_oof_selection `
  --reference-run .\reference_metadata `
  --output-dir ..\retrained_full320 `
  --seeds 52,53,54 `
  --device cuda
```

## 结果解释边界

- 当前结果主要证明仿真数据构建、迁移学习、严格验证和工程封装链路可运行；
- Basilisk V1.0 不是硬件标定后的真实在轨数据，未知赛题测试域可能仍存在明显域差异；
- SOH 误差较低不代表 RUL 必然准确，RUL 还受到基础寿命、退化速度和 knee 位置影响；
- 当前输出为 RUL 点估计，尚未建立删失似然、生存分布或经过校准的预测区间；
- NASA 迁移没有在独立 validation 上稳定超过强目标控制，因此不作超出实验依据的性能声明。

## 数据与许可

代码采用仓库中的 MIT License。NASA 源数据仍受其原始数据使用条款约束；Basilisk V1.0 数据为项目生成的研究性仿真数据，仅用于复现和算法研究。上传步骤见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。
