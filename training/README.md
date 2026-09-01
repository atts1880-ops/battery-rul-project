# 最终 V1.0 训练与复现程序

本目录补齐最终短寿命模型的完整训练依赖链。主入口为
`train_bhump_v10_nasa_dynamics_full320.py`，其运行顺序为：

1. 以 NASA5 的 SOH、多跨度 DeltaSOH、退化速度和加速度进行动态预训练；
2. 使用 Basilisk V1.0 的 320 台公开训练设备执行目标域 SSL 与监督微调；
3. 训练 38 维因果 B_stats 分支和 NASA 跨电池参考分支；
4. 使用 seed 52、53、54 的冻结选择配置重训并导出等权集成。

`frozen_oof_selection/` 仅包含严格设备级 OOF 实验已经冻结的轮数、策略和性能记录；它不含验证、sealed、OOD 或 OOF 测试设备数据。`reference_metadata/` 是最终运行清单所需的参考实验元数据。

## 环境

在交付包根目录执行：

```powershell
pip install -r model\requirements.txt
```

## 最终 Full320 重训

在 `training` 目录执行以下命令。其只读取本交付包中的 NASA5 和 Basilisk V1.0 训练数据，不读取任何 validation 或 sealed 数据。

```powershell
python train_bhump_v10_nasa_dynamics_full320.py `
  --data-root ..\datasets\target_basilisk_v10\features `
  --source-file ..\datasets\source_nasa5\features\nasa_source_rich.csv `
  --selection-run .\frozen_oof_selection `
  --reference-run .\reference_metadata `
  --output-dir ..\retrained_full320 `
  --seeds 52,53,54 `
  --device cuda
```

无 CUDA 时将 `--device cuda` 改为 `--device cpu`。中断同一轮训练后可追加 `--resume`。输出目录中的权重是重训产物；当前 `model/battery_rul_ensemble_v10.pt` 是已经冻结、可直接部署的现成集成模型。

## 程序清单

- `train_bhump_v10_nasa_dynamics_pretrain.py`：NASA5 多时间跨度退化动力学预训练。
- `train_bhump_v10_nasa_dynamics_full320.py`：最终全 320 台 V1.0 迁移微调与三种子集成入口。
- `prepare_bhump_transfer_data.py`、`prepare_bhump_v11_spectrum.py`：V1.4.1/V1.5 研究分支复用的冻结 BHUMP 曲线特征提取模块。
- `train.py`：公共 TCN 编码器和底层训练组件。
- `evaluate_bhump_v10_160_rul.py`：RUL 候选构造与交叉拟合评价工具。
- 其余 `train_bhump_*.py` 和 `bhump_common.py`：TCN24、B_stats38、跨电池参考、数据审计与训练工具依赖。

训练只使用冻结的 16 维 BHUMP 特征作为网络输入。容量、真实 SOH、RUL、EOL、真实内阻和未来曲线只可用于标签或审计，不会被接入推理输入。
