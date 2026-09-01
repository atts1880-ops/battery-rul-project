# V1.0锚定的V1.5连续迁移实验

该实验从已有的NASA5→V1.0 full320检查点开始，依次加入V1.5桥接子集和四个微域。每个新阶段继承上一阶段模型参数，但重置优化器；V1.0在所有阶段固定占50%监督权重。

## 1. 生成四个微域

生成器必须使用已安装Basilisk 2.11.0的虚拟环境：

```powershell
cd D:\basilisk-2.11.0\battery_target_domain
& D:\basilisk-2.11.0\basilisk-2.11.0\.venv\Scripts\python.exe `
  .\generate_battery_v15_microdomains.py `
  --formal --workers 4 --resume
```

输出位于`battery_target_domain/output/v1.5_incremental_microdomains/`。每个微域包含40台训练设备和8台锁定测试设备；没有sealed设备。

## 2. 提取冻结BHUMP特征

```powershell
cd D:\basilisk-2.11.0\battery_tcn_lstm_reproduction
& D:\miniconda3\envs\rul_gpu\python.exe `
  .\prepare_bhump_v15_microdomains.py `
  --mode formal
```

该步骤调用与V1.0/V1.5相同的曲线特征算法，生成252维审计表，但模型只读取V1.0冻结的16维合同。

## 3. 连续训练

```powershell
& D:\miniconda3\envs\rul_gpu\python.exe `
  .\train_bhump_v10_v15_continual.py `
  --parent-nasa-dir .\bhump_v10_nasa_dynamics_full320_runs `
  --parent-target-dir .\bhump_v10_full320_target_control_runs `
  --v15-train-units 40 `
  --microdomains knee_spectrum,thermal_load,decoupled_aging,path_nonstationary `
  --v10-anchor-weight 0.5 `
  --methods inherited_ft,inherited_mldg,inherited_mldg_groupdro `
  --seeds 52,53,54 `
  --swad-best `
  --device cuda `
  --resume
```

`--resume`只在当前阶段中断时恢复优化器；进入下一个微域时始终继承模型参数并重置优化器。正式输出包括逐阶段检查点、selection结果、最终域指标、配对bootstrap和`acceptance_report.json`。

## 4. 冒烟测试

已验证的CPU smoke命令：

```powershell
& D:\miniconda3\envs\rul_gpu\python.exe `
  .\train_bhump_v10_v15_continual.py `
  --micro-root .\bhump_transfer_v15_microdomains_smoke_data\smoke `
  --microdomains knee_spectrum `
  --methods inherited_mldg_groupdro `
  --seeds 52 --device cpu --smoke `
  --skip-final-evaluation --skip-target-control
```

完整训练前可运行：

```powershell
& D:\miniconda3\envs\rul_gpu\python.exe -m unittest `
  .\test_bhump_v10_v15_continual.py -v
```
