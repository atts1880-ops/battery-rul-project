# Battery RUL Final Ensemble V1.0

这是当前可直接部署的三种子集成模型。模型路径为：

`battery_rul_ensemble_v10.pt`

模型结构：NASA5动态预训练 → 24周期TCN → 38维因果B_stats → NASA退化进度参考分支 → 三种子等权RUL集成。

## 已冻结的性能依据

- 严格设备级OOF RUL MAE：4.3456 EFC
- 严格设备级OOF SOH MAE：0.01151
- full320成员：seed 52、53、54
- 每个成员使用320台Basilisk V1.0训练设备
- EOL阈值：SOH=0.80

full320训练集回放的3.5556 EFC不是泛化性能，不能代替严格OOF结果。

## 输入方式一：周期级16维特征表

CSV必须包含`unit_id`、`time`以及`model_manifest.json`列出的16个特征。允许包含额外字段，但模型只读取规定字段，不会使用容量、SOH、RUL或EOL。

```powershell
python predict_battery_rul.py `
  --input cycle_features.csv `
  --output predictions.csv `
  --device cuda
```

## 输入方式二：原始V-I-T诊断曲线长表

也可直接输入以下字段的CSV或CSV.GZ：

`unit_id,time,sample_index,elapsed_s,voltage_v,current_a,temperature_c`

程序会使用冻结的BHUMP算法提取曲线特征，并在同一设备内相对首周期计算变化特征。输入应包含该设备从首次观测到当前时刻的完整历史，否则累计B_stats会缺失早期信息。

```powershell
python predict_battery_rul.py `
  --input battery_curves.csv.gz `
  --output predictions.csv `
  --device cuda
```

没有CUDA时使用`--device cpu`，或保留默认的`--device auto`。

## 输出字段

- `unit_id`：设备编号
- `time`：当前EFC
- `predicted_soh`：三模型平均SOH
- `predicted_rul_cycles`：最终集成RUL，单位EFC
- `rul_ensemble_std`：三个模型之间的预测标准差
- `model_count`：集成成员数量，固定为3

如需查看每个种子的SOH、基础RUL和NASA分支RUL，增加`--diagnostics`。

## 环境

```powershell
pip install -r requirements.txt
```

本目录不包含NASA或Basilisk原始数据、标签、验证集、sealed数据或OOF预测。模型文件只保留权重、归一化参数和推理所需的压缩NASA参考节点。
