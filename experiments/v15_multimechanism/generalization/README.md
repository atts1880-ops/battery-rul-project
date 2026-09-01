# V1.0 + V1.5 双域泛化与 Multi-expert

状态：**SCREENING FAILED / 筛选未通过**。

实现的方法包括：

- `v10_only`、`v15_only`；
- `pooled` 与参数量匹配的 `pooled_wide_control`；
- 四类退化专家与因果软门控的 `multi_expert`；
- `pooled_target_only`、`multi_expert_target_only`，用于隔离 NASA 迁移贡献。

现有三种子严格 OOF 结果中，设备级双域宏平均 MAE 为：`pooled` 7.7612 EFC，`multi_expert` 7.9305 EFC，`multi_expert_target_only` 8.0147 EFC。Multi-expert 没有比 pooled 降低至少 2%，V1.0 误差也没有满足相对专用模型最多恶化 5% 的条件，因此没有补跑 seed 55/56，也没有访问 validation 或 sealed 数据。

`checkpoints/` 保存 7 种方法、3 个种子、5 个外层 fold 的 OOF 头部 checkpoint；它们用于复核 OOF 实验，不是 full-data 部署权重。`results/` 保存模型汇总、机理指标、专家使用率、泄漏审计和筛选结论，大体积逐点预测未复制到 GitHub 仓库。

运行前将仓库根目录 `training/` 加入 `PYTHONPATH`，并准备 V1.0、V1.5 数据目录：

```powershell
$env:PYTHONPATH = (Resolve-Path .\training)
python .\experiments\v15_multimechanism\generalization\train_bhump_v10_v15_generalization.py `
  --data-roots <V1.0数据目录>,<V1.5数据目录> `
  --device cuda --resume
```
