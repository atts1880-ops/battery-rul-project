# V1.0 锚定的 V1.5 连续迁移

状态：**ACCEPTANCE FAILED / 正式验收未通过**。

该分支从同种子的 V1.0 父 checkpoint 继承参数，按固定顺序加入 V1.5 桥接域、`knee_spectrum`、`thermal_load`、`decoupled_aging` 和 `path_nonstationary`。每一阶段重置优化器，同时累计回放历史域，V1.0 始终占监督采样权重的 50%。

实现的方法包括：

- `inherited_ft`：继承微调、累计回放和 L2-SP；
- `inherited_mldg`：增加 MLDG 模拟未知域；
- `inherited_mldg_groupdro`：进一步增加 GroupDRO；
- `swad_inherited_mldg`：对选定训练轨迹执行 SWAD；
- 匹配的纯目标域连续训练控制。

内部选择阶段的最差域 MAE 为：MLDG 17.0951 EFC、普通继承微调 18.7650 EFC、MLDG+GroupDRO 18.9942 EFC。最终验收失败的主要原因是 NASA 版本没有比匹配目标控制改善至少 2%，配对 bootstrap 区间未通过，SOH MAE 也超出预设边界。因此这些权重不能替代根目录 V1.0 最终模型。

`checkpoints_stage5/` 只保留各方法在最终阶段的 seed 52/53/54 checkpoint；中间阶段 checkpoint 和大体积逐点预测未复制。`results/` 保存最终域指标、方法选择结果、bootstrap 与验收报告。

完整运行口径见 `CONTINUAL_V10_V15_README.md`。运行前需准备 V1.5 桥接域和四个微域数据，并将仓库根目录 `training/` 加入 `PYTHONPATH`。
