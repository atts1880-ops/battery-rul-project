# ALT26 -> Basilisk V1.4.1 长寿命分支

状态：**UNTRAINED / 未训练**。本目录没有模型 checkpoint，也没有性能声明。

该分支面向约 202-541 EFC 的长寿命范围，源域为 12 块具有完整 EOL 的 NASA ALT26 电池，目标域为 ALT26 寿命范围匹配、退化机理多样的 Basilisk V1.4.1。模型仍采用冻结的 16 维 BHUMP 特征、TCN24、B_stats38 和退化进度跨电池参考，但寿命尺度与短寿命 V1.0 模型完全独立。

目录内容：

- `train_bhump_v14_alt26_transfer.py`：严格设备级 OOF 训练入口；
- `prepare_bhump_v14_alt26_diverse.py`：冻结特征契约的数据准备入口；
- `longlife_alt26_v14_model_config.json`：模型、数据和评价配置；
- `data_manifest/`：ALT26 来源、设备清单和字段审计；
- 其余 Markdown 文件：数据与模型交接说明。

预期比较 `target_ssl_bstats`、`target_reference_control`、`alt26_direct` 和 `alt26_progress_intercell`。只有 ALT26 候选在严格设备级 OOF 下超过两个目标域控制，才可生成正式 checkpoint 或性能声明。

本仓库没有复制约 496 MB 的 V1.4.1 rich 特征表，也没有包含 validation、sealed 或 custodian 数据。运行训练前，需要按原始准备脚本生成 `bhump_transfer_v14_alt26_diverse_data`，并通过 `--data-root` 指向该目录。训练脚本复用仓库根目录 `training/` 中的公共模型模块。
