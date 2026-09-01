# 研究分支

本目录保存未晋升为正式提交模型的独立研究分支。仓库根目录的 NASA5 -> Basilisk V1.0 三种子集成模型仍是唯一冻结的短寿命提交模型。

| 分支 | 状态 | 用途 |
| --- | --- | --- |
| `long_life_alt26_v141/` | 未训练 | 定义 ALT26 -> Basilisk V1.4.1 的独立长寿命迁移路线 |
| `v15_multimechanism/generalization/` | 筛选未通过 | 比较 V1.0-only、V1.5-only、pooled 和 multi-expert |
| `v15_multimechanism/continual/` | 验收未通过 | 比较继承微调、MLDG、GroupDRO 和 SWAD |

这些分支用于展示模型设计、严格对照和负结果，不参与根目录最终模型的推理集成。
