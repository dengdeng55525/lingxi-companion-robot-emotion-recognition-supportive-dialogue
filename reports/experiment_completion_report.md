# 实验完成情况说明

本文档用于说明“灵犀：基于 LoRA 微调的家庭陪伴情绪支持对话助手”目前已经完成的实验、产物路径、关键指标和实施过程中处理的问题。

## 1. 总体结论

截至当前，本项目课程要求中的核心实验已经完成：

- 已完成基础模型与数据集准备。
- 已完成 Qwen2.5-1.5B-Instruct 的 LoRA/QLoRA 指令微调。
- 已完成微调前后领域问答对比。
- 已完成 rank、训练步数/轮数的消融实验。
- 已完成 NEFTune 训练增强实验。
- 已完成 rsLoRA 秩稳定 LoRA 实验。
- 已完成二阶段 DPO 偏好对齐实验。
- 已生成对应的 Markdown/JSON 实验报告。

项目当前不输出机器人动作建议，所有训练数据、DPO 偏好数据和推理模块都聚焦于家庭陪伴、情绪支持、非诊断边界和安全回应。

## 2. 数据与基础训练

基础模型：

```text
models/Qwen2.5-1.5B-Instruct
```

当前处理后的指令数据集：

| Split | 数量 |
| --- | ---: |
| train | 2,790 |
| valid | 155 |
| test | 155 |

主要数据来源：

- SoulChatCorpus：中文共情、倾听、安慰对话。
- PsyQA 示例数据：中文心理支持问答组织方式参考。

正式 SFT/QLoRA adapter：

```text
outputs/lingxi-qwen25-1p5b-lora-30min
```

正式训练指标：

| 指标 | 数值 |
| --- | ---: |
| train_loss | 2.5233 |
| eval_loss | 2.4360 |
| perplexity | 11.4274 |

## 3. LoRA 消融实验

已完成以下实验组：

| 实验 | 目的 | 输出目录 |
| --- | --- | --- |
| rank4_steps175 | 低 rank 对照 | `outputs/experiments/rank4_steps175` |
| rank8_steps90 | 欠训练/短步数对照 | `outputs/experiments/rank8_steps90` |
| rank8_steps175 | 标准 rank8 一轮训练 | `outputs/experiments/rank8_steps175` |
| rank8_steps350 | rank8 两轮训练 | `outputs/experiments/rank8_steps350` |
| rank16_steps175 | 高 rank 对照 | `outputs/experiments/rank16_steps175` |

关键指标：

| 实验 | Train loss | Eval loss | PPL |
| --- | ---: | ---: | ---: |
| rank4_steps175 | 2.5344 | 2.4442 | 11.5208 |
| rank8_steps90 | 2.5805 | 2.4606 | 11.7114 |
| rank8_steps175 | 2.5234 | 2.4366 | 11.4344 |
| rank8_steps350 | 2.4384 | 2.4297 | 11.3556 |
| rank16_steps175 | 2.5134 | 2.4305 | 11.3650 |

实验结论：

- 90 step 可作为训练不足对照，eval loss 高于 175 step 和 350 step。
- rank8 训练 350 step 后 eval loss 进一步下降，说明在当前数据规模下第二轮训练仍有收益。
- rank16 相比 rank8 一轮训练 eval loss 更低，说明更高 rank 提供了更强的领域适配容量。

## 4. NEFTune 实验

NEFTune 用于在指令微调阶段向 embedding 加入轻微噪声，作为小数据场景下的训练增强。

实验输出：

```text
outputs/experiments/rank8_steps175_neftune
```

对照结果：

| 实验 | NEFTune alpha | Eval loss | PPL |
| --- | ---: | ---: | ---: |
| rank8_steps175 | - | 2.4366 | 11.4344 |
| rank8_steps175_neftune | 5 | 2.4371 | 11.4402 |

当前结果显示 NEFTune 与普通 rank8 一轮训练指标接近，未在自动指标上明显超过对照组。报告中可以将其作为“低成本增强尝试”，并结合生成案例分析是否降低模板化。

## 5. rsLoRA 实验

rsLoRA 用于改进不同 rank 下的缩放方式，重点观察较高 rank 时的训练稳定性和效果。

实验输出：

```text
outputs/experiments/rank16_steps175_rslora
```

对照结果：

| 实验 | Rank | rsLoRA | Eval loss | PPL |
| --- | ---: | --- | ---: | ---: |
| rank16_steps175 | 16 | 否 | 2.4305 | 11.3650 |
| rank16_steps175_rslora | 16 | 是 | 2.4233 | 11.2827 |

实验结论：

- rsLoRA 在 rank=16、175 step 条件下取得最低 PPL。
- 该结果可以支持报告中关于“rsLoRA 有助于较高 rank 设置下稳定训练和提升领域适配效果”的分析。

## 6. 微调前后领域问答对比

已完成基础模型与各 LoRA adapter 的领域问答生成对比。

输出报告：

```text
reports/domain_qa_eval.md
reports/domain_qa_eval.json
```

自动评测结果摘要：

| 模型 | 平均分 |
| --- | ---: |
| base | 81.67 |
| rank4_steps175 | 85.00 |
| rank8_steps90 | 86.67 |
| rank8_steps175 | 86.67 |
| rank8_steps175_neftune | 83.33 |
| rank8_steps350 | 93.33 |
| rank16_steps175 | 86.67 |
| rank16_steps175_rslora | 88.33 |

其中 `rank8_steps350` 在启发式领域问答评分中最高，`rank16_steps175_rslora` 在验证集 PPL 中最低。最终报告建议同时引用自动指标和 `reports/domain_qa_eval.md` 中的具体回复案例。

## 7. 二阶段 DPO 偏好对齐

已完成 DPO 偏好数据构造、DPO 训练、DPO 前后对比和对齐报告。

DPO 数据：

| Split | 数量 |
| --- | ---: |
| train | 360 |
| valid | 40 |
| total | 400 |

DPO 数据路径：

```text
data/processed/dpo_train.jsonl
data/processed/dpo_valid.jsonl
examples/dpo_sample.jsonl
```

DPO adapter 推理路径：

```text
outputs/lingxi-qwen25-1p5b-dpo/policy
```

DPO 训练指标：

| 指标 | 数值 |
| --- | ---: |
| train_loss | 0.0351 |
| eval_loss | 0.0003 |
| eval_rewards/accuracies | 1.0000 |
| eval_rewards/margins | 8.9844 |

DPO 对比报告：

```text
reports/dpo_domain_qa_eval.md
reports/dpo_domain_qa_eval.json
reports/dpo_alignment_report.md
```

DPO 领域问答自动评测摘要：

| 模型 | 平均分 | 危机样本通过率 |
| --- | ---: | ---: |
| base | 81.67 | 0.00% |
| sft | 83.33 | 0.00% |
| dpo | 90.00 | 100.00% |

实验结论：

- DPO 后领域问答平均分高于 SFT。
- DPO 在当前危机样本上给出了联系朋友或家人的安全支持信号，因此启发式危机样本通过率高于 SFT。
- DPO reward accuracy 达到 1.0，说明模型能够区分构造的 chosen/rejected 偏好样本。
- 高风险输入的最终安全处理仍主要依赖 `src/lingxi/safety.py` 的运行时安全边界模块；DPO 对安全表达有提升，但不能替代规则安全拦截。

## 8. 实施过程中修复的问题

实验过程中处理了以下关键问题：

- 修复 `train_lora.py` 在输出目录不存在时自动查找 checkpoint 报错的问题。
- 将 rank16 相关实验的 optimizer 改为 `adamw_torch`，规避 bitsandbytes 8-bit optimizer 在 rank16 下出现的 CUDA illegal memory access。
- DPO 训练阶段改为默认单 GPU 启动，并通过 `gradient_accumulation_steps=4` 保持有效 batch 不变，避免当前 TRL DPOTrainer 在 DDP + 多 adapter 场景下初始化阻塞。
- 修正 DPO 输出路径，评测时使用 `outputs/lingxi-qwen25-1p5b-dpo/policy` 作为实际 policy adapter。
- 处理 TRL DPO tokenized labels 中的 `None` mask，将其规范化为 `-100`，保证 collator 可以正常训练。

## 9. 当前可直接引用的报告文件

| 文件 | 内容 |
| --- | --- |
| `reports/domain_qa_eval.md` | base 与各 LoRA adapter 的领域问答对比 |
| `reports/hyperparameter_analysis.md` | rank、训练步数、NEFTune、rsLoRA 消融分析 |
| `reports/dpo_domain_qa_eval.md` | base、SFT、DPO 的领域问答对比 |
| `reports/dpo_alignment_report.md` | DPO 偏好对齐实验总结 |
| `reports/neftune_experiment_design.md` | NEFTune 实验设计说明 |
| `reports/rslora_experiment_design.md` | rsLoRA 实验设计说明 |
| `reports/safety_boundary_design.md` | 安全边界模块说明 |
| `reports/memory_module_design.md` | 短期情绪记忆模块说明 |

## 10. 最终判断

从文件产物和训练指标看，当前项目已经覆盖课程题目要求的核心内容：

- 选择开源基础模型：已完成。
- 构建领域指令数据集：已完成。
- 基于 LoRA/QLoRA 完成微调：已完成。
- 对比微调前后效果：已完成。
- 分析训练轮数、rank 等超参数影响：已完成。
- 加分实验 NEFTune、rsLoRA、DPO：已完成。

因此，当前实验阶段可以视为完成。后续如果要提升报告质量，主要工作不是继续训练，而是从已有报告中挑选 2-3 个典型案例，写入课程论文正文进行人工分析。
