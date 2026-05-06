# NEFTune 训练增强实验设计

## 实验目的

为缓解小规模情绪陪伴数据带来的过拟合和模板化回复问题，本项目引入 NEFTune，在 LoRA-SFT 训练阶段对词嵌入加入轻微扰动，提高模型对不同情绪表达的泛化能力。

## 对比设置

| 实验 | Rank | Steps | NEFTune alpha | 说明 |
| --- | ---: | ---: | ---: | --- |
| `rank8_steps175` | 8 | 175 |  | 普通 LoRA-SFT |
| `rank8_steps175_neftune` | 8 | 175 | 5 | LoRA-SFT + NEFTune |

两组实验保持数据、rank、学习率、batch、训练步数完全一致，只改变 `neftune_noise_alpha`。

## 执行命令

NEFTune 实验默认使用项目训练流水线启动双卡 GPU：

```bash
cd .
bash scripts/run_domain_experiments.sh
```

## 报告分析角度

- 验证集 `eval_loss` 和 `perplexity`：观察加入噪声后是否仍能稳定收敛。
- 领域问答自动分数：观察共情性、安全边界和具体建议是否提升。
- 具体回复文本：重点看是否减少固定模板，如反复使用“我理解你的感受”但缺少针对性。

## 可写入报告的表述

本项目进一步引入 NEFTune 作为训练增强方法，在 LoRA 指令微调阶段向 embedding 表示加入轻微扰动。该方法不需要额外数据，适合小规模情绪陪伴数据场景。通过与普通 LoRA-SFT 在相同 rank、训练步数和数据规模下对比，可以分析 NEFTune 对回复多样性、共情表达和模板化程度的影响。
