# rsLoRA 秩稳定 LoRA 实验设计

## 实验目的

为缓解普通 LoRA 在不同 rank 设置下缩放方式带来的训练稳定性差异，本项目加入 rsLoRA（Rank-Stabilized LoRA）作为参数高效微调改进实验。rsLoRA 在保持 LoRA 参数高效优势的同时，改进较高 rank 下的缩放方式，适合作为课程项目中的方法增强点。

## 对比设置

| 实验 | 方法 | Rank | Alpha | Steps | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| `rank16_steps175` | LoRA | 16 | 32 | 175 | 普通 LoRA 高 rank 对照 |
| `rank16_steps175_rslora` | rsLoRA | 16 | 32 | 175 | 开启 `use_rslora` |

两组实验保持数据、训练步数、学习率、batch 和 target modules 完全一致，只改变 `use_rslora`。

## 执行命令

```bash
cd .
bash scripts/run_domain_experiments.sh
```

## 报告分析角度

- 验证集 `eval_loss` 和 `perplexity`：观察高 rank 下 rsLoRA 是否更稳定。
- 领域问答自动分数：观察共情性、安全边界和具体建议是否提升。
- 训练过程 loss 波动：若普通 rank16 波动较大而 rsLoRA 更平滑，可作为稳定性分析依据。

## 可写入报告的表述

本项目进一步尝试 rsLoRA 作为 LoRA 改进方法。rsLoRA 通过调整 LoRA 在不同 rank 下的缩放方式，提高较高 rank 设置下的训练稳定性。实验中保持 rank、alpha、数据和训练步数一致，仅比较普通 LoRA 与 rsLoRA 的差异，以分析秩稳定缩放是否能改善家庭陪伴情绪支持任务中的领域适配效果。
