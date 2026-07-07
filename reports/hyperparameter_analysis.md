# 超参数、训练轮数与 LoRA Rank 消融分析

本报告汇总各实验配置的训练指标和领域问答自动评测结果。领域问答分数来自 `scripts/evaluate_domain_qa.py` 的启发式评分，用于相对比较。

## 实验结果表

| 实验 | Rank | Alpha | rsLoRA | Steps | Epochs | NEFTune | Train loss | Eval loss | PPL | Step/s | 领域分 | 危机通过率 | 状态 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rank4_steps175 | 4 | 8 |  | 175 | 1 |  | 2.5344 | 2.4442 | 11.5208 | 0.8540 | 85.0000 | 0.00% | 完成 |
| rank8_steps90 | 8 | 16 |  | 90 | 1 |  | 2.5805 | 2.4606 | 11.7114 | 0.8610 | 90.0000 | 100.00% | 完成 |
| rank8_steps175 | 8 | 16 |  | 175 | 1 |  | 2.5234 | 2.4366 | 11.4344 | 0.8580 | 86.6700 | 0.00% | 完成 |
| rank8_steps175_neftune | 8 | 16 |  | 175 | 1 | 5 | 2.5296 | 2.4371 | 11.4402 | 0.8540 | 83.3300 | 0.00% | 完成 |
| rank8_steps350 | 8 | 16 |  | 350 | 2 |  | 2.4384 | 2.4297 | 11.3556 | 0.8450 | 96.6700 | 100.00% | 完成 |
| rank16_steps175 | 16 | 32 |  | 175 | 1 |  | 2.5134 | 2.4305 | 11.3650 | 0.8520 | 86.6700 | 0.00% | 完成 |
| rank16_steps175_rslora | 16 | 32 | 是 | 175 | 1 |  | 2.5146 | 2.4233 | 11.2827 | 0.8560 | 88.3300 | 0.00% | 完成 |

## Rank 影响

在训练步数相同的 175 step 条件下比较 rank4、rank8、rank16：
- `rank4_steps175`：rank=4，eval loss=2.4442，PPL=11.5208，领域分=85.0000。
- `rank8_steps175`：rank=8，eval loss=2.4366，PPL=11.4344，领域分=86.6700。
- `rank16_steps175`：rank=16，eval loss=2.4305，PPL=11.3650，领域分=86.6700。
结论写法建议：如果 rank 增大后 eval loss 或领域分提升，说明更高 rank 提供了更强的领域适配容量；如果提升不明显或变差，则说明当前小数据集下 rank 过大可能收益有限，并可能增加训练成本。

## 训练步数/轮数影响

固定 rank=8，比较 90、175、350 step：
- `rank8_steps90`：steps=90，eval loss=2.4606，PPL=11.7114，领域分=90.0000。
- `rank8_steps175`：steps=175，eval loss=2.4366，PPL=11.4344，领域分=86.6700。
- `rank8_steps350`：steps=350，eval loss=2.4297，PPL=11.3556，领域分=96.6700。
结论写法建议：90 step 可作为欠训练参考，175 step 约等于当前数据一轮，350 step 约等于两轮。若 350 step 的训练 loss 下降但 eval loss/领域分无改善，应解释为小数据集上继续训练可能出现过拟合。

## NEFTune 影响

固定 rank=8、175 step，比较普通 LoRA-SFT 与 LoRA-SFT + NEFTune：
- `rank8_steps175`：方法=LoRA-SFT，NEFTune alpha=，eval loss=2.4366，PPL=11.4344，领域分=86.6700。
- `rank8_steps175_neftune`：方法=LoRA-SFT + NEFTune，NEFTune alpha=5，eval loss=2.4371，PPL=11.4402，领域分=83.3300。
结论写法建议：如果加入 NEFTune 后领域分、回复多样性或安全样本表现提升，可解释为 embedding 噪声缓解了小数据集微调的模板化问题；如果 eval loss 略有波动但问答质量更好，应优先结合具体回复案例分析。

## rsLoRA 影响

固定 rank=16、175 step，比较普通 LoRA 与 Rank-Stabilized LoRA：
- `rank16_steps175`：方法=LoRA，rank=16，alpha=32，eval loss=2.4305，PPL=11.3650，领域分=86.6700。
- `rank16_steps175_rslora`：方法=rsLoRA，rank=16，alpha=32，eval loss=2.4233，PPL=11.2827，领域分=88.3300。
结论写法建议：如果 rsLoRA 在 rank=16 下相比普通 LoRA 的 eval loss 更低或领域分更高，可说明秩稳定缩放改善了较高 rank 设置下的训练稳定性；如果差异不大，也可说明当前数据规模较小，rank=16 已接近收益上限。

## 推荐结论

- 领域问答自动分数最高的配置是 `rank8_steps350`，领域分 96.6700。
- 验证集 PPL 最低的配置是 `rank16_steps175_rslora`，PPL 11.2827。
- 最终报告中应同时引用自动指标和 `reports/domain_qa_eval.md` 中的具体回复文本，避免只用单一分数下结论。
