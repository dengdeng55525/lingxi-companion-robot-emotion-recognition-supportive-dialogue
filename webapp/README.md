# 灵犀 Web 控制台

启动：

```bash
cd .
python webapp/server.py --host 127.0.0.1 --port 7860
```

页面地址：

```text
http://127.0.0.1:7860
```

主要入口：

- 总览：GPU、数据集、adapter、报告和训练流程。
- 陪伴对话：安全检测、短期记忆、prompt 调试、手动回复写入、模型生成。
- 实验训练：启动环境检查、数据构建、正式 SFT、消融实验、DPO pipeline 和报告生成。
- 报告中心：查看领域问答、超参数分析、DPO 对齐等 Markdown 报告。
- 数据配置：查看 JSONL 样本和 YAML 配置。
- 任务日志：查看后台任务状态和实时日志。
