# KnowledgeOps Agent 评测

该目录保存可复现的 RAG 检索与回答质量评测。数据集基于仓库根目录的
5 份业务文档人工整理，不使用模型自动生成标准答案。

## 数据集

`dataset.json` 共 50 条：

- 40 条有答案问题：员工制度、报销制度、考勤制度、扫地机器人和智能摄像头各 8 条
- 10 条知识库外问题：用于测试资料不足时能否拒绝回答

每条有答案问题都包含期望来源和答案关键词组。关键词组内部为可接受的
不同写法，组与组之间必须全部命中。

## 运行

运行纯向量 Top-3 基线：

```powershell
python -m evaluation.benchmark --mode retrieval `
  --retrieval-strategy vector `
  --candidate-k 6 `
  --retrieval-k 3 `
  --hit-k 3 `
  --output evaluation/results/retrieval-vector.json `
  --report evaluation/results/retrieval-vector.md
```

运行候选 Top-6、BM25/RRF 融合输出 Top-3 的检索评测：

```powershell
python -m evaluation.benchmark --mode retrieval `
  --retrieval-strategy hybrid `
  --candidate-k 6 `
  --retrieval-k 3 `
  --hit-k 3 `
  --output evaluation/results/retrieval-hybrid.json `
  --report evaluation/results/retrieval-hybrid.md
```

使用本地 Ollama `qwen3:4b-instruct` 运行完整评测：

```powershell
python -m evaluation.benchmark --mode full `
  --retrieval-strategy hybrid `
  --candidate-k 6 `
  --retrieval-k 3 `
  --hit-k 3 `
  --llm-model qwen3:4b-instruct `
  --max-tokens 96 `
  --output evaluation/results/full-hybrid.json `
  --report evaluation/results/full-hybrid.md
```

运行前需要：

- 本地 Ollama 已安装 `qwen3-embedding:0.6b`
- 本地 Ollama 已运行并安装 `qwen3:4b-instruct`

如需复测项目默认的 DashScope Embedding：

```powershell
python -m evaluation.benchmark --mode retrieval `
  --embedding-provider dashscope `
  --embedding-model text-embedding-v2
```

此模式需要有效的 `DASHSCOPE_API_KEY`。

结果默认写入：

- `evaluation/results/latest.json`：逐条检索结果、回答与评分
- `evaluation/results/latest.md`：指标摘要和失败用例

## 指标

- Top-3 来源命中率：期望文档是否出现在前三个召回片段
- MRR@3：期望来源在前三个召回结果中的平均倒数排名
- 答案上下文命中率@3：前三个片段中是否存在同时满足正确来源和全部答案关键词组的片段
- 答案上下文 MRR@3：首个有效答案片段的平均倒数排名
- 答案关键词通过率：回答是否覆盖全部期望关键词组
- 来源标注率：回答是否明确标注期望来源文件
- 无答案拒答准确率：知识库外问题是否明确说明资料不足
- P50/P95 延迟：分别反映典型请求和慢请求耗时

所有指标都基于导出的原始记录计算，不能代表生产环境准确率或并发性能。

## 已验证结果（2026-07-31）

| 运行 | 答案上下文 Hit@3 | 答案上下文 MRR@3 | 检索 P50/P95 |
| --- | ---: | ---: | ---: |
| 纯向量 Top-3 | 87.5% | 0.8458 | 297/355 ms |
| BM25/RRF 融合 Top-3 | 100% | 0.9000 | 305/361 ms |

完整融合评测还得到：答案关键词通过率 100%（40/40）、来源标注率
100%（40/40）、无答案拒答准确率 100%（10/10）、生成错误率 0%。
本地 4B 模型端到端延迟 P50/P95 为 34.2/48.1 秒。

正式证据文件：

- `results/retrieval-vector.json` / `results/retrieval-vector.md`
- `results/retrieval-hybrid.json` / `results/retrieval-hybrid.md`
- `results/full-hybrid.json` / `results/full-hybrid.md`
