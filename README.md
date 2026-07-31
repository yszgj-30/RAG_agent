# KnowledgeOps Agent：企业知识运营与合规问答智能体

基于通义千问、LangChain、LangGraph、Chroma、FastAPI 和 Streamlit 的企业知识运营智能体。项目不仅提供文档问答，还覆盖知识库管理、检索调试、Agent 轨迹、用户反馈和离线评测闭环。

## 功能概述

- 支持 TXT、PDF、Markdown 批量导入和结构感知切分
- LangGraph 对问题分类，路由到普通对话或知识库问答
- 通义千问通过原生 Tool Calling 选择检索工具，由 ToolNode 执行
- 支持多轮记忆、回答来源和异常降级
- 文档级知识库管理：统计、切块预览、单文档删除
- 混合检索：向量候选召回、字符级 BM25 排名与 RRF 融合
- Top-K 检索调试：融合排名、向量分数和检索耗时
- 结构化 Agent 轨迹：分类、工具选择、检索、生成和异常状态
- 回答级反馈：满意度、改进意见和反馈汇总
- 离线评测集：答案关键词、期望来源、通过率与平均耗时
- 同时提供 Streamlit 工作台和 FastAPI 接口

## 已验证的量化结果

在仓库内 5 份业务文档、50 条人工标注用例上，使用
`qwen3-embedding:0.6b` 与固定的 800/150 切块参数完成了可复现评测：

| 指标 | 纯向量 Top-3 | 候选 Top-6 + BM25/RRF Top-3 |
| --- | ---: | ---: |
| 答案上下文命中率@3 | 87.5%（35/40） | 100%（40/40） |
| 答案上下文 MRR@3 | 0.8458 | 0.9000 |
| 检索延迟 P50 | 297 ms | 305 ms |
| 检索延迟 P95 | 355 ms | 361 ms |

在融合检索基础上使用本地 `qwen3:4b-instruct` 完整运行 50 条用例：
答案关键词通过 40/40、来源标注 40/40、知识库外拒答 10/10，生成错误 0 条；
单机端到端延迟 P50/P95 为 34.2/48.1 秒。

上述结果只代表当前小规模固定数据集和本地单进程环境，不代表生产准确率、
吞吐量或并发性能。原始逐条结果与复现命令见
[`evaluation/README.md`](evaluation/README.md)。

## 环境要求

- Python 3.10+
- 阿里云 DashScope API Key（[获取地址](https://dashscope.aliyun.com)）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动系统

启动 Streamlit 页面：

```bash
streamlit run app.py
```

或启动 FastAPI 后端：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

启动后可访问 `http://localhost:8000/docs` 调试接口。

FastAPI 提供以下核心接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/documents` | 上传 TXT / PDF / MD 并入库 |
| `GET` | `/api/v1/documents` | 按来源查看文档和片段统计 |
| `GET` | `/api/v1/documents/{source}/chunks` | 查看文档实际切块 |
| `DELETE` | `/api/v1/documents/{source}` | 删除指定文档 |
| `GET` | `/api/v1/knowledge-base` | 查询知识库状态 |
| `DELETE` | `/api/v1/knowledge-base` | 清空知识库 |
| `POST` | `/api/v1/chat` | 智能体对话，返回分类、来源和结构化轨迹 |
| `POST` | `/api/v1/retrieval/debug` | 独立调试 Top-K 检索 |
| `POST/GET` | `/api/v1/feedback` | 写入反馈并查看反馈看板 |
| `POST` | `/api/v1/evaluations/cases` | 添加评测用例 |
| `GET` | `/api/v1/evaluations` | 查看用例、运行记录和质量指标 |
| `POST` | `/api/v1/evaluations/run` | 运行全部或指定评测用例 |
| `DELETE` | `/api/v1/sessions/{session_id}` | 清空指定会话 |

### 3. 使用流程

1. 浏览器打开 `http://localhost:8501`
2. 左侧边栏输入 DashScope API Key并连接模型
3. 上传企业文档并导入知识库
4. 在「知识库管理」检查切块结果
5. 在「检索调试」验证召回质量
6. 在「智能问答」提问并查看 Agent 轨迹和来源
7. 在「评测与反馈」建立固定测试集并重复运行

## 项目结构

```
RAGagent/
├── app.py                     # Streamlit 主程序入口
├── api_server.py              # FastAPI 后端入口
├── requirements.txt           # Python 依赖清单
├── models/
│   ├── __init__.py
│   └── tongyi_llm.py          # 通义千问自定义 LLM 适配类
├── document/
│   ├── __init__.py
│   └── processor.py           # 文档加载 / 清洗 / 切片
├── vectordb/
│   ├── __init__.py
│   └── store.py               # Chroma 向量数据库封装
├── agent/
│   ├── __init__.py
│   ├── tools.py               # 检索工具注册
│   └── core.py                # RAG 智能体核心引擎
├── memory/
│   ├── __init__.py
│   └── manager.py             # 对话记忆管理器
├── operations/
│   ├── __init__.py
│   └── store.py               # 反馈和评测 SQLite 数据仓库
├── ui/
│   ├── __init__.py
│   ├── layout.py              # 问答与侧栏组件
│   └── workspaces.py          # 管理、调试和评测工作台
├── chroma_db/                 # Chroma 本地持久化目录（自动创建）
├── README.md                  # 本文件
├── project_doc.md             # 项目说明文档
├── defense_ppt.md             # 答辩 PPT 文稿
└── demo_script.md             # 演示讲解稿
```

## 技术架构

```
用户 → Streamlit / FastAPI → LangGraph 问题分类
                              ├─ 普通对话 → 通义千问
                              └─ 知识库问答
                                   ↓
                         模型生成 tool_calls
                                   ↓
          ToolNode → 向量候选 → BM25/RRF 融合 Top-3
                                   ↓
                         基于资料生成回答
                                   ↓
                来源展示 + 结构化轨迹 + 多轮记忆

知识运营侧：

文档管理 → 切块预览 → 检索调试 → 固定评测集
     ↑                                  ↓
     └──────── 用户反馈与质量指标 ────────┘
```

## 注意事项

- 首次运行需确保网络畅通，DashScope API 需要公网访问
- 知识库数据保存在 `chroma_db/` 目录，清空知识库将删除全部向量数据
- 对话记录保存在进程内存中；Streamlit 刷新或 FastAPI 服务重启后会丢失（知识库数据保留）
- FastAPI 使用 `session_id` 隔离对话，不提供时会自动生成
- 默认模型为 `qwen3.7-plus`；FastAPI 可通过 `QWEN_MODEL` 环境变量切换模型
- 用户反馈与评测记录默认保存在 `~/.rag_agent/operations.db`
- 可通过 `RAG_AGENT_DATA_DIR` 指定运营数据目录

## 验证

```bash
python -m pytest -q
python -m pip check
```

测试覆盖 Agent 路由与 Tool Calling、FastAPI 接口、反馈存储和评测记录。

## 开源参考说明

产品信息架构参考了 RAGFlow、MaxKB 和 LangChain Agent Chat UI 的公开交互模式；本仓库未复制这些项目的源码，LangGraph、FastAPI、知识库管理、反馈与评测实现均位于本仓库。
