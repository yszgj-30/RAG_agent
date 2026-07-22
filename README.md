# 企业知识库 RAG 智能问答 Agent

《AI大模型》课程期末作业 —— 基于通义千问 + LangChain + Chroma + Streamlit / FastAPI 的检索增强生成问答系统

## 功能概述

- 🔑 侧边栏配置通义千问 API 密钥，一键初始化大模型连接
- 📁 支持 TXT / PDF / Markdown 格式文档批量上传，自动构建私有知识库
- 🧠 智能体自主规划任务、调用检索工具、校验答案质量
- 💬 多轮对话记忆，支持连续追问与上下文理解
- 📎 回答附带原文参考来源，知识库外内容如实反馈
- 🎨 Streamlit 可视化界面，布局规整，适配演示录屏

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
| `GET` | `/api/v1/knowledge-base` | 查询知识库状态 |
| `DELETE` | `/api/v1/knowledge-base` | 清空知识库 |
| `POST` | `/api/v1/chat` | RAG 问答，通过 `session_id` 保持多轮记忆 |
| `DELETE` | `/api/v1/sessions/{session_id}` | 清空指定会话 |

### 3. 使用流程

1. 浏览器打开 `http://localhost:8501`
2. 左侧边栏输入 DashScope API Key，点击「初始化连接」
3. 上传企业文档（支持 TXT/PDF/MD），点击「构建知识库」
4. 在底部输入框提问，智能体自动检索知识库并回答

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
├── ui/
│   ├── __init__.py
│   └── layout.py              # Streamlit UI 组件
├── chroma_db/                 # Chroma 本地持久化目录（自动创建）
├── README.md                  # 本文件
├── project_doc.md             # 项目说明文档
├── defense_ppt.md             # 答辩 PPT 文稿
└── demo_script.md             # 演示讲解稿
```

## 技术架构

```
用户 → Streamlit UI → RAG Agent → 通义千问 LLM
                        ↓
                   检索工具 (Tools)
                        ↓
                   Chroma 向量库
                        ↓
                   文档处理层
                        ↓
               TXT / PDF / MD 文件
```

## 注意事项

- 首次运行需确保网络畅通，DashScope API 需要公网访问
- 知识库数据保存在 `chroma_db/` 目录，清空知识库将删除全部向量数据
- 对话记录仅保存在浏览器会话中，刷新页面后聊天记录会丢失（知识库数据保留）
- 推荐使用 `qwen-plus` 模型以获得最佳性价比，可修改 `app.py` 中 `model_name` 切换模型
