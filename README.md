# RAG 知识库问答系统（案例项目）

## 项目简介

一个基于 **LangChain + 阿里云百炼通义千问 + ChromaDB** 的本地知识库问答（RAG, Retrieval-Augmented Generation）案例项目。

项目解决的核心问题：大模型本身不了解你的私有知识，直接提问容易答错或"编造"。本项目把知识文档（TXT）上传后自动分块、向量化存入本地 ChromaDB；用户提问时，系统先从知识库中检索出相关的知识片段，再连同问题一起交给通义千问大模型生成回答，从而让回答"有据可依"。同时支持多轮对话，历史消息持久化在本地文件中，不同会话相互隔离。

项目定位：RAG 技术的学习/演示案例，代码结构清晰、链路完整，适合作为 RAG 项目实战参考。

## 核心功能

- **知识库更新（上传入库）**：通过 Web 页面上传 TXT 文档，自动完成 MD5 去重 → 文本分块 → Embedding 向量化 → 写入 ChromaDB 的完整流程
- **RAG 问答**：用户提问 → 向量检索召回相关知识片段 → 结合检索结果由通义千问（qwen-max）生成回答
- **多轮对话记忆**：基于 LangChain `RunnableWithMessageHistory` + 自定义本地文件历史存储，同一会话 ID 下自动保留上下文
- **会话隔离**：不同会话 ID 对应独立的本地历史文件，互不干扰
- **重复内容去重**：通过内容 MD5 校验，避免同一文档被重复向量化入库

## 技术栈

- **语言**：Python 3.10+
- **Web 框架**：Streamlit（两个页面：问答页 / 知识库更新页）
- **AI/Agent 框架**：LangChain（`langchain-core`、`langchain-community`、`langchain-chroma`、`langchain-text-splitters`）
- **大模型**：阿里云百炼通义千问（`ChatTongyi`，模型 `qwen-max`）
- **Embedding**：阿里云百炼 DashScope（`DashScopeEmbeddings`，模型 `text-embedding-v4`）
- **向量数据库**：ChromaDB（本地持久化，数据目录 `./chroma_db`）
- **对话历史存储**：本地 JSON 文件（目录 `./chat_history`，按会话 ID 一个文件）
- **其他**：`hashlib`（MD5 去重）、标准库 `json/os/datetime`

## 项目结构

```
RAG项目案例实例/
├── app_chat.py                # Streamlit 问答页面：会话设置 + 聊天界面，复用 RagService
├── app_file_uploader.py       # Streamlit 知识库更新页面：上传 TXT 文件并入库
├── rag.py                     # RAG 问答链路（RagService）：检索 -> 组装 Prompt -> 通义千问 -> 解析输出
├── knowledge_base.py          # 知识库入库服务（KnowledgeBaseService）：分块、向量化、MD5 去重、写入 ChromaDB
├── vector_stores.py           # 向量库封装（VectorStoreService）：ChromaDB 连接与检索器
├── file_history_store.py      # 本地文件型对话历史（FileChatMessageHistory，继承 BaseChatMessageHistory）
├── config_data.py             # 全局配置：模型名、分块参数、检索阈值、存储路径等
├── data/                      # 示例知识库源文档（TXT）
│   ├── 尺码推荐.txt           #   服装尺码参考
│   ├── 洗涤养护.txt           #   服装洗涤与养护指南
│   └── 颜色选择.txt           #   服装颜色选择参考
├── requirements.txt           # Python 依赖清单
├── .env.example               # 环境变量配置模板（复制为 .env 填写真实值）
└── .gitignore

# 以下目录/文件为运行时生成，已加入 .gitignore，不纳入版本管理：
# chroma_db/   —— ChromaDB 向量库本地数据
# chat_history/ —— 对话历史 JSON 文件
# md5.text      —— 上传内容 MD5 去重记录
# __pycache__/  —— Python 字节码缓存
```

核心文件作用：

| 文件 | 作用 |
| --- | --- |
| `rag.py` | RAG 核心链路：向量检索器召回文档片段 → 拼入 System Prompt → `ChatTongyi` 生成回答 → 输出解析；用 `RunnableWithMessageHistory` 注入历史消息 |
| `knowledge_base.py` | 文档入库：MD5 去重、`RecursiveCharacterTextSplitter` 分块、`DashScopeEmbeddings` 向量化、`Chroma.add_texts` 写入 |
| `vector_stores.py` | 封装 ChromaDB 连接与 `as_retriever` 检索器，统一供入库与问答复用 |
| `file_history_store.py` | 自定义 `BaseChatMessageHistory` 实现，消息以 LangChain dict 格式序列化到本地文件 |
| `config_data.py` | 集中管理模型名、分块大小/重叠、检索阈值、存储路径，改配置不动业务代码 |
| `app_chat.py` / `app_file_uploader.py` | 两个 Streamlit 页面，只做界面交互，复用上述服务 |

## 环境要求

- **Python 3.10+**
- **阿里云百炼账号**：需要开通百炼（DashScope）服务并创建 API Key（用于通义千问和向量化模型）
- 可联网访问阿里云百炼接口（`https://dashscope.aliyuncs.com`）

## 安装与运行

```bash
# 1. 克隆仓库
git clone <repository-url>
cd RAG项目案例实例

# 2. 创建并激活虚拟环境
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（任选其一）
#    方式一：复制模板并填写真实 Key（Streamlit 启动时会自动加载项目根目录的 .env）
#    cp .env.example .env     （Windows 用: copy .env.example .env）
#    方式二：在命令行导出
#    macOS/Linux:  export DASHSCOPE_API_KEY=sk-你的Key
#    Windows CMD:  set DASHSCOPE_API_KEY=sk-你的Key

# 5. 启动知识库问答页面（默认 http://localhost:8501）
streamlit run app_chat.py

# 6. （可选）另开终端启动知识库更新页面（端口自动 +1，为 8502）
streamlit run app_file_uploader.py
```

命令行自测 RAG 链路：

```bash
python rag.py
```

## 环境变量配置

| 变量名 | 说明 | 示例值 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 阿里云百炼（DashScope）API Key，用于通义千问对话与向量化模型 | `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |

将 `.env.example` 复制为 `.env` 并填入真实 Key 即可（`.env` 已被 `.gitignore` 忽略，不会提交）。

## 使用方法

1. **导入知识**：打开"知识库更新服务"页面（`app_file_uploader.py`），上传 TXT 文档（可先用 `data/` 下的示例文档）。页面会显示文件名、格式、大小，入库成功后提示 `[成功],内容已成功载入向量库`；重复上传同一内容会提示 `[跳过]内容已经被处理过`。
2. **开始问答**：打开"知识库问答服务"页面（`app_chat.py`），侧边栏可设置会话 ID（默认 `user_001`，不同 ID 历史相互隔离）。在输入框提问，例如"我体重180斤，尺码推荐"，系统会检索知识库并基于检索结果回答。
3. **多轮对话**：同一会话 ID 下的历史消息会自动保存，可连续追问（历史存储在 `./chat_history` 下对应文件）。
4. **调参**：模型名、分块大小、检索数量等配置集中在 `config_data.py`，按需修改。

## 项目效果

- **入库**：TXT 文档经分块（`chunk_size=1000`、重叠 `100`）后向量化存入本地 ChromaDB；相同内容通过 MD5 校验自动跳过，避免重复入库。
- **问答**：例如提问"我体重180斤，尺码推荐"，系统召回 `data/尺码推荐.txt` 中的相关片段，由通义千问给出"建议 3XL"等基于资料的回答；检索不到相关内容时会明确告知"无相关参考资料"，而不是凭空编造。
- **多轮记忆**：连续提问时，模型能结合此前对话上下文回答，历史记录以 JSON 形式持久化在本地文件。

## 项目特点

- **检索增强生成（RAG）**：回答严格以知识库检索结果为参考，降低大模型幻觉。
- **链路完整且简洁**：LangChain LCEL 表达式组装 `检索 → Prompt → 模型 → 解析` 的问答链，`RunnableWithMessageHistory` 一行接入多轮记忆。
- **自定义文件型历史**：继承 `BaseChatMessageHistory` 实现本地 JSON 持久化，不依赖外部数据库，会话间天然隔离。
- **MD5 去重**：入库前对内容取 MD5 并记录到 `md5.text`，重复文档自动跳过，节省向量化费用。
- **空文本防护**：对空输入、纯空白分块做过滤，避免向 DashScope 发送空文本触发 400 校验错误。
- **配置集中管理**：模型名、分块参数、检索阈值统一在 `config_data.py`，便于演示调参。
- **服务缓存**：`@st.cache_resource` 缓存 `RagService` 实例，避免 Streamlit 页面重跑时重复初始化链路。

## 注意事项

- **API Key 安全**：`DASHSCOPE_API_KEY` 是敏感信息，请通过环境变量或 `.env` 配置，切勿提交真实 Key 到仓库（`.env` 已在 `.gitignore` 中）。
- **向量库为本地数据**：`./chroma_db` 是运行时生成的本地向量库，已加入 `.gitignore`。换机器 / 重新克隆后需要重新上传文档入库。
- **对话历史为本地数据**：`./chat_history` 与 `md5.text` 均为运行时生成，已忽略，不随仓库分发。
- **模型与配额**：`qwen-max` / `text-embedding-v4` 调用会产生百炼平台的费用，请注意账号额度。
- **依赖版本**：`requirements.txt` 未锁定精确版本，如遇兼容问题请根据报错调整对应包版本。
- **配置调整**：分块大小、重叠、检索返回数量（`similarity_threshold`）等参数在 `config_data.py` 中修改。

## License

本项目暂未声明开源许可证，默认保留所有权利。如需开源发布，请自行选择合适的许可证（如 MIT、Apache-2.0）并在仓库根目录补充 LICENSE 文件。
